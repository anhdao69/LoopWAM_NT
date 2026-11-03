#!/usr/bin/env python3
"""Training-throughput probe of the CL-0 elastic student on one GPU (real data and optimizer step).

For each speed option set and micro-batch size, times one full micro-batch (forward, backward, fused AdamW) on
every FLAT9 configuration. A configuration whose block calls reach --ckpt-min-blocks uses block checkpointing
(the trainer's adaptive rule); OOM rows are reported, not fatal. The projected update time for a world size W is
seconds_per_microbatch * 128 / (W * microbatch), averaged uniformly over FLAT9 (the balanced sampler's mix).
"""
import argparse, json, os, time
import torch

p = argparse.ArgumentParser()
p.add_argument("--run", default="CL0-E0"); p.add_argument("--common-init", required=True)
p.add_argument("--microbatch", type=int, default=32); p.add_argument("--ckpt-min-blocks", type=int, default=10**9)
p.add_argument("--options", default="eager,bf16,bf16+graph"); p.add_argument("--schedules", default="all")
p.add_argument("--iters", type=int, default=3); p.add_argument("--out", required=True)
p.add_argument("--profile", help="schedule to profile once per option with torch.profiler (after the timed iterations)")
a = p.parse_args()
torch.use_deterministic_algorithms(True)
from fastwam.datasets.transport_cached import TransportCachedDataset
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.models.wan22.transport_policy import TransportPolicy
from fastwam.models.wan22.residual_transport import SCHEDULES, FLAT9
from fastwam.models.wan22.loopwam_init import sha256_file
from train_loopwam import ExactDistributedBatches
from train_transport import sample_noise, load_elastic_init, bf16_frozen
from transport_experiments import ELASTIC
from elastic_schedules import block_calls
dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
data = TransportCachedDataset(os.environ["ELASTIC_MANIFEST"], os.environ["ELASTIC_LATENTS"])
order = next(iter(ExactDistributedBatches(len(data), a.microbatch, 0, 1, 42)))
batch = torch.utils.data.default_collate([data[i] for i in order])
noise, anoise, atau = (x.to(dev) for x in sample_noise(batch["training_index"].tolist(), 0, 42))
sample = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
sample.update(action_scale=data.transform.normalizer["action"].scale.to(dev),
              action_offset=data.transform.normalizer["action"].offset.to(dev))
teacher = os.environ["ELASTIC_TEACHER"]; teacher_sha = sha256_file(teacher)
config = ELASTIC[a.run]
schedules = list(FLAT9) if a.schedules == "all" else a.schedules.split(",")
results = dict(args=vars(a), gpu=torch.cuda.get_device_name(), torch=torch.__version__, options={})
for option in a.options.split(","):
    torch.utils.deterministic.fill_uninitialized_memory = "nofill" not in option
    parent = create_loopwam(checkpoint_path=teacher, vae_path=os.environ["ELASTIC_VAE"], device=str(dev)).requires_grad_(False).eval()
    model = TransportPolicy(parent, **{k: config[k] for k in ("kind", "scope", "step_cond", "regime", "anchor")},
                            checkpoint_blocks=False, reuse_teacher_prefix=True, collect_diagnostics=True,
                            teacher_graph="graph" in option).to(dev)
    load_elastic_init(model, config, a.common_init, teacher_sha)
    if "bf16" in option: bf16_frozen(model)
    opt = torch.optim.AdamW(model.optimizer_groups(), betas=(.9, .95), eps=1e-8, fused=True)
    model.train(); rows = []
    for name in schedules:
        sched = SCHEDULES[name]; model.checkpoint_blocks = block_calls(sched) >= a.ckpt_min_blocks
        times = []
        try:
            torch.cuda.reset_peak_memory_stats()
            for _ in range(a.iters):
                torch.cuda.synchronize(); t = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _logs = model(sample, sched, noise, anoise, atau)
                (loss / a.microbatch).backward()
                opt.step(); opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); times.append(time.perf_counter() - t)
            row = dict(schedule=name, checkpoint=model.checkpoint_blocks, seconds=times, best=min(times[1:] or times),
                       peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        except torch.OutOfMemoryError:
            row = dict(schedule=name, checkpoint=model.checkpoint_blocks, oom=True)
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        if a.profile == name and not row.get("oom"):
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _logs = model(sample, sched, noise, anoise, atau)
                (loss / a.microbatch).backward(); opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
            events = prof.key_averages()
            gpu = sum(e.self_device_time_total for e in events) / 1e6
            launches = sum(e.count for e in events if e.key.startswith(("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC")))
            row["profile"] = dict(gpu_busy_seconds=gpu, kernel_launches=launches, wall_seconds=row["best"],
                top_gpu=[(e.key[:90], e.count, round(e.self_device_time_total / 1e3, 2)) for e in
                         sorted(events, key=lambda e: -e.self_device_time_total)[:25]],
                top_cpu=[(e.key[:90], e.count, round(e.self_cpu_time_total / 1e3, 2)) for e in
                         sorted(events, key=lambda e: -e.self_cpu_time_total)[:25]])
            open(a.out + f".{option}.profile.txt", "w").write(events.table(sort_by="self_cuda_time_total", row_limit=60))
        rows.append(row); print(json.dumps(dict(option=option, **row)), flush=True)
    ok = [r for r in rows if not r.get("oom")]
    mean = sum(r["best"] for r in ok) / len(ok) if len(ok) == len(rows) else None
    results["options"][option] = dict(rows=rows, mean_seconds_per_microbatch=mean,
        projected_seconds_per_update={w: None if mean is None else mean * 128 / (w * a.microbatch) for w in (1, 2, 4)
                                      if 128 % (w * a.microbatch) == 0},
        max_peak_gb=max([r["peak_gb"] for r in ok] or [0]))
    print(json.dumps(dict(option=option, **{k: v for k, v in results["options"][option].items() if k != "rows"})), flush=True)
    del model, parent, opt; torch.cuda.empty_cache()
    open(a.out, "w").write(json.dumps(results, indent=2))
