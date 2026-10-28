#!/usr/bin/env python3
"""Throughput/memory probe of the RT TransportPolicy on real full-LIBERO data (no training state)."""
import argparse, json, os, time, math
import torch
p = argparse.ArgumentParser()
p.add_argument("--scope", default="pc"); p.add_argument("--kind", default="T0")
p.add_argument("--microbatch", type=int, default=64); p.add_argument("--ckpt", type=int, default=0)
p.add_argument("--schedules", default="1,1;1x10;4x10"); p.add_argument("--iters", type=int, default=3)
p.add_argument("--out", required=True)
a = p.parse_args()
torch.use_deterministic_algorithms(True)
from fastwam.datasets.transport_cached import TransportCachedDataset
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.models.wan22.transport_policy import TransportPolicy
from train_loopwam import ExactDistributedBatches
from train_transport import sample_noise
dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
torch.manual_seed(42)
data = TransportCachedDataset(os.environ["ELASTIC_MANIFEST"], os.environ["ELASTIC_LATENTS"])
t = time.time(); order = next(iter(ExactDistributedBatches(len(data), a.microbatch, 0, 1, 42)))
batch = torch.utils.data.default_collate([data[i] for i in order]); load_s = time.time() - t
parent = create_loopwam(checkpoint_path=os.environ["ELASTIC_TEACHER"], vae_path=os.environ["ELASTIC_VAE"], device=str(dev))
parent.requires_grad_(False).eval()
model = TransportPolicy(parent, kind=a.kind, scope=a.scope, step_cond=True, checkpoint_blocks=bool(a.ckpt)).to(dev)
opt = torch.optim.AdamW(model.optimizer_groups(), betas=(.9, .95), eps=1e-8, fused=True)
model.train()
scale = data.transform.normalizer["action"].scale.to(dev); offset = data.transform.normalizer["action"].offset.to(dev)
noise, anoise, atau = sample_noise(batch["training_index"].tolist(), 0, 42)
sample = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in batch.items()}
sample.update(action_scale=scale, action_offset=offset)
def parse(s):
    if "x" in s: k, n = s.split("x"); return (int(k),) * int(n)
    return tuple(int(x) for x in s.split(","))
results = dict(args=vars(a), load_seconds_per_microbatch=load_s, gpu=torch.cuda.get_device_name(), torch=torch.__version__, rows=[])
for spec in a.schedules.split(";"):
    sched = parse(spec); times = []
    try:
        torch.cuda.reset_peak_memory_stats()
        for it in range(a.iters):
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, comp = model(sample, sched, noise.to(dev), anoise.to(dev), atau.to(dev))
            (loss / a.microbatch).backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        row = dict(schedule=spec, seconds_per_microbatch=times, peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                   est_seconds_per_update128=min(times[1:] or times) * 128 / a.microbatch,
                   loss=float(loss) / a.microbatch, comps={k: float(v) / a.microbatch for k, v in comp.items()})
    except torch.OutOfMemoryError as e:
        row = dict(schedule=spec, oom=True, peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
    results["rows"].append(row); print(json.dumps(row), flush=True)
open(a.out, "w").write(json.dumps(results, indent=2))
