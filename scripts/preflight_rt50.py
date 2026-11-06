#!/usr/bin/env python3
"""Real-model preflight of the RoboTwin-50 elastic port (RT50-E0/E1/E2) on one GPU. Writes one JSON report.

--checks (any of identity,dataset,teacher,init):
  identity  teacher = the --family recipes' parent (sha, format, step, next epoch), VAE / normalization / data manifest =
            the teacher's training contract, 6,075,103 windows, latent-cache provenance (read-only open);
  dataset   elastic samples = the RoboTwin v0 trainer's RobotWindows.fields() (its cached text path) and the raw cache
            bits, bit for bit, on random windows plus episode first/last windows;
  teacher   the elastic teacher rollout (prefill + TransportPolicy.velocity(teacher=True), 10 Euler steps) = the
            deployed policy's RobotLoopWAM.infer_action (same latents, proprio, text, CPU-seeded noise, BF16 autocast);
  init      common_v1 student action = teacher action expert, shared step-condition tensors, trainable counts.
--speed-runs: for each run and each option in --options, bit equality of the loss, every logged component and every
  gradient against the eager path over --schedules and two micro-batches (the second exercises the graph copy-in).
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch

UPDATES_PER_EPOCH = 47462  # RoboTwin-50 at global batch 128


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--common-init"); p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--checks", default="identity,dataset,teacher,init")
    p.add_argument("--speed-runs", default="")
    p.add_argument("--schedules", default="flat_k1_n2,flat_k2_n5,flat_k4_n10,flat_k1_n10")
    p.add_argument("--options", default="bf16_frozen,teacher_graph,no_fill,all")
    p.add_argument("--dataset-windows", type=int, default=384); p.add_argument("--queries", type=int, default=4)
    p.add_argument("--family", default="RT50", help="recipe family: RT50 (epoch-10 teacher) or RT50-ep5 (epoch 5)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.use_deterministic_algorithms(True)
    dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
    from fastwam.datasets.transport_robotwin import TransportRobotWinDataset, WINDOWS
    from fastwam.datasets.chronoloop_robot import RobotWindows, robot_spec
    from fastwam.models.wan22.loopwam_robot import create_loopwam_robot, PARAMETERS
    from fastwam.models.wan22.transport_policy import TransportPolicy
    from fastwam.models.wan22.residual_transport import SCHEDULES
    from fastwam.models.wan22.loopwam_init import sha256_file
    from train_loopwam import ExactDistributedBatches
    from train_transport import sample_noise, load_elastic_init, bf16_frozen
    from transport_experiments import ELASTIC
    checks = a.checks.split(",") if a.checks else []
    report = dict(started=time.time(), args=vars(a), checks={})
    def check(name, ok, **detail):
        report["checks"][name] = dict(passed=bool(ok), **detail)
        print(json.dumps({name: report["checks"][name]}, default=str), flush=True)
        Path(a.out).write_text(json.dumps(report, indent=2, default=str))
        if not ok: raise SystemExit(f"preflight failed: {name}")

    teacher, vae = os.environ["ELASTIC_TEACHER"], os.environ["ELASTIC_VAE"]
    teacher_sha, vae_sha = sha256_file(teacher), sha256_file(vae)
    header = torch.load(teacher, map_location="cpu", mmap=True, weights_only=False)
    contract = header["training_state"]["contract"]
    if "identity" in checks:
        parent_ = ELASTIC[f"{a.family}-E0"]["parent"]
        check("teacher_identity", teacher_sha == parent_["sha256"]
              and all(ELASTIC[f"{a.family}-{e}"]["parent"] == parent_ for e in ("E1", "E2"))
              and header["format_version"] == parent_["format"] == "loopwam-s-robot-v1" and header["step"] == parent_["step"]
              and header["training_state"]["next_epoch"] * UPDATES_PER_EPOCH == parent_["step"] and vae_sha == contract["vae_sha256"],
              teacher_sha256=teacher_sha, step=header["step"], vae_sha256=vae_sha)
    data = TransportRobotWinDataset(os.environ["ELASTIC_MANIFEST"], os.environ["ELASTIC_LATENTS"], vae_sha)
    if "identity" in checks:
        check("data_identity", len(data) == WINDOWS and data.manifest["normalization_sha256"] == contract["normalization_sha256"]
              and data.data_manifest_sha256 == contract["data_manifest_sha256"],
              windows=len(data), normalization_sha256=data.manifest["normalization_sha256"],
              data_manifest_sha256=data.data_manifest_sha256)
    del header

    if "dataset" in checks:
        reference = RobotWindows(robot_spec("robotwin50"), cache_dir=Path(os.environ["ELASTIC_MANIFEST"]).parent, text_threads=4)
        rng = np.random.default_rng(20261106)
        starts = np.asarray([e[2] for e in reference.episodes]); ends = starts + np.asarray([e[3] for e in reference.episodes]) - 1
        indices = sorted(set(rng.integers(0, len(data), a.dataset_windows).tolist()
                             + rng.choice(starts, 32).tolist() + rng.choice(ends, 32).tolist() + [0, len(data) - 1]))
        bad = []; padded = 0
        for i in indices:
            s, f = data[i], reference.fields(i)
            raw = np.array(data.latents[i, :, :1], copy=True)
            same = (torch.equal(s["action"], f["action"]) and torch.equal(s["proprio"], f["proprio"][:1])
                    and torch.equal(s["context"], f["context"]) and torch.equal(s["context_mask"], f["context_mask"])
                    and torch.equal(s["action_is_pad"], f["action_is_pad"]) and s["sample_valid"] is True
                    and s["training_index"] == i and s["first_frame_latents"].shape == (16, 1, 48, 40)
                    and np.array_equal(s["first_frame_latents"].to(torch.bfloat16).view(torch.int16).numpy(), raw.view(np.int16)))
            padded += int(f["action_is_pad"].any())
            if not same: bad.append(i)
        dummy = data[-1]
        check("dataset_equals_robotwin_trainer_fields", not bad and dummy["sample_valid"] is False and dummy["training_index"] == -1,
              windows=len(indices), with_padding=padded, mismatches=bad[:10])
        del reference

    parent, payload = create_loopwam_robot(vae, checkpoint_path=teacher, device=str(dev), checkpoint_blocks=False)
    del payload
    parent.requires_grad_(False).eval()
    count = lambda module: sum(q.numel() for q in module.parameters())
    if "identity" in checks:
        check("parameter_counts", count(parent.mot) + count(parent.proprio_encoder) == PARAMETERS,
              base_policy=count(parent.mot) + count(parent.proprio_encoder), action_expert=count(parent.action_expert))

    if "teacher" in checks:
        policy = TransportPolicy(parent, kind="T0", scope="pc", step_cond=True, regime="fu", anchor=True,
                                 checkpoint_blocks=False).to(dev)
        rows = []
        order = next(iter(ExactDistributedBatches(len(data), a.queries, 0, 1, 7)))
        for q, index in enumerate(order):
            s = data[index]
            latents = s["first_frame_latents"][None].to(dev)
            original = parent._encode_input_image_latents_tensor
            parent._encode_input_image_latents_tensor = lambda input_image, tiled=False: latents
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    native = parent.infer_action(input_image=torch.zeros(3, 384, 320), proprio=s["proprio"][0],
                        context=s["context"], context_mask=s["context_mask"], num_inference_steps=10, seed=q,
                        rand_device="cpu")["action"]
            finally:
                parent._encode_input_image_latents_tensor = original
            z = torch.randn((1, 32, 14), generator=torch.Generator(device="cpu").manual_seed(q), dtype=torch.float32).to(dev)
            sample = dict(first_frame_latents=latents, proprio=s["proprio"][None].to(dev), context=s["context"][None].to(dev),
                          context_mask=s["context_mask"][None].to(dev))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                context, mask, keys, values, attention = policy.prefill(sample)
                condition = policy.conditioning(parent.action_expert, context, mask)
                times, deltas = parent.infer_action_scheduler.build_inference_schedule(10, dev, z.dtype, shift_override=1.)
                for j in range(10):
                    v, _ = policy.velocity(z, (times[j] / 1000).expand(1), condition, (keys, values, attention),
                                           teacher=True, native_timestep=times[j].expand(1))
                    z = z + v * deltas[j]
            x, y = native.float().cpu(), z[0].float().cpu()
            rows.append(dict(window=int(index), bit_equal=torch.equal(x, y), max_abs_diff=(x - y).abs().max().item(),
                             action_abs_mean=x.abs().mean().item()))
            print(json.dumps(rows[-1]), flush=True)
        check("teacher_rollout_equals_deployed_infer_action", all(r["bit_equal"] for r in rows), rows=rows)
        del policy; torch.cuda.empty_cache()

    def build(parent_, name, **kw):
        config = ELASTIC[name]
        model = TransportPolicy(parent_, **{k: config[k] for k in ("kind", "scope", "step_cond", "regime", "anchor")},
                                checkpoint_blocks=False, reuse_teacher_prefix=True, collect_diagnostics=True,
                                end_prefix=config["end_prefix"], gripper_loss=config["gripper_loss"], **kw).to(dev)
        load_elastic_init(model, config, a.common_init, teacher_sha)
        return model.train()

    if "init" in checks:
        reference = build(parent, f"{a.family}-E0").initial_state()
        init_ok = all(torch.equal(reference["student_action"][k], v.cpu()) for k, v in parent.action_expert.state_dict().items())
        trainable = {}; parts = {}
        for name in (f"{a.family}-E0", f"{a.family}-E1", f"{a.family}-E2"):
            m = build(parent, name); trainable[name] = sum(q.numel() for q in m.parameters() if q.requires_grad)
            state = m.initial_state()
            init_ok &= all(torch.equal(state[part][k], reference[part][k]) for part in ("student_action", "step_condition") for k in reference[part])
            parts[name] = dict(step_condition=count(m.step_condition), transport=sum(q.numel() for q in m.transport.parameters() if q.requires_grad))
            del m
        action = count(parent.action_expert)
        e0, e1, e2 = (f"{a.family}-{e}" for e in ("E0", "E1", "E2"))
        counts_ok = (trainable[e0] == trainable[e1] == action + parts[e0]["step_condition"]
                     and trainable[e2] == trainable[e0] + parts[e2]["transport"])
        artifact = json.loads(Path(a.common_init).with_suffix(".json").read_text())
        check("common_initialization_and_trainable_counts", init_ok and counts_ok and artifact["teacher_sha256"] == teacher_sha,
              trainable=trainable, action_expert=action, parts=parts, init_tensor_sha256=artifact["tensor_sha256"])
        torch.cuda.empty_cache()

    def batch(index):
        batches_ = iter(ExactDistributedBatches(len(data), a.microbatch, 0, 1, 42))
        for _ in range(index + 1): order = next(batches_)
        b = torch.utils.data.default_collate([data[i] for i in order])
        noise, anoise, atau = (x.to(dev) for x in sample_noise(b["training_index"].tolist(), 0, 42, dim=14))
        sample = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
        sample.update(action_scale=data.transform.normalizer["action"].scale.to(dev),
                      action_offset=data.transform.normalizer["action"].offset.to(dev))
        return sample, noise, anoise, atau
    def step(model, schedule, data_):
        sample, noise, anoise, atau = data_
        model.zero_grad(set_to_none=True); model.teacher_passes = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, logs = model(sample, SCHEDULES[schedule], noise, anoise, atau)
        loss.backward()
        grads = {n: None if q.grad is None else q.grad.detach().clone() for n, q in model.named_parameters() if q.requires_grad}
        return loss.detach(), {k: v.detach() for k, v in logs.items()}, grads, model.teacher_passes
    def compare(x, y):
        loss_a, logs_a, grads_a, passes_a = x; loss_b, logs_b, grads_b, passes_b = y
        diff = max([(g.float() - grads_b[k].float()).abs().max().item() for k, g in grads_a.items()
                    if g is not None and grads_b[k] is not None] or [0.])
        equal = (torch.equal(loss_a, loss_b) and logs_a.keys() == logs_b.keys() and all(torch.equal(logs_a[k], logs_b[k]) for k in logs_a)
                 and all((grads_a[k] is None and grads_b[k] is None) or torch.equal(grads_a[k], grads_b[k]) for k in grads_a))
        return dict(bit_equal=equal, passes=(passes_a, passes_b), loss=(float(loss_a), float(loss_b)), max_grad_abs_diff=diff,
                    grip=float(logs_a["grip"]))

    runs = [r for r in a.speed_runs.split(",") if r]
    if runs:
        batches = [batch(0), batch(1)]
        schedules = a.schedules.split(",")
        options = dict(bf16_frozen=dict(bf16=True, graph=False, nofill=False), teacher_graph=dict(bf16=False, graph=True, nofill=False),
                       no_fill=dict(bf16=False, graph=False, nofill=True), all=dict(bf16=True, graph=True, nofill=True))
        options = {k: v for k, v in options.items() if k in a.options.split(",")}
        for run in runs:
            eager = build(parent, run)
            expected = {(s, i): step(eager, s, batches[i]) for s in schedules for i in (0, 1)}
            finite = all(torch.isfinite(v[0]).item() for v in expected.values())
            grip_zero = all(v[1]["grip"].item() == 0 for v in expected.values())
            del eager; torch.cuda.empty_cache()
            for option, flags in options.items():
                torch.utils.deterministic.fill_uninitialized_memory = not flags["nofill"]
                fast_parent, _ = create_loopwam_robot(vae, checkpoint_path=teacher, device=str(dev), checkpoint_blocks=False)
                fast_parent.requires_grad_(False).eval()
                fast = build(fast_parent, run, teacher_graph=flags["graph"])
                converted = bf16_frozen(fast) if flags["bf16"] else 0
                rows = {}
                for s in schedules:
                    for i in (0, 1):
                        rows[f"{s}/batch{i}"] = compare(expected[(s, i)], step(fast, s, batches[i]))
                graph = getattr(fast, "_teacher_graph", None)
                check(f"speed_{option}_{run}", finite and grip_zero and all(r["bit_equal"] and r["passes"][0] == r["passes"][1] for r in rows.values()),
                      converted_linear=converted, graph_captures=None if graph is None else graph.captures, paths=rows)
                del fast, fast_parent, graph; torch.cuda.empty_cache()
                torch.utils.deterministic.fill_uninitialized_memory = True
            del expected
    report.update(finished=time.time(), passed=True)
    Path(a.out).write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
