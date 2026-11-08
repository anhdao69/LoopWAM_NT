#!/usr/bin/env python3
"""Verify RoboCasa-GR1 elastic exports (loopwam-s-gr1-rt-v1) against the recovery state they were written from.

Port of scripts/verify_rt50_export.py (RoboTwin-50) with the GR1 teacher format, 29-D actions and the GR1 dataset; one
addition: --server-schedules runs the elastic policy server class (experiments/robocasa_gr1/elastic_policy_server.py,
the eval server) on full observations (frames through the VAE) and requires its 32x29 chunk to equal the de-normalized
loader rollout of the same export on the same latents/proprio/text/noise.

For every --pairs EXPORT:RECOVERY (both in --run-dir) it rebuilds the trainer's in-memory model from RECOVERY exactly as
train_transport.py does on --resume (teacher parent, TransportPolicy of the recipe, common init, recovery weights,
--bf16-frozen when the contract used it) and checks:
- static: format names; loop/architecture metadata equal the teacher's; every frozen mot tensor (outside
  mixtures.action.*) and the proprio encoder bit-equal (key set, dtype, shape, value) to the FP32 teacher file;
  mot["mixtures.action.*"] == student_action == the recovery weights; step_condition / trainable transport == recovery;
  the student differs from the teacher once update > 0; transport_config / contract / epoch / update / windows_seen;
  parent_training_state == the teacher's training_state;
- functional: the export loaded through transport_inference_gr1.load_gr1_transport rolls out bit-identically to the
  rebuilt training model on real RoboCasa-GR1 windows for every FLAT9 schedule (BF16 autocast, CPU-seeded noise).
--write-export early|epoch first writes an export from the first pair's RECOVERY with the trainer's own export_epoch
(pre-launch test; that pair's EXPORT name is replaced by the written file, step_00002000.pt or epoch_EE.pt). From a
mid-epoch recovery the epoch-end condition is then not required. --static-only skips the GPU parts (no CUDA use).
--padding-check also runs a training forward/backward on an all-padding microbatch (the last update of a 4-GPU epoch
gives rank 3 one): the loss must be exactly 0 with finite, all-zero gradients.
Environment: plans/elastic_gr1/env.sh (ELASTIC_TEACHER, ELASTIC_VAE, ELASTIC_MANIFEST, ELASTIC_LATENTS, ELASTIC_INIT).
"""
import argparse, json, os, time
from pathlib import Path
import torch

FLAT9 = ["flat_k1_n2", "flat_k1_n5", "flat_k1_n10", "flat_k2_n2", "flat_k2_n5", "flat_k2_n10",
         "flat_k4_n2", "flat_k4_n5", "flat_k4_n10"]


def same_tensors(a, b):
    if set(a) != set(b): return False, f"key sets differ ({len(set(a) ^ set(b))} keys)"
    bad = [k for k in a if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or not torch.equal(a[k], b[k])]
    return not bad, (f"{len(bad)} tensors differ, e.g. {bad[:3]}" if bad else f"{len(a)} tensors bit-equal")


def deep_equal(x, y):
    if torch.is_tensor(x) or torch.is_tensor(y):
        return torch.is_tensor(x) and torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(deep_equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return type(x) == type(y) and len(x) == len(y) and all(deep_equal(p, q) for p, q in zip(x, y))
    return x == y


def server_check(export, policy, schedules, queries=3):
    """The eval server class (full observation -> chunk) vs the loader rollout of the same export, de-normalized."""
    import numpy as np
    from experiments.robocasa_gr1.loopwam_policy_server import STATE_KEYS, VIDEO_KEY, TEXT_KEY
    from experiments.robocasa_gr1.elastic_policy_server import LoopWAMGR1ElasticPolicy
    from fastwam.datasets.lerobot.transforms import gr1 as gr1_select
    from fastwam.datasets.loopwam_long import DEFAULT_PROMPT
    from fastwam.datasets.robocasa_gr1 import preprocess_frames_gr1
    from fastwam.models.wan22.residual_transport import SCHEDULES
    stats = str(Path(os.environ["ELASTIC_MANIFEST"]).parent / "dataset_stats_29d.json")
    rng = np.random.default_rng(1108); rows = []
    prompts = json.loads(Path(os.environ["ELASTIC_MANIFEST"]).read_text())["prompts"]
    lo_hi = json.loads(Path(stats).read_text())["state"]["default"]
    for s in schedules:
        server = LoopWAMGR1ElasticPolicy(str(export), stats, os.environ["ELASTIC_VAE"], s, seed=42, n_action_steps=12, device="cuda")
        p = server.model.parent
        for q in range(queries):
            frame = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
            state = rng.uniform(np.asarray(lo_hi["global_min"]), np.asarray(lo_hi["global_max"])).astype(np.float32)
            text = prompts[int(rng.integers(0, len(prompts)))]
            obs = {VIDEO_KEY: frame[None, None], TEXT_KEY: [text]}
            for k in STATE_KEYS:
                lo, hi = gr1_select.OUTPUT_SLICES[k[len("state."):]]
                obs[k] = state[None, None, lo:hi]
            got = server.predict_action(observation=obs, n_action_steps=12)["action_chunk"][0]
            # Independent construction: the frozen-VAE latents of the preprocessed frame, the normalized state, the cached
            # text context, the CPU-seeded noise, then load_gr1_transport's rollout and the action de-normalization.
            image = preprocess_frames_gr1(torch.from_numpy(frame).permute(2, 0, 1)[None])[0]
            context, mask = server.text(DEFAULT_PROMPT.format(task=text))
            proprio = server.normalizer["state"].forward(torch.from_numpy(state)[None])[0]
            noise = torch.randn((1, 32, 29), generator=torch.Generator(device="cpu").manual_seed(42), dtype=torch.float32).to(p.device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                latents = policy.parent._encode_input_image_latents_tensor(input_image=image[None].to(p.device, torch.float32))
                sample = dict(first_frame_latents=latents, proprio=proprio[None, None].to(p.device), context=context[None].to(p.device),
                              context_mask=mask[None].to(p.device))
                z = policy.rollout(sample, SCHEDULES[s], noise)
            want = server.normalizer["action"].backward(z[0].float().cpu()[None])[0].numpy().astype(np.float32)
            rows.append(dict(schedule=s, query=q, bit_equal=bool(np.array_equal(got, want)), finite=bool(np.isfinite(got).all()),
                             max_abs_diff=float(np.abs(got - want).max()), nfe=server.metadata["num_inference_steps"],
                             loops=server.metadata["loops"]))
            print(json.dumps(rows[-1]), flush=True)
        del server; torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--pairs", required=True, help="EXPORT:RECOVERY file names in --run-dir, comma separated")
    p.add_argument("--write-export", choices=("early", "epoch"))
    p.add_argument("--padding-check", action="store_true")
    p.add_argument("--static-only", action="store_true")
    p.add_argument("--windows", type=int, default=3)
    p.add_argument("--server-schedules", default="", help="comma list of schedules for the eval-server check (first pair)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.static_only and (a.write_export or a.padding_check): p.error("--static-only cannot write or run the padding check")
    torch.use_deterministic_algorithms(True)
    dev = torch.device("cuda", 0)
    if not a.static_only: torch.cuda.set_device(dev)
    from fastwam.datasets.transport_gr1 import TransportGR1Dataset
    from fastwam.models.wan22.loopwam_gr1 import create_loopwam_gr1, FORMAT
    from fastwam.models.wan22.transport_policy import TransportPolicy
    from fastwam.models.wan22.transport_inference_gr1 import load_gr1_transport, EXPORT_FORMAT
    from fastwam.models.wan22.residual_transport import SCHEDULES
    from fastwam.models.wan22.loopwam_init import sha256_file
    from train_loopwam import ExactDistributedBatches
    from train_transport import load_elastic_init, bf16_frozen, export_epoch, sample_noise, GR1_METADATA
    run = Path(a.run_dir)
    report = dict(started=time.time(), args=vars(a), pairs={}, passed=False)
    def save(): Path(a.out).write_text(json.dumps(report, indent=2, default=str))

    teacher_path, vae = os.environ["ELASTIC_TEACHER"], os.environ["ELASTIC_VAE"]
    teacher_sha = sha256_file(teacher_path)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=False, mmap=True)
    if not a.static_only:
        data = TransportGR1Dataset(os.environ["ELASTIC_MANIFEST"], os.environ["ELASTIC_LATENTS"], sha256_file(vae))
        scale = data.transform.normalizer["action"].scale.to(dev); offset = data.transform.normalizer["action"].offset.to(dev)
    pairs = [tuple(x.split(":")) for x in a.pairs.split(",")]
    A = "mixtures.action."

    def rebuild(recovery):
        """The trainer's model on --resume from this recovery file."""
        contract = recovery["contract"]; config = contract["config"]
        if contract["asset_sha256"]["teacher"] != teacher_sha or contract["teacher"] != str(Path(teacher_path).resolve()):
            raise ValueError("Recovery was trained from another teacher")
        parent, _ = create_loopwam_gr1(vae, checkpoint_path=teacher_path, device=str(dev), checkpoint_blocks=False)
        parent.requires_grad_(False).eval()
        model = TransportPolicy(parent, **{k: config[k] for k in ("kind", "scope", "step_cond", "regime", "anchor")},
            checkpoint_blocks=contract["checkpoint_blocks"], loss_weights=config.get("loss_weights"),
            reuse_teacher_prefix=config.get("reuse_teacher_prefix", False), collect_diagnostics=config.get("collect_diagnostics", False),
            teacher_graph=False, end_prefix=config.get("end_prefix", 10), gripper_loss=config.get("gripper_loss", True)).to(dev)
        load_elastic_init(model, config, os.environ["ELASTIC_INIT"], teacher_sha)
        if contract.get("speed_options", {}).get("bf16_frozen"): bf16_frozen(model)
        names = {n for n, q in model.named_parameters() if q.requires_grad}
        if names != set(recovery["weights"]): raise ValueError("Recovery weights do not match the recipe's trainable set")
        with torch.no_grad():
            for n, q in model.named_parameters():
                if q.requires_grad: q.copy_(recovery["weights"][n])
        return model

    def samples(n):
        order = next(iter(ExactDistributedBatches(len(data), n, 0, 1, 20261106)))
        out = []
        for q, i in enumerate(order):
            s = data[i]
            out.append((int(i), dict(first_frame_latents=s["first_frame_latents"][None].to(dev), proprio=s["proprio"][None].to(dev),
                context=s["context"][None].to(dev), context_mask=s["context_mask"][None].to(dev)),
                torch.randn((1, 32, 29), generator=torch.Generator(device="cpu").manual_seed(1000 + q), dtype=torch.float32).to(dev)))
        return out
    windows = None if a.static_only else samples(a.windows)

    all_ok = True
    for index, (export_name, recovery_name) in enumerate(pairs):
        checks = {}
        recovery = torch.load(run / recovery_name, map_location="cpu", weights_only=False)
        contract, config = recovery["contract"], recovery["contract"]["config"]
        model = None if a.static_only else rebuild(recovery)
        written = index == 0 and a.write_export is not None
        if written:
            target = run / ("step_00002000.pt" if a.write_export == "early" else f"epoch_{recovery['epoch']:02d}.pt")
            if target.exists(): raise ValueError(f"Refusing to overwrite {target}")
            started = time.perf_counter()
            path = export_epoch(model, run, recovery["epoch"], config, recovery["update"], contract, early=a.write_export == "early")
            if Path(path) != target: raise ValueError(f"export_epoch wrote {path}, expected {target}")
            export_name = target.name
            report["export_seconds"] = time.perf_counter() - started
            print(json.dumps(dict(event="exported", file=export_name, seconds=report["export_seconds"])), flush=True)
        x = torch.load(run / export_name, map_location="cpu", weights_only=False, mmap=True)
        st = x.get("training_state", {})
        early = export_name.startswith("step_")
        checks["format"] = x.get("format_version") == EXPORT_FORMAT and x.get("base_format_version") == FORMAT \
            and x.get("inference_action_format") == "unmerged-exact"
        checks["metadata_equal_teacher"] = all(deep_equal(x[k], teacher[k]) for k in GR1_METADATA)
        frozen = {k: v for k, v in x["mot"].items() if not k.startswith(A)}
        tfrozen = {k: v for k, v in teacher["mot"].items() if not k.startswith(A)}
        checks["frozen_mot_equal_fp32_teacher"], why_frozen = same_tensors(frozen, tfrozen)
        checks["proprio_equal_teacher"], _ = same_tensors(dict(x["proprio_encoder"]), dict(teacher["proprio_encoder"]))
        checks["frozen_all_fp32"] = all(v.dtype == torch.float32 for v in frozen.values() if v.is_floating_point())
        sa = x["student_action"]
        action = {k[len(A):]: v for k, v in x["mot"].items() if k.startswith(A)}
        taction = {k[len(A):]: v for k, v in teacher["mot"].items() if k.startswith(A)}
        checks["mot_action_equals_student"], _ = same_tensors(action, sa)
        student_w = {k[len("student." + A):]: v for k, v in recovery["weights"].items() if k.startswith("student." + A)}
        checks["student_equals_recovery"] = set(student_w) == set(sa) and all(torch.equal(sa[k], student_w[k]) for k in sa)
        checks["student_fp32"] = all(v.dtype == torch.float32 for v in sa.values() if v.is_floating_point())
        changed = sum(not torch.equal(sa[k], taction[k]) for k in sa)
        checks["student_trained"] = changed > 0 if recovery["update"] > 0 else True
        sc = x["step_condition"]
        sc_w = {k[len("step_condition."):]: v for k, v in recovery["weights"].items() if k.startswith("step_condition.")}
        checks["step_condition_equals_recovery"] = sc is not None and config["step_cond"] and set(sc) == set(sc_w) \
            and all(torch.equal(sc[k], sc_w[k]) for k in sc)
        tr_w = {k[len("transport."):]: v for k, v in recovery["weights"].items() if k.startswith("transport.")}
        checks["transport_trainable_equals_recovery"] = set(tr_w) <= set(x["transport"]) \
            and all(torch.equal(x["transport"][k], tr_w[k]) for k in tr_w) and (bool(tr_w) == (config["kind"] == "T4"))
        checks["transport_config"] = deep_equal(x["transport_config"], config) and x["transport_config"]["dataset"] == "robocasa_gr1" \
            and config["parent"]["sha256"] == teacher_sha
        windows_seen = recovery["update"] * contract["global_batch"] if early else recovery["epoch"] * contract["windows"]
        checks["training_state"] = (deep_equal(st.get("contract"), contract) and st.get("update") == recovery["update"] == x["step"]
            and st.get("windows_seen") == windows_seen and st.get("merged_lora") is False
            and (early or st.get("epoch") == recovery["epoch"])
            and st.get("parent_step") == teacher["step"] and deep_equal(st.get("parent_training_state"), teacher["training_state"]))
        if not early and not written:
            checks["recovery_is_epoch_end"] = recovery["next_micro"] == 0 and recovery["epoch"] >= 1 \
                and export_name == f"epoch_{recovery['epoch']:02d}.pt"
        if early and not written:
            checks["recovery_is_update_2000"] = recovery["update"] == 2000 and export_name == "step_00002000.pt"
        rows = []
        if a.static_only:
            report["pairs"][export_name] = dict(recovery=recovery_name, passed=all(checks.values()), checks=checks,
                frozen=why_frozen, student_tensors_changed=f"{changed}/{len(sa)}", update=recovery["update"],
                epoch=recovery["epoch"], kind=config["kind"], static_only=True, export_sha256=sha256_file(run / export_name))
            all_ok &= all(checks.values())
            print(json.dumps({export_name: dict(passed=all(checks.values()), checks=checks)}), flush=True)
            save(); del recovery, x
            continue
        # Functional: export through the loader vs the rebuilt training model.
        policy, _ = load_gr1_transport(run / export_name, vae, device=dev)
        model.eval()
        for w, sample, noise in windows:
            for s in FLAT9:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    y_train = model.rollout(sample, SCHEDULES[s], noise.clone())
                    y_export = policy.rollout(sample, SCHEDULES[s], noise.clone())
                rows.append(dict(window=w, schedule=s, bit_equal=torch.equal(y_train, y_export),
                                 max_abs_diff=(y_train.float() - y_export.float()).abs().max().item(),
                                 finite=bool(torch.isfinite(y_export).all())))
        checks["rollout_export_equals_training_model"] = all(r["bit_equal"] and r["finite"] for r in rows)
        if a.server_schedules and index == 0:
            report["server"] = server_check(run / export_name, policy, [s for s in a.server_schedules.split(",") if s])
            checks["eval_server_equals_loader_rollout"] = all(r["bit_equal"] and r["finite"] for r in report["server"])
        del policy; torch.cuda.empty_cache()
        if a.padding_check and index == 0:
            model.train()
            b = torch.utils.data.default_collate([data[-1] for _ in range(4)])
            noise, anoise, atau = (t.to(dev) for t in sample_noise(b["training_index"].tolist(), 0, 42, dim=29))
            sample = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
            sample.update(action_scale=scale, action_offset=offset)
            rows_pad = {}
            for s in ("flat_k1_n2", "flat_k4_n10"):
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, logs = model(sample, SCHEDULES[s], noise, anoise, atau)
                loss.backward()
                grads = [q.grad for q in model.parameters() if q.requires_grad and q.grad is not None]
                rows_pad[s] = dict(loss=loss.item(), logs={k: v.item() for k, v in logs.items()},
                    grads_finite=all(bool(torch.isfinite(g).all()) for g in grads),
                    grads_zero=all(int(torch.count_nonzero(g)) == 0 for g in grads), grad_tensors=len(grads))
            checks["all_padding_microbatch_zero"] = all(r["loss"] == 0 and r["grads_finite"] and r["grads_zero"]
                                                        and all(v == 0 for v in r["logs"].values()) for r in rows_pad.values())
            report["padding"] = rows_pad
        ok = all(checks.values())
        all_ok &= ok
        report["pairs"][export_name] = dict(recovery=recovery_name, passed=ok, checks=checks, frozen=why_frozen,
            student_tensors_changed=f"{changed}/{len(sa)}", update=recovery["update"], epoch=recovery["epoch"],
            run=config.get("parent", {}).get("name"), kind=config["kind"], rollout=rows,
            export_sha256=sha256_file(run / export_name))
        print(json.dumps({export_name: dict(passed=ok, checks=checks)}), flush=True)
        save()
        del model, recovery, x; torch.cuda.empty_cache()
    report.update(finished=time.time(), passed=all_ok)
    save()
    if not all_ok: raise SystemExit("export verification failed")


if __name__ == "__main__":
    main()
