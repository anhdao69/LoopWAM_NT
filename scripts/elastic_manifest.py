#!/usr/bin/env python3
"""Machine-readable per-run manifest for an elastic run (plan section 17), from the run's own records."""
import argparse, collections, hashlib, json, time
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(); p.add_argument("--output-dir", required=True)
    out = Path(p.parse_args().output_dir)
    config = json.loads((out / "config.json").read_text())
    status = json.loads((out / "status.json").read_text())
    rows = {}
    with (out / "metrics.jsonl").open() as f:
        for line in f:
            row = json.loads(line); rows[row["update"]] = row  # a resumed segment rewrites later updates
    updates = [rows[u] for u in sorted(rows)]
    if [r["update"] for r in updates] != list(range(1, len(updates) + 1)):
        raise ValueError("Metrics are not contiguous from update 1")
    per = collections.defaultdict(lambda: collections.Counter())
    losses = ("local", "end", "grip", "anchor", "total")
    for r in updates:
        c = per[r["schedule"]]; c["updates"] += 1; c["valid_windows"] += r["windows"]; c["seconds"] += r["seconds"]
        for key in losses:
            if key in r: c[key + "_sum"] += r[key] * r["windows"]
    per_config = {name: dict(updates=c["updates"], valid_windows=c["valid_windows"], seconds=round(c["seconds"], 1),
                             **{f"mean_{k}": c[k + "_sum"] / c["valid_windows"] for k in losses if k + "_sum" in c})
                  for name, c in sorted(per.items())}
    seconds = sum(r["seconds"] for r in updates)
    checkpoints = {}
    for label in ("step_00002000", "epoch_08", "epoch_09", "epoch_10"):
        path = out / f"{label}.pt"
        if not path.exists(): continue
        receipt = out / f"{label}.pt.uploaded.json"
        record = json.loads(receipt.read_text()) if receipt.exists() else {}
        checkpoints[label] = dict(sha256=record.get("sha256") or sha256(path), bytes=path.stat().st_size,
                                  hub_path=record.get("path"), hub_revision=record.get("revision"))
    sequence = out / "schedule_sequence.json"
    contract = {k: config[k] for k in ("asset_sha256", "config", "windows", "world", "microbatch", "global_batch",
                                       "epochs", "planned_updates", "normalization_sha256", "checkpoint_blocks",
                                       "deterministic_algorithms") if k in config}
    for key in ("schedule_sequence_sha256", "checkpoint_min_blocks", "init_parts", "latent_content_digest"):
        contract[key] = config.get(key)
    cfg = config["config"]
    manifest = dict(
        format="elastic-run-manifest-v1", run=config["run"], created_utc=time.time(), status=status["status"],
        source=dict(git_revision=config["git_revision"], branch="elastic", plan="plans/LoopWAM_elastic.md"),
        environment=dict(gpu=config["gpu"], torch=config["torch_version"], slurm_job_id=config.get("slurm_job_id")),
        contract=contract, transport=cfg["kind"], scope=cfg["scope"], supported_schedules=cfg["schedules"],
        sampler=cfg["sampler"], initialization=dict(recipe=cfg["init"], artifact_sha256=config["asset_sha256"].get("elastic_init"),
                                                     parts=config.get("init_parts")),
        trainable_parameters=config["trainable_parameters"],
        optimizer=dict(type="AdamW fused", betas=[.9, .95], eps=1e-8,
                       groups=dict(step_and_transport=dict(lr=1e-3, weight_decay=0.), lora=dict(lr=2e-4, weight_decay=0.),
                                   action_expert=dict(lr=3e-5, weight_decay=.01)),
                       warmup_updates=651, decay="cosine to 1% (train_loopwam.lr_factor)", clip_global_norm=1.,
                       fresh_optimizer=True, ema_decay=.999, exported_weights="raw (EMA kept only in recovery)"),
        precision="FP32 weights/optimizer, BF16 autocast compute", seeds=dict(data_order=cfg["seed"], noise=cfg["seed"],
                  schedule=cfg["seed"]), loss_weights=dict(local=1., end=.5, grip=.1, anchor=.5),
        truncation="detach z and transport history at the start of student step j when j <= N-3",
        teacher="frozen BASE10 (v0-video4-action4), deployed 0.1 Euler grid; first local target reused bit-identically from the endpoint trajectory",
        schedule_sequence=dict(sha256=json.loads(sequence.read_text())["sha256"] if sequence.exists() else None),
        totals=dict(updates=len(updates), valid_windows=sum(r["windows"] for r in updates), train_seconds=round(seconds, 1),
                    gpu_hours=round(seconds * config["world"] / 3600, 2),
                    peak_allocated_gb=max(r["peak_allocated_gb"] for r in updates)),
        per_configuration=per_config, checkpoints=checkpoints, selected_checkpoint="epoch_10 (raw, predeclared)")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(run=manifest["run"], updates=len(updates), checkpoints=list(checkpoints))))

if __name__ == "__main__":
    main()
