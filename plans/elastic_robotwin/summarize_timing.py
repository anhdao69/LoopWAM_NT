#!/usr/bin/env python3
"""Summarize elastic timing runs: steady FLAT9-mean seconds/update and projected hours.

Usage: summarize_timing.py <timing dir> <out.json> [out.md]. Each run dir holds launch.json, config.json, metrics.jsonl.
The balanced sampler makes every block of 9 updates a permutation of FLAT9, so the mean over whole blocks after the first
(warm-up: CUDA-graph capture, allocator growth) is the exact FLAT9 average a full run converges to. 'wall' divides the
time between the first and last measured update stamps by their distance, so it also counts logging and syncs.
"""
import json, sys
from pathlib import Path
BUDGETS = {"1 RoboTwin epoch (47,462 updates)": 47462, "LIBERO-recipe update count (21,700)": 21700,
           "10 RoboTwin epochs (474,620 updates)": 474620}

def summarize(run):
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    launch = json.loads((run / "launch.json").read_text())
    n = len(rows); blocks = n // 9
    measured = rows[9:blocks * 9]
    if not measured: return dict(run=run.name, launch=launch, updates=n, error="fewer than two full FLAT9 blocks")
    sec = sum(r["seconds"] for r in measured) / len(measured)
    wall = (measured[-1]["time"] - rows[8]["time"]) / len(measured)
    per = {}
    for r in measured: per.setdefault(r["schedule"], []).append(r["seconds"])
    return dict(run=run.name, tag=run.parent.name, launch=launch, updates=n, measured_updates=len(measured),
        seconds_per_update=sec, wall_seconds_per_update=wall,
        per_schedule={k: sum(v) / len(v) for k, v in sorted(per.items())},
        data_wait_max=max(r["data_wait_seconds"] for r in measured),
        data_wait_mean=sum(r["data_wait_seconds"] for r in measured) / len(measured),
        peak_allocated_gb=max(r["peak_allocated_gb"] for r in rows), grad_norm_max=max(r["grad_norm"] for r in rows),
        first_update={k: rows[0][k] for k in ("local", "end", "grip", "anchor", "grad_norm", "schedule")},
        hours={k: wall * v / 3600 for k, v in BUDGETS.items()})

def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    runs = sorted(p.parent for p in root.glob("*/*/metrics.jsonl"))
    result = [summarize(r) for r in runs]
    out.write_text(json.dumps(result, indent=2))
    lines = ["| layout | run | GPUs | mb | s/update (wall) | data wait max | peak GB | 1 epoch h | 21,700 upd h | 10 epochs h |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result:
        if "error" in r: lines.append(f"| {r['run']} | {r['error']} |"); continue
        h = list(r["hours"].values())
        lines.append(f"| {r['tag']} | {r['run']} | {r['launch']['gpus']} | {r['launch']['microbatch']} | "
                     f"{r['seconds_per_update']:.3f} ({r['wall_seconds_per_update']:.3f}) | {r['data_wait_max']:.3f} | "
                     f"{r['peak_allocated_gb']:.1f} | {h[0]:.1f} | {h[1]:.1f} | {h[2]:.1f} |")
    text = "\n".join(lines); print(text)
    if len(sys.argv) > 3: Path(sys.argv[3]).write_text(text + "\n")

if __name__ == "__main__":
    main()
