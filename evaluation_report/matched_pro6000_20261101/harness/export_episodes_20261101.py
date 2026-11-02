#!/usr/bin/env python3
"""Export every scored episode of the 2026-11-01 batch to one CSV per condition (complete conditions only).

Usage: python3 export_episodes_20261101.py <output dir>
Columns: suite, task, seed, initial_state_index, success, policy_steps, queries, action_digest.
"""
import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path("/data/anh/code/wam_baselines")
FIELDS = ["suite", "task", "seed", "initial_state_index", "success", "policy_steps", "queries", "action_digest"]
ELASTIC = {"E0_k3_n10": "E0__flat_k3_n10", "E0_k5_n10": "E0__flat_k5_n10", "E0_k6_n10": "E0__flat_k6_n10",
           "Z10_parent2": "Z10__parent2", "Z10_parent5": "Z10__parent5", "Z10_k1_n2": "Z10__flat_k1_n2"}


def write(path, rows):
    rows.sort(key=lambda r: (r["suite"], r["task"], r["seed"], r["initial_state_index"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(path, len(rows), sum(bool(r["success"]) for r in rows))


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(Path(f).read_text()) for f in glob.glob(str(ROOT / "runs/loopwam_v0_v4a2_matched/episodes/*.json"))]
    if len(rows) == 6000:
        write(out / "episodes_v0_v4a2.csv", rows)
    for name, key in ELASTIC.items():
        rows = []
        for f in glob.glob(str(ROOT / f"runs/elastic_matched_20261101/final_rollout/matched6000pro__{key}__*/groups/*.json")):
            rows += json.loads(Path(f).read_text())["episodes"]
        if len(rows) == 6000:
            write(out / f"episodes_{name}.csv", rows)
        else:
            print(name, "incomplete", len(rows))


if __name__ == "__main__":
    main()
