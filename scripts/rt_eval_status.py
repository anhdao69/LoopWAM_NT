#!/usr/bin/env python3
"""Progress table for the H100 RT missing-condition evaluation."""
import json, sys, time
from pathlib import Path
root = Path(sys.argv[1] if len(sys.argv) > 1 else "/groups/yshang/an221229/checkpoints/LoopWAM_NT/rt_eval_h100") / "final_rollout"
tot_e = tot_g = 0
for d in sorted(root.iterdir()):
    gs = [json.loads(p.read_text()) for p in (d / "groups").glob("*.json")] if (d / "groups").exists() else []
    e = sum(len(g["episodes"]) for g in gs); s = sum(g["successes"] for g in gs)
    claims = len(list((d / "claims").glob("*.claim"))) if (d / "claims").exists() else 0
    m = json.loads((d / "manifest.json").read_text())
    tot_e += e; tot_g += len(gs)
    print(f"{m['run']:8s} {m['schedule']:8s} groups {len(gs):3d}/120 running {claims:2d} episodes {e:5d} SR {s / e * 100 if e else float('nan'):6.2f}")
print(f"total groups {tot_g}/{120 * len(list(root.iterdir()))} episodes {tot_e}")
