#!/usr/bin/env python3
"""Progress of the CL-0 matched evaluation: completed task/seed groups per condition (240 each), running success
rate, claims in flight, throughput from the runner logs and an ETA. Usage: eval_status.py <EVAL_OUT>"""
import glob, json, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

root = Path(sys.argv[1]); fr = root / "final_rollout"
ORDER = [f"{r}:{s}" for r in ("CL0-E0", "CL0-E1", "CL0-E2") for s in
         ("flat_k1_n2", "flat_k2_n2", "flat_k1_n5", "flat_k4_n2", "flat_k2_n5", "flat_k1_n10", "flat_k4_n5", "flat_k2_n10", "flat_k4_n10")]
total_groups = done_groups = 0; recent = []
for key in ORDER:
    run, s = key.split(":")
    dirs = glob.glob(str(fr / f"elastic__{run}__{s}__*"))
    total_groups += 240
    if not dirs:
        print(f"{run:7s} {s:12s}   0/240 groups"); continue
    d = Path(dirs[0]); recs = [json.loads(p.read_text()) for p in (d / "groups").glob("*.json")]
    claims = len(list((d / "claims").glob("*.claim"))) if (d / "claims").exists() else 0
    eps = sum(len(r["episodes"]) for r in recs); succ = sum(r["successes"] for r in recs)
    done_groups += len(recs); recent += [r["completed_at"] for r in recs]
    backend = {r["backend"] for r in recs}
    summ = "summary.json" if (d / "summary.json").exists() else ""
    rate = f"{100 * succ / eps:6.2f}%" if eps else "   -  "
    print(f"{run:7s} {s:12s} {len(recs):3d}/240 groups  {eps:5d} eps  success {rate}  in-flight {claims:2d}  "
          f"{','.join(sorted(backend))} {summ}")
now = time.time(); last_hour = sum(1 for t in recent if now - t < 3600)
print(f"total {done_groups}/{total_groups} groups ({100 * done_groups / total_groups:.1f}%), {done_groups * 50} episodes; "
      f"groups completed in the last hour: {last_hour}")
if last_hour:
    eta = datetime.now(timezone.utc) + timedelta(hours=(total_groups - done_groups) / last_hour)
    print(f"ETA at the last-hour rate: {eta:%Y-%m-%d %H:%M} UTC ({(eta - timedelta(hours=5)):%H:%M} EST)")
for log in sorted((root / "logs").glob("runner_*.log")):
    lines = [l for l in log.read_text().splitlines() if l.startswith('{"event": "group"')]
    if lines:
        m = json.loads(lines[-1]); print(f"{log.name}: last group {m['condition'].split('__')[1]}/{m['group']} "
                                         f"{m['successes']}/{m['episodes']} in {m['seconds']:.0f}s, eps/hour {m['eps_per_hour']}")
    errs = [l for l in log.read_text().splitlines() if "Traceback" in l or '"event": "error"' in l]
    if errs: print(f"{log.name}: {len(errs)} error lines")
