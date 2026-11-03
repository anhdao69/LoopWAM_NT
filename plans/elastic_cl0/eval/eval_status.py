#!/usr/bin/env python3
"""Progress of the CL-0 matched evaluation: completed task/seed groups per condition (4 suites x 10 tasks x 3 seeds =
120 groups of 50 episodes each), running success rate, claims in flight, and a work-weighted ETA.
Usage: eval_status.py <EVAL_OUT>

ETA: each group's cost is predicted from the reference BASE10 run of the same index and schedule (CL0-Ex <-> Ex):
sum over its 50 episodes of (policy steps + 30 settling) + 2.7 x queries x query_cost(schedule), query_cost growing
with action calls as the CUDA-graph latency did in the campaign (24 + 0.246 x calls ms, normalized to [1,1]). The
ratio of measured to predicted seconds over the completed groups converts the pending cost into worker-seconds, which
the workers in flight share. Heavier schedules come last in the queue, so a plain groups-per-hour rate is optimistic.
"""
import csv, glob, json, sys, time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

root = Path(sys.argv[1]); fr = root / "final_rollout"
REF = root / "reference_hf" / "libero-all-suites" / "Elastic" / "eval" / "conditions"
RUNS = ("CL0-E0", "CL0-E1", "CL0-E2")
CALLS = dict(flat_k1_n2=24, flat_k2_n2=36, flat_k1_n5=60, flat_k4_n2=60, flat_k2_n5=90, flat_k1_n10=120,
             flat_k4_n5=150, flat_k2_n10=180, flat_k4_n10=300)
ORDER = [(r, s) for s in CALLS for r in RUNS]                     # queue order (eval_env.sh): schedule-major
GROUPS = [f"{su}_seed{sd}_task{t}" for su in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
          for sd in (42, 43, 44) for t in range(10)]
PER_COND = len(GROUPS)                                             # 120


def predicted_costs(run, s):
    """Predicted relative cost of every group of condition (run, s), from the matching reference condition."""
    path = REF / f"{run.replace('CL0-', '')}__{s}" / "episodes.csv"
    qf = (24 + 0.246 * CALLS[s]) / (24 + 0.246 * 24)
    cost = defaultdict(float)
    if path.exists():
        for row in csv.DictReader(open(path)):
            cost[f"{row['suite']}_seed{row['seed']}_task{row['task']}"] += \
                int(row["policy_steps"]) + 30 + 2.7 * int(row["queries"]) * qf
    return cost


total = done = 0; recent = []; measured = predicted = pending_cost = 0.0; in_flight = 0
for run, s in ORDER:
    cost = predicted_costs(run, s)
    dirs = glob.glob(str(fr / f"elastic__{run}__{s}__*"))
    recs = {}
    claims = 0
    if dirs:
        d = Path(dirs[0])
        recs = {p.stem: json.loads(p.read_text()) for p in (d / "groups").glob("*.json")}
        claims = len(list((d / "claims").glob("*.claim"))) if (d / "claims").exists() else 0
    in_flight += claims; total += PER_COND; done += len(recs)
    for g, r in recs.items():
        recent.append(r["completed_at"])
        if g in cost:
            measured += r["elapsed_seconds"]; predicted += cost[g]
    pending_cost += sum(cost[g] for g in GROUPS if g not in recs)
    eps = sum(len(r["episodes"]) for r in recs.values()); succ = sum(r["successes"] for r in recs.values())
    rate = f"{100 * succ / eps:6.2f}%" if eps else "   -  "
    backend = ",".join(sorted({r["backend"] for r in recs.values()}))
    summ = "summary.json" if dirs and (Path(dirs[0]) / "summary.json").exists() else ""
    print(f"{run:7s} {s:12s} {len(recs):3d}/{PER_COND} groups {eps:5d} eps  success {rate}  in-flight {claims:2d}  "
          f"{backend} {summ}")
now = time.time(); last_hour = sum(1 for t in recent if now - t < 3600)
print(f"total {done}/{total} groups ({100 * done / total:.1f}%), {done * 50} of {total * 50} episodes; "
      f"in flight {in_flight}; groups completed in the last hour: {last_hour}")
if predicted and in_flight:
    # In-flight groups are counted as pending (their partial progress is not visible), so this errs late by at most
    # one group per worker.
    hours = pending_cost * measured / predicted / in_flight / 3600
    eta = datetime.now(timezone.utc) + timedelta(hours=hours)
    print(f"work-weighted ETA: {hours:.1f} h -> {eta:%Y-%m-%d %H:%M} UTC ({eta - timedelta(hours=5):%Y-%m-%d %H:%M} EST)"
          f"  [{measured / predicted * 1e3:.1f} ms per predicted unit over {done} groups, {in_flight} workers]")
for log in sorted((root / "logs").glob("runner_*.log")):
    text = log.read_text().splitlines()
    lines = [l for l in text if l.startswith('{"event": "group"')]
    if lines:
        m = json.loads(lines[-1]); print(f"{log.name}: last group {m['condition'].split('__')[1]}/"
                                         f"{m['condition'].split('__')[2]}/{m['group']} {m['successes']}/{m['episodes']} "
                                         f"in {m['seconds']:.0f}s; runner reports {m['pending']} pending")
    errs = [l for l in text if "Traceback" in l or '"event": "error"' in l]
    if errs: print(f"{log.name}: {len(errs)} error lines")
