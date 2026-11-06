#!/usr/bin/env python3
"""Independent recomputation of the CL-0 LIBERO and RoboCasa-GR1 tables from the per-episode evidence stored in this folder.

Usage: python3 verify_h100_additions_20261106.py   (run from anywhere; reads the folders next to this script)
Writes elastic_cl0_libero_20261104/independent_check.json and robocasa_gr1_v0_20261101/independent_check.json.
"""
import csv, io, json, re, sys, tarfile
from collections import defaultdict
from pathlib import Path

ER = Path(__file__).resolve().parent
A = ER / "elastic_cl0_libero_20261104"; B = ER / "robocasa_gr1_v0_20261101"
rec = dict(cl0={}, gr1={}, failures=[])
fail = rec["failures"].append
SCHED = {"[1,1]": "flat_k1_n2", "[2,2]": "flat_k2_n2", "[1]x5": "flat_k1_n5", "[4,4]": "flat_k4_n2", "[2]x5": "flat_k2_n5",
         "[1]x10": "flat_k1_n10", "[4]x5": "flat_k4_n5", "[2]x10": "flat_k2_n10", "[4]x10": "flat_k4_n10"}

# ---------------- CL-0: report rows vs per-episode CSV vs per-group rollout JSON
ev = A / "eval"
report = {}
for line in (ev / "cl0_eval_report.md").read_text().splitlines():
    c = [x.strip() for x in line.strip("|").split("|")]
    if len(c) == 8 and c[0].startswith("CL0-E"):
        report[(c[0], SCHED[c[1]])] = dict(sr=float(c[2].split()[0]), suites=[float(x) for x in c[3:7]], seeds=[float(x) for x in c[7].split("/")])
groups_by_cond = {}
with tarfile.open(ev / "final_rollout.tar.gz") as t:
    for member in t.getmembers():
        parts = member.name.split("/")
        if len(parts) != 4 or parts[2] != "groups" or not member.isfile():
            continue
        m = re.match(r"elastic__(CL0-E\d)__(flat_k\d_n\d+)__([0-9a-f]{64})$", parts[1])
        x = json.loads(t.extractfile(member).read())
        assert len(x["episodes"]) == 50 and sum(e["success"] for e in x["episodes"]) == x["successes"], member.name
        c = groups_by_cond.setdefault((m[1], m[2]), dict(sha=m[3], groups=0, eps={}))
        c["groups"] += 1
        for e in x["episodes"]:
            c["eps"][(e["suite"], e["task"], e["initial_state_index"], e["seed"])] = bool(e["success"])
for f in sorted((ev / "report_csv").glob("*__episodes.csv")):
    run, sched = f.name.split("__")[:2]
    rows = list(csv.DictReader(f.open()))
    keys = {(r["suite"], int(r["task"]), int(r["initial_state_index"]), int(r["seed"])) for r in rows}
    succ = lambda sel: 100 * sum(r["success"] == "True" for r in sel) / len(sel)
    sr = succ(rows)
    suites = [succ([r for r in rows if r["suite"] == s]) for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10")]
    seeds = [succ([r for r in rows if r["seed"] == s]) for s in ("42", "43", "44")]
    want = report.get((run, sched))
    g = groups_by_cond[(run, sched)]
    csv_eps = {(r["suite"], int(r["task"]), int(r["initial_state_index"]), int(r["seed"])): r["success"] == "True" for r in rows}
    ok = (len(rows) == 6000 and len(keys) == 6000 and want is not None and g["groups"] == 120 and g["eps"] == csv_eps
          and abs(round(sr, 2) - want["sr"]) < 0.006 and all(abs(round(a, 2) - b) < 0.006 for a, b in zip(suites, want["suites"]))
          and all(abs(round(a, 2) - b) < 0.006 for a, b in zip(seeds, want["seeds"])))
    rec["cl0"][f"{run} {sched}"] = dict(episodes=len(rows), unique=len(keys), successes=sum(csv_eps.values()), sr=round(sr, 4),
                                        report_sr=want and want["sr"], groups=g["groups"], csv_equals_rollout_json=g["eps"] == csv_eps, passed=ok)
    if not ok:
        fail(f"CL0 {run} {sched}")
rec["cl0_conditions"] = len(rec["cl0"]); rec["cl0_report_rows"] = len(report)
if len(rec["cl0"]) != 27 or len(report) != 27:
    fail("CL0 condition count")

# ---------------- GR1: per-task episode JSON vs summary.json vs markdown tables
def table(md, first_col="Task"):
    lines = [l for l in md.read_text().splitlines() if l.startswith("|")]
    head = [x.strip() for x in lines[0].strip("|").split("|")]
    body = {}
    for l in lines[2:]:
        c = [x.strip().strip("*") for x in l.strip("|").split("|")]
        body[c[0]] = dict(zip(head[1:], c[1:]))
    return head, body

def run_dir(d, n):
    per = {}
    for f in sorted((d / f"n{n}").glob("*.json")):
        x = json.loads(f.read_text())
        assert x["n_action_steps"] == n and x["n_episodes"] == 50 and len(x["successes"]) == 50, f
        k = sum(bool(s) for s in x["successes"])
        assert k == x["successes_first_n"], f
        per[f.stem.replace("_GR1ArmsAndWaistFourierHands_Env", "")] = k
    summ = json.loads((d / "summary.json").read_text())["by_n_action_steps"][str(n)]
    same = {t.replace("_GR1ArmsAndWaistFourierHands_Env", ""): v["successes"] for t, v in summ["per_task"].items()} == per
    return per, summ, same

WS = B / "eval_ws"
runs = {"ws ep3": (WS / "results/official_epoch03", 12), "H100 ep4": (B / "eval_h100/official_epoch04", 12),
        "H100 ep5": (B / "eval_h100/official_epoch05", 12), "ws ep4": (WS / "results/official_epoch04", 12),
        "ws ep5": (WS / "results/official_epoch05", 12)}
_, official = table(WS / "results/official_results_all.md")
_, sweep = table(WS / "results/sweep_epoch04/sweep_epoch04_report.md")
for n in (8, 16, 24, 32):
    runs[f"n {n}"] = (WS / "results/sweep_epoch04", n)
for col, (d, n) in runs.items():
    per, summ, same = run_dir(d, n)
    tab = official if not col.startswith("n ") else sweep
    mean = 100 * sum(per.values()) / (50 * len(per))
    task_ok = all(int(tab[t][col]) == 2 * k for t, k in per.items())
    avg_ok = abs(float(tab["Average"][col]) - round(mean, 2)) < 0.006
    ok = len(per) == 24 and same and summ["complete"] and task_ok and avg_ok and abs(summ["mean_success"] * 100 - mean) < 1e-9
    rec["gr1"][col] = dict(tasks=len(per), episodes=50 * len(per), successes=sum(per.values()), mean_percent=round(mean, 4),
                           summary_json_equal=same, table_tasks_equal=task_ok, table_average_equal=avg_ok, passed=ok)
    if not ok:
        fail(f"GR1 {col}")
# n 12 column of the sweep table is the official ws ep4 run
rec["gr1"]["sweep n 12 = ws ep4"] = all(sweep[t]["n 12 (official)"] == official[t]["ws ep4"] for t in official)
if not rec["gr1"]["sweep n 12 = ws ep4"]:
    fail("sweep n12 vs ws ep4")
rec["passed"] = not rec["failures"]
(A / "independent_check.json").write_text(json.dumps(dict(passed=not any(f.startswith("CL0") for f in rec["failures"]), conditions=rec["cl0"],
    checks="27 conditions x 6,000 episodes: unique episode keys; per-episode CSV == per-group rollout JSON (final_rollout.tar.gz); "
           "overall, per-suite and per-seed success rates == cl0_eval_report.md"), indent=1) + "\n")
(B / "independent_check.json").write_text(json.dumps(dict(passed=not any(not f.startswith("CL0") for f in rec["failures"]), runs=rec["gr1"],
    checks="9 runs x 24 tasks x 50 episodes: per-episode successes == summary.json == official_results_all.md / sweep_epoch04_report.md "
           "task cells and averages; the sweep's n 12 column == the official ws epoch-4 run"), indent=1) + "\n")
print(json.dumps(dict(passed=rec["passed"], failures=rec["failures"], cl0=len(rec["cl0"]), gr1=list(rec["gr1"]))))
