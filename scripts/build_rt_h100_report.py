#!/usr/bin/env python3
"""Collect H100 RT evaluations plus remote campaign RT results into report evidence and filled tables.

Paired differences use (suite, task, initial state, seed) keys with a cluster bootstrap over
(suite, task, initial state), 10,000 resamples, 95% percentile intervals.
"""
import csv, json, statistics, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H100 = Path(sys.argv[1] if len(sys.argv) > 1 else "/groups/yshang/an221229/checkpoints/LoopWAM_NT/rt_eval_h100")
EVID = ROOT / "plans/residual_transport_evidence/h100_eval"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CALLS = {"S1": 192, "S2": 138, "S3": 168, "S4": 156, "S5": 102, "S6": 78, "S7": 48, "S8": 42, "parent2": 60}
REMOTE_GPU = "NVIDIA RTX PRO 6000 Blackwell Max-Q (remote campaign)"


def load_local():
    out = {}
    for d in sorted((H100 / "final_rollout").iterdir()):
        m = json.loads((d / "manifest.json").read_text())
        rows = []
        for p in sorted((d / "groups").glob("*.json")):
            g = json.loads(p.read_text())
            assert g["checkpoint_sha256"] == m["sha256"] and g["schedule"] == m["schedule"]
            rows += g["episodes"]
        out[f"{m['run']} {m['schedule']}"] = dict(run=m["run"], schedule=m["schedule"], sha256=m["sha256"],
            source="H100 (this work)", gpu=m.get("gpu"), backend=m["backend"], rows=rows, key=d.name,
            complete=len(rows) == 6000)
    return out


def load_remote():
    det = json.loads((ROOT / "evaluation_report/campaign/detailed_results.json").read_text())
    status = {f"{c['run']} {c['schedule']}": c for c in det["conditions"] if c["family"] == "residual_trans"}
    rows = {}
    for r in csv.DictReader(open(ROOT / "evaluation_report/campaign/episode_details.csv")):
        if r["model"].startswith("RT-"):
            rows.setdefault(r["model"], []).append(dict(suite=r["suite"], seed=int(r["seed"]), task=int(r["task"]),
                episode=int(r["episode"]), success=r["success"] == "True", policy_steps=int(r["policy_steps"])))
    out = {}
    for name, rs in rows.items():
        c = status[name]
        out[name] = dict(run=c["run"], schedule=c["schedule"], sha256=c["checkpoint_sha256"], source="remote campaign",
                         gpu=REMOTE_GPU, backend=(c.get("manifest") or {}).get("protocol", {}).get("backend"),
                         rows=rs, key=c["key"], complete=c["status"] == "complete" and len(rs) == 6000)
    return out


def summary(c):
    rows = c["rows"]; res = dict(episodes=len(rows), successes=sum(r["success"] for r in rows))
    per = {}
    for s in SUITES:
        rs = [r for r in rows if r["suite"] == s]
        if not rs: continue
        seeds = {}
        for seed in (42, 43, 44):
            q = [r for r in rs if r["seed"] == seed]
            if q: seeds[seed] = 100 * sum(r["success"] for r in q) / len(q)
        per[s] = dict(episodes=len(rs), successes=sum(r["success"] for r in rs),
                      success_pct=100 * sum(r["success"] for r in rs) / len(rs), seed_pct=seeds,
                      seed_std_pct=statistics.stdev(seeds.values()) if len(seeds) > 1 else None)
    res["per_suite"] = per
    if len(per) == 4 and c["complete"]:
        res["four_suite_mean_pct"] = sum(v["success_pct"] for v in per.values()) / 4
        seed_means = [sum(per[s]["seed_pct"][seed] for s in SUITES) / 4 for seed in (42, 43, 44)]
        res["seed_four_suite_mean_pct"] = dict(zip((42, 43, 44), seed_means))
        res["seed_std_pct"] = statistics.stdev(seed_means)
    return res


def paired(a, b, rng, n=10000):
    ka = {(r["suite"], r["task"], r["episode"], r["seed"]): r["success"] for r in a["rows"]}
    kb = {(r["suite"], r["task"], r["episode"], r["seed"]): r["success"] for r in b["rows"]}
    keys = sorted(set(ka) & set(kb))
    clusters = {}
    for k in keys:
        clusters.setdefault(k[:3], []).append(ka[k] - kb[k])
    sums = np.array([sum(v) for v in clusters.values()], float); cnt = np.array([len(v) for v in clusters.values()], float)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    boot = sums[idx].sum(1) / cnt[idx].sum(1) * 100
    diff = sums.sum() / cnt.sum() * 100
    return dict(paired_episodes=len(keys), diff_pp=diff, ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                a_only=sum(ka[k] and not kb[k] for k in keys), b_only=sum(kb[k] and not ka[k] for k in keys),
                cross_hardware=a["source"] != b["source"])


def main():
    EVID.mkdir(parents=True, exist_ok=True)
    conds = {**load_remote(), **load_local()}
    summ = {k: dict(run=c["run"], schedule=c["schedule"], source=c["source"], gpu=c["gpu"], backend=c["backend"],
                    checkpoint="epoch_10.pt", checkpoint_sha256=c["sha256"], complete=c["complete"],
                    action_calls=CALLS.get(c["schedule"]), **summary(c)) for k, c in conds.items()}
    (EVID / "conditions.json").write_text(json.dumps(summ, indent=1))
    with open(EVID / "suite_seed_results.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "source", "suite", "seed", "successes", "episodes", "success_pct"])
        for k, c in conds.items():
            for s in SUITES:
                for seed in (42, 43, 44):
                    q = [r for r in c["rows"] if r["suite"] == s and r["seed"] == seed]
                    if q: w.writerow([k, c["source"], s, seed, sum(r["success"] for r in q), len(q), f"{100 * sum(r['success'] for r in q) / len(q):.2f}"])
    with open(EVID / "h100_episodes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "checkpoint_sha256", "suite", "seed", "task", "episode", "initial_state_index", "success", "policy_steps", "queries"])
        for k, c in conds.items():
            if c["source"].startswith("H100"):
                for r in sorted(c["rows"], key=lambda r: (r["suite"], r["seed"], r["task"], r["episode"])):
                    w.writerow([k, c["sha256"], r["suite"], r["seed"], r["task"], r["episode"], r["initial_state_index"], int(r["success"]), r["policy_steps"], r["queries"]])
    rng = np.random.default_rng(20261013)
    comps = [("Gate 1 / Phase-1 R2", "RT-A S1", "RT-B4 S1", "RT-A minus RT-B4 >= +1 (Gate 1) / >= +2 with CI lower > 0 (R2)"),
             ("Learned transport", "RT-A S1", "RT-T1ft S1", ">= +1"), ("Learned transport", "RT-A S2", "RT-T1ft S2", ">= +1"),
             ("Transport-only scope (partial: RT-B5 not evaluated)", "RT-Pa S2", "RT-A S2", "RT-Pa within 1 point of RT-A"),
             ("Transport-only scope (S1, descriptive)", "RT-Pa S1", "RT-A S1", "descriptive"),
             ("FU vs TF", "RT-A S2", "RT-TF S2", ">= +1"), ("FU vs TF (S1, descriptive)", "RT-A S1", "RT-TF S1", "descriptive"),
             ("Phase-2 retention", "RT-A2 S1", "RT-A S1", "within 1 point"), ("Phase-2 retention", "RT-A2 S2", "RT-A S2", "within 1 point"),
             ("Phase-2 R1 (partial: no RT-B1 line)", "RT-A2 S6", "RT-B2a parent2", "+2 with CI lower > 0 at equal latency, or equal at >=20% lower latency"),
             ("Phase-2 R1 (partial: no RT-B1 line)", "RT-A2 S7", "RT-B2a parent2", "same"),
             ("RT-A schedule sweep (descriptive)", "RT-A S4", "RT-A S2", "descriptive")]
    out = []
    for name, a, b, rule in comps:
        if a in conds and b in conds and conds[a]["complete"] and conds[b]["complete"]:
            out.append(dict(decision=name, a=a, b=b, rule=rule, **paired(conds[a], conds[b], rng)))
        else:
            out.append(dict(decision=name, a=a, b=b, rule=rule, unavailable="one side incomplete or not evaluated"))
    (EVID / "paired_comparisons.json").write_text(json.dumps(out, indent=1))
    for k, v in summ.items():
        mean = v.get("four_suite_mean_pct"); long = v["per_suite"].get("libero_10", {}).get("success_pct")
        print(f"{k:16s} {v['source']:17s} complete={v['complete']!s:5s} long={long if long is None else round(long, 2)} mean={mean if mean is None else round(mean, 2)}")
    for c in out:
        print(c["decision"], c["a"], "-", c["b"], round(c["diff_pp"], 2) if "diff_pp" in c else c.get("unavailable"), [round(x, 2) for x in c.get("ci95", [])])


if __name__ == "__main__":
    main()
