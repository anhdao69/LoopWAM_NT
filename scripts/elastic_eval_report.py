#!/usr/bin/env python3
"""Results of the elastic decisive evaluation (plan sections 12.2 and 13), from the matched runner's group records.

Predeclared on 2026-10-28, before any final-checkpoint episode was scored:
- a condition is reported only when all 6,000 episodes (4 suites x 10 tasks x 50 states x seeds 42/43/44) exist;
  paired contrasts require identical episode keys (suite, task, initial state, seed) in both arms;
- primary: at flat_k1_n2 ([1,1]) the differences E1-E0, E2-E0 and E2-E1 in success rate;
- estimator: difference of success rates (equal to the mean of per-cluster differences, a cluster being
  (suite, task, initial state) with its three evaluation seeds);
- interval: 95% percentile interval from 10,000 cluster-bootstrap resamples (all seeds of a cluster kept together);
- test: two-sided paired cluster sign-flip permutation test (10,000 flips of the per-cluster differences), with
  Holm correction across the three primary contrasts; [1]x10 and [4]x10 contrasts are secondary and unadjusted;
- per-suite differences use the same resampling within the suite.
A nonsignificant difference is not evidence of equivalence.
Extended on 2026-10-29 at the user's request, before any SP11 or remaining-grid episode was scored (plan sections 8,
12.3 and 13.1: grid results are secondary unless separately preregistered):
- conditions: E0/E1/E2 over all nine flat schedules, and SP11 at flat_k1_n2;
- secondary, unadjusted: the three contrasts at each of the other eight schedules, and SP11-E0 at flat_k1_n2 (the
  practical equal-total-budget cost of elasticity at the fastest point; not a pure weight-sharing penalty);
- every interval and test draws from its own random stream seeded by (20261028, CRC32 of its name), so results do not
  depend on which other conditions are present; the primary analysis is otherwise unchanged.
"""
import argparse, csv, glob, json, math, zlib
from pathlib import Path
import numpy as np

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SHORT = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}
RUNS = ("E0", "E1", "E2")
# Ascending action calls (plan section 3.2): 24, 36, 60, 60, 90, 120, 150, 180, 300.
SCHEDULES = ("flat_k1_n2", "flat_k2_n2", "flat_k1_n5", "flat_k4_n2", "flat_k2_n5", "flat_k1_n10", "flat_k4_n5",
             "flat_k2_n10", "flat_k4_n10")
LABEL = {"flat_k1_n2": "[1,1]", "flat_k2_n2": "[2,2]", "flat_k1_n5": "[1]x5", "flat_k4_n2": "[4,4]", "flat_k2_n5": "[2]x5",
         "flat_k1_n10": "[1]x10", "flat_k4_n5": "[4]x5", "flat_k2_n10": "[2]x10", "flat_k4_n10": "[4]x10"}
EXTRA = (("SP11", "flat_k1_n2"),)
CONTRASTS = (("E1", "E0"), ("E2", "E0"), ("E2", "E1"))
RESAMPLES, SEED = 10_000, 20261028


def load(root, run, schedule):
    dirs = glob.glob(str(Path(root) / "final_rollout" / f"elastic__{run}__{schedule}__*"))
    if len(dirs) != 1:
        return None
    d = Path(dirs[0])
    if not (d / "summary.json").exists():
        return None
    summary = json.loads((d / "summary.json").read_text())
    manifest = json.loads((d / "manifest.json").read_text())
    rows = []
    for p in sorted((d / "groups").glob("*.json")):
        rec = json.loads(p.read_text())
        if rec["checkpoint_sha256"] != summary["checkpoint_sha256"] or rec["schedule"] != schedule:
            raise ValueError(f"identity mismatch in {p}")
        rows += rec["episodes"]
    keys = {(r["suite"], r["task"], r["initial_state_index"], r["seed"]) for r in rows}
    if len(rows) != 6000 or len(keys) != 6000:
        raise ValueError(f"{d.name}: expected 6000 unique episodes, found {len(rows)} rows / {len(keys)} keys")
    outcome = {(r["suite"], r["task"], r["initial_state_index"], r["seed"]): int(bool(r["success"])) for r in rows}
    return dict(dir=str(d), summary=summary, manifest=manifest, rows=rows, outcome=outcome)


def clusters(outcome, suite=None):
    """{(suite, task, state): mean success over its seeds}."""
    acc = {}
    for (s, t, i, seed), y in outcome.items():
        if suite is None or s == suite:
            acc.setdefault((s, t, i), []).append(y)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def stream(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def paired(a, b, rng, suite=None):
    """Difference a-b with cluster bootstrap interval and sign-flip permutation p-value."""
    if set(a) != set(b):
        raise ValueError("paired arms have different episode keys")
    ca, cb = clusters(a, suite), clusters(b, suite)
    keys = sorted(ca)
    diff = np.array([ca[k] - cb[k] for k in keys])
    estimate = diff.mean()
    boots = diff[rng.integers(0, len(diff), size=(RESAMPLES, len(diff)))].mean(axis=1)
    signs = rng.choice((-1.0, 1.0), size=(RESAMPLES, len(diff)))
    null = (signs * diff).mean(axis=1)
    p = (1 + np.sum(np.abs(null) >= abs(estimate) - 1e-12)) / (1 + RESAMPLES)
    return dict(difference=float(estimate), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                p_permutation=float(p), clusters=len(diff),
                discordant_clusters=int(np.sum(diff != 0)))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adjusted = [0.0] * len(ps); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i])); adjusted[i] = running
    return adjusted


def rate_ci(outcome, rng):
    c = np.array(list(clusters(outcome).values()))
    boots = c[rng.integers(0, len(c), size=(RESAMPLES, len(c)))].mean(axis=1)
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_eval")
    ap.add_argument("--output", required=True, help="results JSON; a Markdown table is written next to it")
    ap.add_argument("--csv-dir", default="", help="optionally write one episodes CSV per condition here")
    a = ap.parse_args()
    data = {(r, s): load(a.eval_root, r, s) for r in RUNS for s in SCHEDULES}
    data.update({(r, s): load(a.eval_root, r, s) for r, s in EXTRA})
    conditions = {}
    for (r, s), d in data.items():
        rng = stream(f"rate:{r}/{s}")
        if d is None:
            conditions[f"{r}/{s}"] = dict(run=r, schedule=s, complete=False); continue
        sm = d["summary"]
        conditions[f"{r}/{s}"] = dict(run=r, schedule=s, label=LABEL[s], complete=True, episodes=sm["episodes"],
            successes=sm["successes"], success_rate=sm["success_rate"], ci95=rate_ci(d["outcome"], rng),
            suite_mean_success_rate=sm["suite_mean_success_rate"],
            per_suite={SHORT[k]: v["success_rate"] for k, v in sm["per_suite"].items()},
            per_seed={str(seed): sum(r_["success"] for r_ in d["rows"] if r_["seed"] == seed) / 2000 for seed in (42, 43, 44)},
            checkpoint_sha256=sm["checkpoint_sha256"], backend=sm["backend"], output=d["dir"])
        if a.csv_dir:
            out = Path(a.csv_dir) / f"{r}__{s}__episodes.csv"; out.parent.mkdir(parents=True, exist_ok=True)
            cols = ("suite", "task", "initial_state_index", "seed", "success", "policy_steps", "queries", "query_seconds")
            with out.open("w", newline="") as f:
                w = csv.writer(f); w.writerow(cols)
                for row in sorted(d["rows"], key=lambda x: (x["suite"], x["task"], x["initial_state_index"], x["seed"])):
                    w.writerow([row[c] for c in cols])
    contrasts = {}
    pairs = [(x, y, s) for s in SCHEDULES for x, y in CONTRASTS] + [("SP11", "E0", "flat_k1_n2")]
    for x, y, s in pairs:
        if data[(x, s)] is None or data[(y, s)] is None: continue
        name = f"{x}-{y}@{s}"; rng = stream(f"contrast:{name}")
        res = paired(data[(x, s)]["outcome"], data[(y, s)]["outcome"], rng)
        res["per_suite"] = {SHORT[k]: paired(data[(x, s)]["outcome"], data[(y, s)]["outcome"], rng, k) for k in SUITES}
        res["role"] = "primary" if s == "flat_k1_n2" and x != "SP11" else "secondary (unadjusted)"
        contrasts[name] = res
    primary = [k for k in contrasts if contrasts[k]["role"] == "primary"]
    if len(primary) == 3:
        for k, adj in zip(primary, holm([contrasts[k]["p_permutation"] for k in primary])):
            contrasts[k]["p_holm"] = adj
    result = dict(format="elastic-decisive-eval-v2", protocol="lightwam_20261009_matched via scripts/evaluate_rt_matched.py",
                  analysis=__doc__.strip(), resamples=RESAMPLES, rng_seed=SEED, conditions=conditions, contrasts=contrasts)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    pct = lambda v: f"{100 * v:.2f}"
    lines = ["| Run | Schedule | Success % (95% CI) | Spatial | Object | Goal | Long | Seeds 42/43/44 |", "|---|---|---|---|---|---|---|---|"]
    for k, c in conditions.items():
        if not c["complete"]:
            lines.append(f"| {c['run']} | {LABEL[c['schedule']]} | pending (incomplete; not scored) | | | | | |"); continue
        lines.append(f"| {c['run']} | {c['label']} | {pct(c['success_rate'])} ({pct(c['ci95'][0])}-{pct(c['ci95'][1])}) | "
                     + " | ".join(pct(c["per_suite"][n]) for n in ("Spatial", "Object", "Goal", "Long")) + " | "
                     + "/".join(pct(c["per_seed"][s]) for s in ("42", "43", "44")) + " |")
    lines += ["", "| Contrast | Role | Difference pp (95% CI) | p (perm.) | p (Holm) | Long pp |", "|---|---|---|---|---|---|"]
    for k, c in contrasts.items():
        holm_p = f"{c['p_holm']:.4f}" if "p_holm" in c else "-"
        lines.append(f"| {k.replace('@', ' @ ')} | {c['role']} | {100 * c['difference']:+.2f} ({100 * c['ci95'][0]:+.2f} to "
                     f"{100 * c['ci95'][1]:+.2f}) | {c['p_permutation']:.4f} | {holm_p} | "
                     f"{100 * c['per_suite']['Long']['difference']:+.2f} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
