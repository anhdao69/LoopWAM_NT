#!/usr/bin/env python3
"""Results of the matched LIBERO evaluation of CL0-E0/E1/E2 (elastic distillation from the CL-0 teacher).

Predeclared on 2026-11-03, before any CL0 episode was scored:
- protocol lightwam_20261009_matched through scripts/evaluate_rt_matched.py (4 suites x 10 tasks x official initial
  states 0..49 x evaluation seeds 42/43/44 = 6,000 episodes per condition), raw epoch_10.pt of each run, CUDA-graph
  backend where the per-run gate is bit-exact against eager; a condition is reported only when all 6,000 episodes
  exist; paired contrasts require identical episode keys (suite, task, initial state, seed) and identical protocol;
- conditions: CL0-E0, CL0-E1, CL0-E2 at all nine flat schedules;
- reference arms: the BASE10-teacher runs E0/E1/E2 (epoch_10 sha256 5569c47f / be1cbe2c / 6d92ac13), from their
  published H100 records (HF anhdao69/LoopWAM_NT libero-all-suites/Elastic/eval/conditions/*/episodes.csv), same
  runner and protocol; the stack used here first reproduced reference E0 episodes (validation/parity_e0.json), and the
  report states whether that reproduction was exact;
- estimators, intervals and tests are those of scripts/elastic_eval_report.py: difference of success rates (= mean
  of per-cluster differences, a cluster being (suite, task, initial state) with its three seeds), 95% percentile
  interval from 10,000 cluster-bootstrap resamples, two-sided paired cluster sign-flip permutation test (10,000 flips);
  every interval/test draws from its own stream seeded by (20261103, CRC32 of its name);
- primary family T (teacher), Holm over its three contrasts: CL0-E0 - E0, CL0-E1 - E1, CL0-E2 - E2 at flat_k1_n2;
- primary family R (transport, replicating the BASE10 campaign's primary question), Holm over its three contrasts:
  CL0-E1 - CL0-E0, CL0-E2 - CL0-E0, CL0-E2 - CL0-E1 at flat_k1_n2;
- secondary, unadjusted: the same six contrasts at each of the other eight flat schedules;
- descriptive only: CL-0 itself (96.10%, NFE 10, RTX PRO 6000 campaign host) and any cross-host number.
A nonsignificant difference is not evidence of equivalence.
"""
import argparse, csv, glob, json, sys, zlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from elastic_eval_report import SUITES, SHORT, SCHEDULES, LABEL, RESAMPLES, clusters, paired, holm, rate_ci  # noqa: E402

SEED = 20261103
CL0 = ("CL0-E0", "CL0-E1", "CL0-E2")
BASE = {"E0": "5569c47fa6c91e136a611d2b819a053c94f6b2956a7f602bbed4db23c69f60e8",
        "E1": "be1cbe2c", "E2": "6d92ac13"}  # E1/E2 prefixes from the campaign report; full hashes read from records
TEACHER = (("CL0-E0", "E0"), ("CL0-E1", "E1"), ("CL0-E2", "E2"))
TRANSPORT = (("CL0-E1", "CL0-E0"), ("CL0-E2", "CL0-E0"), ("CL0-E2", "CL0-E1"))


def stream(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def keyed(rows):
    out = {(r["suite"], int(r["task"]), int(r["initial_state_index"]), int(r["seed"])): int(r["success"] in (True, "True"))
           for r in rows}
    if len(rows) != 6000 or len(out) != 6000:
        raise ValueError(f"expected 6000 unique episodes, found {len(rows)} rows / {len(out)} keys")
    return out


def load_cl0(root, run, schedule):
    dirs = glob.glob(str(Path(root) / "final_rollout" / f"elastic__{run}__{schedule}__*"))
    if len(dirs) != 1 or not (Path(dirs[0]) / "summary.json").exists():
        return None
    d = Path(dirs[0]); summary = json.loads((d / "summary.json").read_text())
    manifest = json.loads((d / "manifest.json").read_text())
    rows = []
    for p in sorted((d / "groups").glob("*.json")):
        rec = json.loads(p.read_text())
        if rec["checkpoint_sha256"] != summary["checkpoint_sha256"] or rec["schedule"] != schedule \
                or rec["checkpoint_sha256"] != manifest["sha256"]:
            raise ValueError(f"identity mismatch in {p}")
        if [e["episode"] for e in rec["episodes"]] != list(range(50)):
            raise ValueError(f"incomplete group {p}")
        rows += rec["episodes"]
    return dict(dir=str(d), summary=summary, manifest=manifest, rows=rows, outcome=keyed(rows))


def load_reference(conditions_dir, run, schedule):
    d = Path(conditions_dir) / f"{run}__{schedule}"
    manifest = json.loads((d / "manifest.json").read_text()); summary = json.loads((d / "summary.json").read_text())
    if not manifest["sha256"].startswith(BASE[run]) or summary["checkpoint_sha256"] != manifest["sha256"] \
            or manifest["schedule"] != schedule:
        raise ValueError(f"reference {run}/{schedule} is not the pinned BASE10 run")
    with open(d / "episodes.csv") as f:
        rows = list(csv.DictReader(f))
    outcome = keyed(rows)
    if sum(outcome.values()) != summary["successes"]:
        raise ValueError(f"reference {run}/{schedule}: episodes.csv disagrees with summary.json")
    return dict(dir=str(d), summary=summary, manifest=manifest, rows=rows, outcome=outcome)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True, help="$EVAL_OUT (final_rollout/ inside)")
    ap.add_argument("--reference", required=True, help="HF download .../Elastic/eval/conditions")
    ap.add_argument("--parity", required=True, help="validation/parity_e0.json")
    ap.add_argument("--output", required=True); ap.add_argument("--csv-dir", default="")
    a = ap.parse_args()
    parity = json.loads(Path(a.parity).read_text())
    data = {(r, s): load_cl0(a.eval_root, r, s) for r in CL0 for s in SCHEDULES}
    data.update({(r, s): load_reference(a.reference, r, s) for r in ("E0", "E1", "E2") for s in SCHEDULES})
    conditions = {}
    for (r, s), d in data.items():
        if d is None:
            conditions[f"{r}/{s}"] = dict(run=r, schedule=s, complete=False); continue
        sm = d["summary"]; rows = d["rows"]
        conditions[f"{r}/{s}"] = dict(run=r, schedule=s, label=LABEL[s], complete=True, episodes=len(rows),
            successes=sum(d["outcome"].values()), success_rate=sum(d["outcome"].values()) / 6000,
            ci95=rate_ci(d["outcome"], stream(f"rate:{r}/{s}")),
            per_suite={SHORT[k]: sum(v for (su, *_), v in d["outcome"].items() if su == k) / 1500 for k in SUITES},
            per_seed={str(z): sum(v for (*_, sd), v in d["outcome"].items() if sd == z) / 2000 for z in (42, 43, 44)},
            checkpoint_sha256=sm["checkpoint_sha256"], backend=sm.get("backend"), source=d["dir"],
            reference=r in ("E0", "E1", "E2"))
        if r in CL0 and abs(conditions[f"{r}/{s}"]["success_rate"] - sm["success_rate"]) > 1e-12:
            raise ValueError(f"{r}/{s}: records disagree with the runner's summary.json")
        if a.csv_dir and r in CL0:
            out = Path(a.csv_dir) / f"{r}__{s}__episodes.csv"; out.parent.mkdir(parents=True, exist_ok=True)
            cols = ("suite", "task", "initial_state_index", "seed", "success", "policy_steps", "queries", "query_seconds")
            with out.open("w", newline="") as f:
                w = csv.writer(f); w.writerow(cols)
                for row in sorted(rows, key=lambda x: (x["suite"], x["task"], x["initial_state_index"], x["seed"])):
                    w.writerow([row[c] for c in cols])
    contrasts = {}
    for family, pairs in (("T", TEACHER), ("R", TRANSPORT)):
        for s in SCHEDULES:
            for x, y in pairs:
                if data[(x, s)] is None or data[(y, s)] is None: continue
                px, py = (dict(data[(z, s)]["manifest"]["protocol"]) for z in (x, y))
                for p_ in (px, py): p_.pop("videos", None)
                if px != py:
                    raise ValueError(f"{x} and {y} at {s} were scored under different protocols")
                name = f"{x}-{y}@{s}"; rng = stream(f"contrast:{name}")
                res = paired(data[(x, s)]["outcome"], data[(y, s)]["outcome"], rng)
                res["per_suite"] = {SHORT[k]: paired(data[(x, s)]["outcome"], data[(y, s)]["outcome"], rng, k) for k in SUITES}
                res["family"] = family
                res["role"] = f"primary ({family})" if s == "flat_k1_n2" else "secondary (unadjusted)"
                contrasts[name] = res
        primary = [k for k, c in contrasts.items() if c["role"] == f"primary ({family})"]
        if len(primary) == 3:
            for k, adj in zip(primary, holm([contrasts[k]["p_permutation"] for k in primary])):
                contrasts[k]["p_holm"] = adj
    result = dict(format="cl0-elastic-eval-v1", protocol="lightwam_20261009_matched via scripts/evaluate_rt_matched.py",
                  analysis=__doc__.strip(), resamples=RESAMPLES, rng_seed=SEED,
                  reference_reproduction=dict(episodes=parity["episodes"], identical=parity["identical"],
                                              same_outcome=parity["same_outcome"], exact=parity["passed"]),
                  conditions=conditions, contrasts=contrasts)
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    pct = lambda v: f"{100 * v:.2f}"
    lines = ["| Run | Schedule | Success % (95% CI) | Spatial | Object | Goal | Long | Seeds 42/43/44 |",
             "|---|---|---|---|---|---|---|---|"]
    for s in SCHEDULES:
        for r in CL0 + ("E0", "E1", "E2"):
            c = conditions[f"{r}/{s}"]; name = r if r in CL0 else f"{r} (BASE10, ref.)"
            if not c["complete"]:
                lines.append(f"| {name} | {LABEL[s]} | pending (incomplete; not scored) | | | | | |"); continue
            lines.append(f"| {name} | {c['label']} | {pct(c['success_rate'])} ({pct(c['ci95'][0])}-{pct(c['ci95'][1])}) | "
                         + " | ".join(pct(c["per_suite"][n]) for n in ("Spatial", "Object", "Goal", "Long")) + " | "
                         + "/".join(pct(c["per_seed"][z]) for z in ("42", "43", "44")) + " |")
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
