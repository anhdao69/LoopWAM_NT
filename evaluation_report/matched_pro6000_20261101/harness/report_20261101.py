#!/usr/bin/env python3
"""Results of the 2026-11-01 matched evaluation batch, as declared in ANALYSIS_PLAN_20261101.md (written before any
scored episode): per-suite / per-seed rates, and secondary unadjusted paired contrasts with a 95% cluster-bootstrap
interval and a two-sided cluster sign-flip permutation p (cluster = suite, task, initial state; its three seeds).

A condition is reported only when its full grid (4 suites x 10 tasks x 50 initial states x seeds 42/43/44) exists and
every record matches the condition's checkpoint SHA-256 (and schedule, for the elastic runner). Incomplete conditions
are listed with their progress only.
"""
import argparse, csv, glob, json, zlib
from pathlib import Path
import numpy as np

ROOT = Path("/data/anh/code/wam_baselines")
CAMPAIGN = Path("/data/anh/code/LoopWAM_NT/evaluation_report/final/episodes.csv")
SEED, RESAMPLES = 20261101, 10000
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SHORT = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}
SEEDS = (42, 43, 44)
GRID = {(s, t, seed, i) for s in SUITES for t in range(10) for seed in SEEDS for i in range(50)}
SHA = dict(E0="5569c47fa6c91e136a611d2b819a053c94f6b2956a7f602bbed4db23c69f60e8",
           Z10="1c659a7c2f6970c3ea5972a8fba4abd9db35857c83a67a053c0ee35555fb602b",
           v4a2="33476d5808965252eaa3f5e996b5f5f75cc0542314f825a0fc07fbdec2f0c9d4")


TEST_TIME = ("flat_k3_n10", "flat_k5_n10", "flat_k6_n10")


class Incomplete(Exception):
    pass


def elastic(run, sched, root):
    dirs = glob.glob(str(Path(root) / "final_rollout" / f"matched6000pro__{run}__{sched}__*"))
    if len(dirs) != 1 or not dirs[0].endswith(SHA[run]):
        raise Incomplete(f"{len(dirs)} output folders")
    out, extra = {}, dict(exact_half_gripper=0, heads=set())
    for f in glob.glob(f"{dirs[0]}/groups/*.json"):
        r = json.loads(Path(f).read_text())
        if r["checkpoint_sha256"] != SHA[run] or r["schedule"] != sched or len(r["episodes"]) != 50:
            raise ValueError(f"bad group record {f}")
        extra["heads"].add(r.get("source_git_head"))
        for e in r["episodes"]:
            if e["episode"] != e["initial_state_index"] or (e["suite"], e["task"], e["seed"]) != (
                    r["group"]["suite"], r["group"]["task"], r["group"]["seed"]):
                raise ValueError(f"inconsistent episode in {f}")
            key = (e["suite"], e["task"], e["seed"], e["initial_state_index"])
            if key in out:
                raise ValueError(f"duplicate episode {key}")
            out[key] = bool(e["success"]); extra["exact_half_gripper"] += e.get("exact_half_gripper", 0)
    m = json.loads(Path(dirs[0], "manifest.json").read_text()); pr = m["protocol"]
    want = dict(sha256=SHA[run], schedule=sched, backend="cuda-graph", source_dirty=False, mujoco_gl="egl",
                zero_shot_kind="T0" if run == "Z10" else None, test_time_schedule=sched in TEST_TIME,
                text_cache_dir=str(ROOT / "text_cache/loopwam"))
    bad = {k: m.get(k) for k, v in want.items() if m.get(k) != v}
    if run == "E0":
        n = int(sched.split("_")[1][1:]); bad.update({} if m["depths"] == [n] * 10 else dict(depths=m["depths"]))
    if set(pr["suites"]) != set(SUITES) or (pr["seeds"], pr["tasks"], pr["episodes"], pr["settling_steps"],
                                             pr["replan_steps"], pr["action_horizon"], pr["camera"], pr["max_policy_steps"]) != (
            list(SEEDS), list(range(10)), 50, 30, 10, 32, 256,
            dict(libero_spatial=400, libero_object=400, libero_goal=400, libero_10=700)):
        bad["protocol"] = pr
    if bad:
        raise ValueError(f"{run} {sched}: manifest differs from the plan: {bad}")
    extra["summary_json"] = "absent"
    if len(out) == 6000 and Path(dirs[0], "summary.json").exists():
        # the runner writes summary.json only when a lane finishes its whole queue, so it can lag the group records
        summ = json.loads(Path(dirs[0], "summary.json").read_text())
        if summ["episodes"] != 6000 or summ["successes"] != sum(out.values()):
            raise ValueError(f"{run} {sched}: summary.json disagrees with the group records")
        extra["summary_json"] = "matches"
    return out, dict(extra, heads=sorted(map(str, extra["heads"])), manifest_head=m["source_git_head"])


def harness(name, sha):
    out = {}
    for f in glob.glob(str(ROOT / f"runs/{name}/episodes/*.json")):
        e = json.loads(Path(f).read_text())
        if e["checkpoint_sha256"] != sha or e["episode"] != e["initial_state_index"]:
            raise ValueError(f"bad episode record {f}")
        out[(e["suite"], e["task"], e["seed"], e["initial_state_index"])] = bool(e["success"])
    return out, {}


def h100(sched):
    p = ROOT / f"checkpoints/loopwam_hf/Elastic/eval/conditions/E0__{sched}/episodes.csv"
    return {(r["suite"], int(r["task"]), int(r["seed"]), int(r["initial_state_index"])): r["success"] == "True"
            for r in csv.DictReader(p.open())}, {}


def campaign(model):
    return {(r["suite"], int(r["task"]), int(r["seed"]), int(r["initial_state_index"])): r["success"] == "True"
            for r in csv.DictReader(CAMPAIGN.open()) if r["model"] == model}, {}


def rates(o):
    def rate(keys):
        return dict(successes=sum(o[k] for k in keys), episodes=len(keys), success_rate=sum(o[k] for k in keys) / len(keys))
    per_suite = {s: dict(rate([k for k in o if k[0] == s]),
                         seeds={str(seed): rate([k for k in o if k[0] == s and k[2] == seed]) for seed in SEEDS})
                 for s in SUITES}
    return dict(rate(list(o)), per_suite=per_suite, per_seed={str(seed): rate([k for k in o if k[2] == seed]) for seed in SEEDS})


def clusters(o):
    acc = {}
    for (s, t, seed, i), y in o.items():
        acc.setdefault((s, t, i), []).append(y)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def stream(name):
    return np.random.default_rng([SEED, zlib.crc32(name.encode())])


def rate_ci(o, name):
    c = np.array(list(clusters(o).values())); rng = stream(f"rate:{name}")
    b = c[rng.integers(0, len(c), size=(RESAMPLES, len(c)))].mean(axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def paired(a, b, name, suite=None):
    if set(a) != set(b):
        raise ValueError(f"{name}: arms have different episode keys")
    if suite:
        a = {k: v for k, v in a.items() if k[0] == suite}; b = {k: v for k, v in b.items() if k[0] == suite}
    ca, cb = clusters(a), clusters(b); keys = sorted(ca)
    d = np.array([ca[k] - cb[k] for k in keys]); est = d.mean(); rng = stream(name + (f":{suite}" if suite else ""))
    boots = d[rng.integers(0, len(d), size=(RESAMPLES, len(d)))].mean(axis=1)
    null = (rng.choice((-1.0, 1.0), size=(RESAMPLES, len(d))) * d).mean(axis=1)
    p = (1 + np.sum(np.abs(null) >= abs(est) - 1e-12)) / (1 + RESAMPLES)
    return dict(difference=float(est), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                p_permutation=float(p), clusters=len(d), discordant_clusters=int(np.sum(d != 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elastic-root", default=str(ROOT / "runs/elastic_matched_20261101"))
    ap.add_argument("--output", required=True, help="results JSON; Markdown written next to it")
    a = ap.parse_args()
    E = a.elastic_root
    sources = {  # name: (loader, host label)
        "v0 4/2": (lambda: harness("loopwam_v0_v4a2_matched", SHA["v4a2"]), "RTX PRO 6000 (this batch)"),
        "E0 [3]x10": (lambda: elastic("E0", "flat_k3_n10", E), "RTX PRO 6000 (this batch)"),
        "E0 [5]x10": (lambda: elastic("E0", "flat_k5_n10", E), "RTX PRO 6000 (this batch)"),
        "E0 [6]x10": (lambda: elastic("E0", "flat_k6_n10", E), "RTX PRO 6000 (this batch)"),
        "Z10 [4,4]": (lambda: elastic("Z10", "parent2", E), "RTX PRO 6000 (this batch)"),
        "Z10 [4]x5": (lambda: elastic("Z10", "parent5", E), "RTX PRO 6000 (this batch)"),
        "Z10 [1,1]": (lambda: elastic("Z10", "flat_k1_n2", E), "RTX PRO 6000 (this batch)"),
        # references
        "base v0 Ka3": (lambda: harness("loopwam_v0_ka3_matched", SHA["Z10"]), "RTX PRO 6000 (2026-10-27/28)"),
        "base v0 Ka5": (lambda: harness("loopwam_v0_ka5_matched", SHA["Z10"]), "RTX PRO 6000 (2026-10-27/28)"),
        "base v0 Ka6": (lambda: harness("loopwam_v0_ka6_matched", SHA["Z10"]), "RTX PRO 6000 (2026-10-27/28)"),
        "campaign v0 4/4": (lambda: campaign("LoopWAM full 4/4"), "RTX PRO 6000 (campaign)"),
        "campaign v0 4/1": (lambda: campaign("LoopWAM full video4/action1"), "RTX PRO 6000 (campaign)"),
        "campaign v0 2/2": (lambda: campaign("LoopWAM full video2/action2"), "RTX PRO 6000 (campaign)"),
        "campaign v0 1/4": (lambda: campaign("LoopWAM full video1/action4"), "RTX PRO 6000 (campaign)"),
        "E0 [4]x10 (H100)": (lambda: h100("flat_k4_n10"), "H100 (elastic evaluation)"),
        "E0 [4,4] (H100)": (lambda: h100("flat_k4_n2"), "H100 (elastic evaluation)"),
        "E0 [4]x5 (H100)": (lambda: h100("flat_k4_n5"), "H100 (elastic evaluation)"),
        "E0 [1,1] (H100)": (lambda: h100("flat_k1_n2"), "H100 (elastic evaluation)"),
    }
    data, results = {}, dict(plan="ANALYSIS_PLAN_20261101.md", conditions={}, incomplete={}, contrasts={})
    for name, (load, host) in sources.items():
        try:
            o, extra = load()
        except Incomplete as e:
            results["incomplete"][name] = str(e); continue
        if set(o) != GRID:
            results["incomplete"][name] = f"{len(set(o) & GRID)}/6000 episodes"; continue
        data[name] = o
        results["conditions"][name] = dict(rates(o), ci95=rate_ci(o, name), host=host, **extra)
    same_host = [("E0 [3]x10", "base v0 Ka3"), ("E0 [5]x10", "base v0 Ka5"), ("E0 [6]x10", "base v0 Ka6"),
                 ("Z10 [4,4]", "Z10 [4]x5"), ("Z10 [4,4]", "Z10 [1,1]"), ("Z10 [4]x5", "Z10 [1,1]"),
]
    # The plan listed the campaign contrasts under "cross-host"; the campaign (evaluation_report/final) was in fact scored on
    # the same hardware model (RTX PRO 6000 Blackwell Max-Q, Threadripper 9960X; driver 580.178.04 then, 580.105.08 now),
    # and gate G4 reproduced its v0 4/4 trajectories 100/100 here, so they get their own label.
    campaign_hw = [("Z10 [4,4]", "campaign v0 4/4"), ("Z10 [4]x5", "campaign v0 4/4"), ("Z10 [1,1]", "campaign v0 4/4"),
                   ("v0 4/2", "campaign v0 4/4"), ("v0 4/2", "campaign v0 4/1"), ("v0 4/2", "campaign v0 2/2"),
                   ("v0 4/2", "campaign v0 1/4")]
    cross_host = [("Z10 [4,4]", "E0 [4,4] (H100)"), ("Z10 [4]x5", "E0 [4]x5 (H100)"), ("Z10 [1,1]", "E0 [1,1] (H100)"),
                  ("E0 [3]x10", "E0 [4]x10 (H100)"), ("E0 [5]x10", "E0 [4]x10 (H100)"), ("E0 [6]x10", "E0 [4]x10 (H100)")]
    for kind, pairs in (("same_host", same_host), ("same_hardware_campaign", campaign_hw), ("cross_host", cross_host)):
        for x, y in pairs:
            if x in data and y in data:
                name = f"{x} - {y}"
                results["contrasts"][name] = dict(kind=kind, overall=paired(data[x], data[y], name),
                                                  per_suite={s: paired(data[x], data[y], name, s) for s in SUITES})
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    pct = lambda v: f"{100 * v:.2f}"  # noqa: E731
    md = ["| Condition | Host | Overall (95% CI) | " + " | ".join(SHORT[s] for s in SUITES) + " | seed 42 | seed 43 | seed 44 |",
          "|---|---|---|" + "---|" * 7]
    for name, r in results["conditions"].items():
        md.append(f"| {name} | {r['host']} | {pct(r['success_rate'])} ({pct(r['ci95'][0])}-{pct(r['ci95'][1])}) | "
                  + " | ".join(pct(r["per_suite"][s]["success_rate"]) for s in SUITES) + " | "
                  + " | ".join(pct(r["per_seed"][str(seed)]["success_rate"]) for seed in SEEDS) + " |")
    md += ["", "| Contrast (secondary, unadjusted) | Kind | Difference, pp (95% CI) | p | "
           + " | ".join(SHORT[s] for s in SUITES) + " |", "|---|---|---|---|" + "---|" * 4]
    for name, c in results["contrasts"].items():
        o = c["overall"]
        md.append(f"| {name} | {c['kind'].replace('_', ' ')} | {100 * o['difference']:+.2f} ({100 * o['ci95'][0]:+.2f} to "
                  f"{100 * o['ci95'][1]:+.2f}) | {o['p_permutation']:.4f} | "
                  + " | ".join(f"{100 * c['per_suite'][s]['difference']:+.2f}" for s in SUITES) + " |")
    if results["incomplete"]:
        md += ["", "Not reported (incomplete): " + "; ".join(f"{k}: {v}" for k, v in results["incomplete"].items())]
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
