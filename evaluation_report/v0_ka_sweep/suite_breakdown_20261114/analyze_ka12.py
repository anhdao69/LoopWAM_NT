#!/usr/bin/env python3
"""Per-suite / per-seed / per-task breakdown of the two scored runs of the original v0 action-depth queue (Ka=1, Ka=2).

Input: source/ka12_groups.tar.gz, a copy of the runs' saved records on the workstation (results/ka{1,2}/groups/*.json:
one record per scored episode; summary.json; manifest.json), and source/workstation_sha256.txt, the sha256sum listing
taken on the workstation. Nothing is re-evaluated. The archive is read in memory; nothing is extracted.

Output (this folder): breakdown.json, suite_results.csv, per_task.csv, episodes_ka1.csv, episodes_ka2.csv.

    python3 analyze_ka12.py
"""
import csv, hashlib, io, json, statistics, sys, tarfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
NAMES = dict(libero_spatial="Spatial", libero_object="Object", libero_goal="Goal", libero_10="Long")
SEEDS = [42, 43, 44]
CHECKPOINT = "1c659a7c2f6970c3ea5972a8fba4abd9db35857c83a67a053c0ee35555fb602b"
STEP_CAPS = dict(libero_spatial=400, libero_object=400, libero_goal=400, libero_10=700)


def rate(s, n):
    return round(100. * s / n, 4)


def load():
    listing = dict((p, h) for h, p in (line.split(None, 1) for line in (HERE / "source/workstation_sha256.txt").read_text().splitlines()))
    files = {}
    with tarfile.open(HERE / "source/ka12_groups.tar.gz") as tar:
        for m in tar.getmembers():
            if m.isfile():
                files[m.name] = tar.extractfile(m).read()
    assert set(files) == set(listing), "archive members differ from the workstation listing"
    bad = [p for p, b in files.items() if hashlib.sha256(b).hexdigest() != listing[p]]
    assert not bad, f"sha256 differs from the workstation listing: {bad[:3]}"
    return files


def analyze(ka, files):
    root = f"results/ka{ka}/"
    summary = json.loads(files[root + "summary.json"]); manifest = json.loads(files[root + "manifest.json"])
    groups = sorted(p for p in files if p.startswith(root + "groups/"))
    episodes = []; issues = []
    for p in groups:
        name = p.rsplit("/", 1)[1][:-5]; rows = json.loads(files[p])
        if {r["id"] for r in rows} != {name} or len(rows) != 50: issues.append(f"{name}: {len(rows)} rows")
        if sorted(r["episode"] for r in rows) != list(range(50)) or sorted(r["initial_state_index"] for r in rows) != list(range(50)):
            issues.append(f"{name}: episodes / initial states are not 0..49")
        for r in rows:
            if (r["checkpoint_sha256"], r["action_loops"], r["video_loops"], r["nfe"]) != (CHECKPOINT, ka, 4, 10):
                issues.append(f"{name}: identity {r['action_loops']}/{r['video_loops']}/{r['nfe']}")
            if not isinstance(r["success"], bool) or r["policy_steps"] > STEP_CAPS[r["suite"]]: issues.append(f"{name} episode {r['episode']}: success/steps")
            if f"{r['suite']}_seed{r['seed']}_task{r['task']}" != r["id"]: issues.append(f"{name}: id fields")
        episodes += rows
    have = defaultdict(int)
    for r in episodes: have[(r["suite"], r["seed"], r["task"])] += 1
    if set(have) != {(s, seed, t) for s in SUITES for seed in SEEDS for t in range(10)} or set(have.values()) != {50}:
        issues.append("grid is not 4 suites x 10 tasks x 3 seeds x 50 episodes")
    if sorted(g["id"] if isinstance(g, dict) else g for g in manifest["groups"]) != sorted(p.rsplit("/", 1)[1][:-5] for p in groups):
        issues.append("group files differ from the run manifest's group list")
    if (manifest["action_loops"], manifest["video_loops"], manifest["nfe"], manifest["checkpoint_sha256"], manifest["mode"]) != (ka, 4, 10, CHECKPOINT, "scored"):
        issues.append("run manifest identity")
    total = sum(r["success"] for r in episodes)
    if (len(episodes), total) != (summary["episodes"], summary["successes"]) or summary["action_loops"] != ka \
            or summary["checkpoint_sha256"] != CHECKPOINT or summary["mode"] != "scored":
        issues.append(f"group files give {total}/{len(episodes)}, summary.json {summary['successes']}/{summary['episodes']}")
    # the three seeds start every task from the same 50 initial states
    states = defaultdict(set)
    for r in episodes: states[(r["suite"], r["task"], r["initial_state_index"])].add(r["initial_state_sha256"])
    same_states = all(len(v) == 1 for v in states.values())
    suites = {}
    for s in SUITES:
        rows = [r for r in episodes if r["suite"] == s]
        seeds = {str(seed): dict(successes=sum(r["success"] for r in rows if r["seed"] == seed), episodes=500,
                                 rate=rate(sum(r["success"] for r in rows if r["seed"] == seed), 500)) for seed in SEEDS}
        tasks = {}
        for t in range(10):
            tr = [r for r in rows if r["task"] == t]
            if len({r["description"] for r in tr}) != 1: issues.append(f"{s} task {t}: several descriptions")
            tasks[str(t)] = dict(description=tr[0]["description"], successes=sum(r["success"] for r in tr), episodes=len(tr),
                                 rate=rate(sum(r["success"] for r in tr), len(tr)),
                                 seeds={str(seed): sum(r["success"] for r in tr if r["seed"] == seed) for seed in SEEDS})
        suites[s] = dict(name=NAMES[s], successes=sum(r["success"] for r in rows), episodes=len(rows),
                         rate=rate(sum(r["success"] for r in rows), len(rows)),
                         seed_sample_std=round(statistics.stdev(v["rate"] for v in seeds.values()), 4), seeds=seeds, tasks=tasks)
    seeds = {str(seed): dict(successes=sum(r["success"] for r in episodes if r["seed"] == seed), episodes=2000,
                             rate=rate(sum(r["success"] for r in episodes if r["seed"] == seed), 2000)) for seed in SEEDS}
    with open(HERE / f"episodes_ka{ka}.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["action_loops", "suite", "task", "seed", "episode", "initial_state_index", "success", "policy_steps", "queries"])
        for r in sorted(episodes, key=lambda r: (SUITES.index(r["suite"]), r["task"], r["seed"], r["episode"])):
            w.writerow([ka, r["suite"], r["task"], r["seed"], r["episode"], r["initial_state_index"], int(r["success"]), r["policy_steps"], r["queries"]])
    listing = hashlib.sha256("".join(f"{hashlib.sha256(files[p]).hexdigest()}  {p}\n" for p in groups).encode()).hexdigest()
    return dict(action_loops=ka, video_loops=4, nfe=10, checkpoint_sha256=CHECKPOINT, episodes=len(episodes), successes=total,
                success_rate=rate(total, len(episodes)), seed_sample_std=round(statistics.stdev(v["rate"] for v in seeds.values()), 4),
                suites=suites, seeds=seeds, same_initial_states_across_seeds=same_states,
                backend=summary["backend"], workers=summary["workers"], group_files=len(groups), group_files_listing_sha256=listing,
                run_summary=summary, issues=issues, passed=not issues)


def main():
    files = load()
    runs = {str(ka): analyze(ka, files) for ka in (1, 2)}
    result = dict(source="results/ka1 and results/ka2 of the original v0 action-depth queue (workstation, 27 October 2026): the runs' saved "
                         "per-episode group files; nothing re-evaluated",
                  archive_sha256=hashlib.sha256((HERE / "source/ka12_groups.tar.gz").read_bytes()).hexdigest(), archive_files=len(files),
                  runs=runs, passed=all(r["passed"] for r in runs.values()))
    (HERE / "breakdown.json").write_text(json.dumps(result, indent=2) + "\n")
    with open(HERE / "suite_results.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["action_loops", "suite", "successes", "episodes", "sr_percent", "seed42_sr", "seed43_sr", "seed44_sr", "seed_sample_std"])
        for ka, r in runs.items():
            for s in SUITES:
                x = r["suites"][s]; w.writerow([ka, s, x["successes"], x["episodes"], x["rate"], *[x["seeds"][str(seed)]["rate"] for seed in SEEDS], x["seed_sample_std"]])
            w.writerow([ka, "all", r["successes"], r["episodes"], r["success_rate"], *[r["seeds"][str(seed)]["rate"] for seed in SEEDS], r["seed_sample_std"]])
    with open(HERE / "per_task.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["action_loops", "suite", "task", "description", "successes", "episodes", "sr_percent", "seed42", "seed43", "seed44"])
        for ka, r in runs.items():
            for s in SUITES:
                for t, x in r["suites"][s]["tasks"].items():
                    w.writerow([ka, s, t, x["description"], x["successes"], x["episodes"], x["rate"], *[x["seeds"][str(seed)] for seed in SEEDS]])
    for ka, r in runs.items():
        print(f"Ka{ka}: {r['successes']}/{r['episodes']} = {r['success_rate']}%  passed={r['passed']} issues={r['issues'][:3]}")
        for s in SUITES:
            x = r["suites"][s]
            print(f"   {x['name']:8s} {x['successes']:5d}/{x['episodes']} = {x['rate']:6.2f}%  seeds {[v['rate'] for v in x['seeds'].values()]}")
        print("   seeds overall", [v["rate"] for v in r["seeds"].values()], "same initial states across seeds:", r["same_initial_states_across_seeds"])
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
