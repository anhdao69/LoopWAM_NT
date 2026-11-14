#!/usr/bin/env python3
"""Write README.md and loop_sweep_all.csv of this folder from breakdown.json (action loops 1 and 2) and from the report's
existing records of the same checkpoint at action loops 3, 5, 6 (matched_baselines) and 4 (final). Every number in the
tables is read from those files; nothing is typed in.

    python3 make_report.py
"""
import csv, json, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ER = HERE.parents[1]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
NAMES = dict(libero_spatial="Spatial", libero_object="Object", libero_goal="Goal", libero_10="Long")
SEEDS = ["42", "43", "44"]
ALIGN = {1: "4", 2: "3,4", 3: "2,3,4", 4: "1,2,3,4", 5: "1,1,2,3,4", 6: "1,1,1,2,3,4"}   # ../setup_report.md, "Native late alignment"

B = json.loads((HERE / "breakdown.json").read_text())
assert B["passed"]
R = B["runs"]


def std(v):
    return statistics.stdev(v)


def loops_table():
    """One row per action-loop count of the same checkpoint, each from its own record."""
    rows = []
    for ka in ("1", "2"):
        r = R[ka]
        rows.append(dict(action_loops=int(ka), record="original queue, 27 Oct 2026 (this folder)",
                         suites={s: (r["suites"][s]["successes"], r["suites"][s]["rate"]) for s in SUITES},
                         successes=r["successes"], episodes=r["episodes"], rate=r["success_rate"],
                         seeds={k: r["seeds"][k]["rate"] for k in SEEDS}))
    final = json.loads((ER / "final/results.json").read_text())
    m = [x for x in final["models"] if x["label"] == "LoopWAM full 4/4"]
    assert len(m) == 1 and m[0]["manifest"]["verified"]["checkpoint_sha256"] == R["1"]["checkpoint_sha256"]
    assert (m[0]["manifest"]["verified"]["video_loops"], m[0]["manifest"]["verified"]["action_loops"]) == (4, 4)
    # statements made in the text below about the back ends of the three records
    assert R["1"]["backend"] == R["2"]["backend"] == m[0]["manifest"]["arguments"]["backend"] == "cuda-graph"
    assert "local 2x RTX PRO 6000 Blackwell host (eager native inference)" in (ER / "matched_baselines/README.md").read_text()
    assert "These two histories must not be conflated" in (ER / "all_results.md").read_text()
    s4 = m[0]["summary"]
    rows.append(dict(action_loops=4, record="final campaign (final/results.json, \"LoopWAM full 4/4\"): the trained depth",
                     suites={s: (s4["per_suite"][s]["successes"], round(100 * s4["per_suite"][s]["successes"] / s4["per_suite"][s]["episodes"], 4)) for s in SUITES},
                     successes=s4["successes"], episodes=s4["episodes"], rate=round(100 * s4["successes"] / s4["episodes"], 4),
                     seeds={k: round(100 * sum(s4["per_suite"][s]["seeds"][k]["successes"] for s in SUITES) / 2000, 4) for k in SEEDS}))
    with open(ER / "matched_baselines/data/suite_results.csv") as f:
        mb = list(csv.DictReader(f))
    for ka in (3, 5, 6):
        v = json.loads((ER / f"matched_baselines/verification/loopwam_v0_ka{ka}.json").read_text())
        assert v["passed"] and set(v["checkpoint_sha256"].values()) == {R["1"]["checkpoint_sha256"]}
        assert ",".join(map(str, v["action_core_video_loops"])) == ALIGN[ka]
        sr = {x["suite"]: x for x in mb if x["model"] == f"LoopWAM v0 Ka{ka}"}
        assert set(sr) == set(SUITES) and sum(int(sr[s]["successes"]) for s in SUITES) == v["successes"]
        for s in SUITES: assert abs(100 * int(sr[s]["successes"]) / 1500 - v["suites"][s]["rate"]) < 1e-3
        rows.append(dict(action_loops=ka, record="matched-baseline campaign (matched_baselines/README.md)",
                         suites={s: (int(sr[s]["successes"]), v["suites"][s]["rate"]) for s in SUITES},
                         successes=v["successes"], episodes=v["episodes"], rate=v["success_rate"], seeds={k: v["seeds"][k] for k in SEEDS}))
    return sorted(rows, key=lambda r: r["action_loops"])


def main():
    L = loops_table()
    with open(HERE / "loop_sweep_all.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["action_loops", "video_loop_read_by_each_action_loop", "record", *[f"{NAMES[s].lower()}_successes" for s in SUITES],
                    *[f"{NAMES[s].lower()}_sr" for s in SUITES], "successes", "episodes", "overall_sr", "seed42_sr", "seed43_sr", "seed44_sr"])
        for r in L:
            w.writerow([r["action_loops"], ALIGN[r["action_loops"]].replace(",", " "), r["record"], *[r["suites"][s][0] for s in SUITES],
                        *[f"{r['suites'][s][1]:.4f}" for s in SUITES], r["successes"], r["episodes"], f"{r['rate']:.4f}", *[f"{r['seeds'][k]:.2f}" for k in SEEDS]])

    o = []
    o += ["# v0 4/4 at 1 and 2 action loops: breakdown by suite, seed and task", "",
          "Added 2026-11-14. Status: **complete**. Nothing was re-evaluated: these are the two scored runs of the "
          "[original action-depth queue](../report.md) (27 October 2026), whose report had the totals only. The numbers below are counted from "
          "the runs' saved per-episode records.", "",
          "Setup of both runs ([setup report](../setup_report.md)): the released v0 checkpoint trained with video/action loops 4/4 "
          f"(policy SHA-256 `{R['1']['checkpoint_sha256']}`), video loops kept at 4, only the number of action loops changed at inference, "
          "10 denoising steps, standard LIBERO, 4 suites × 10 tasks × 50 initial states × seeds 42/43/44 = 6,000 episodes per run.", ""]

    o += ["## Result by suite", "", "| Action loops | Spatial | Object | Goal | Long | Overall |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for ka in ("1", "2"):
        r = R[ka]
        o.append(f"| {ka} | " + " | ".join(f"{r['suites'][s]['rate']:.2f}% ({r['suites'][s]['successes']}/{r['suites'][s]['episodes']})" for s in SUITES)
                 + f" | **{r['success_rate']:.2f}%** ({r['successes']}/{r['episodes']}) |")
    o += ["", "[suite_results.csv](suite_results.csv)", ""]

    o += ["## Result by seed", "", "Success rate in percent; each suite cell is 500 episodes, each overall cell 2,000. "
          "The last column is the mean of the three seeds ± their sample standard deviation.", "",
          "| Action loops | Suite | Seed 42 | Seed 43 | Seed 44 | Mean ± std |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for ka in ("1", "2"):
        r = R[ka]
        for s in SUITES:
            x = r["suites"][s]
            o.append(f"| {ka} | {NAMES[s]} | " + " | ".join(f"{x['seeds'][k]['rate']:.2f}" for k in SEEDS) + f" | {x['rate']:.2f} ± {x['seed_sample_std']:.2f} |")
        o.append(f"| {ka} | **Overall** | " + " | ".join(f"**{r['seeds'][k]['rate']:.2f}**" for k in SEEDS) + f" | **{r['success_rate']:.2f} ± {r['seed_sample_std']:.2f}** |")
    o.append("")
    top = all(R[ka]["suites"][s]["seeds"]["42"]["rate"] > max(R[ka]["suites"][s]["seeds"]["43"]["rate"], R[ka]["suites"][s]["seeds"]["44"]["rate"])
              for ka in ("1", "2") for s in SUITES)
    low = all(R[ka]["suites"][s]["seeds"]["44"]["rate"] <= R[ka]["suites"][s]["seeds"]["43"]["rate"] for ka in ("1", "2") for s in SUITES)
    assert R["1"]["same_initial_states_across_seeds"] and R["2"]["same_initial_states_across_seeds"]
    if top and low:
        o += ["Seed 42 is the highest and seed 44 the lowest (or tied lowest) in every suite of both runs: overall "
              + " / ".join(f"{R['1']['seeds'][k]['rate']:.2f}" for k in SEEDS) + "% at 1 loop and "
              + " / ".join(f"{R['2']['seeds'][k]['rate']:.2f}" for k in SEEDS) + "% at 2 loops. The three seeds start every task from the same 50 "
              "initial states (identical initial-state hashes in the records). The cause of the seed spread was not investigated here.", ""]

    o += ["## Result by task", "", "Successes out of 150 episodes per task (3 seeds × 50 initial states). [per_task.csv](per_task.csv) has the per-seed counts.", ""]
    for s in SUITES:
        o += [f"### {NAMES[s]}", "", "| Task | Instruction | 1 loop | 2 loops |", "| ---: | --- | ---: | ---: |"]
        for t in map(str, range(10)):
            a, b = R["1"]["suites"][s]["tasks"][t], R["2"]["suites"][s]["tasks"][t]
            assert a["description"] == b["description"]
            o.append(f"| {t} | {a['description']} | {a['successes']} ({a['rate']:.1f}%) | {b['successes']} ({b['rate']:.1f}%) |")
        o += [f"| | **{NAMES[s]} total** | **{R['1']['suites'][s]['successes']}/{R['1']['suites'][s]['episodes']}** | **{R['2']['suites'][s]['successes']}/{R['2']['suites'][s]['episodes']}** |", ""]
    zero = {ka: {s: sum(x["successes"] == 0 for x in R[ka]["suites"][s]["tasks"].values()) for s in SUITES} for ka in ("1", "2")}
    o += ["Tasks with no success in 150 episodes: at 1 loop " + ", ".join(f"{zero['1'][s]} of 10 in {NAMES[s]}" for s in SUITES)
          + "; at 2 loops " + ", ".join(f"{zero['2'][s]} in {NAMES[s]}" for s in SUITES) + ".", ""]

    o += ["## All action-loop counts of this checkpoint in one view", "",
          "The same checkpoint has been scored at six action-loop counts, in three separate records. The rows for 3, 4, 5 and 6 loops are copied "
          "from those records, not from this folder's data. All rows use the same checkpoint, video loops 4, 10 denoising steps, seeds 42/43/44 "
          "and 6,000 episodes, but they were not run as one queue: 1 and 2 loops are the original queue (cuda-graph back end); 4 loops is the "
          "earlier final campaign (cuda-graph back end); 3, 5 and 6 loops are the later matched-baseline campaign, which its README describes as "
          "run on a \"local 2x RTX PRO 6000 Blackwell host (eager native inference)\". The consolidated report says the original queue and the matched-baseline "
          "campaign must not be conflated; the table puts them side by side for reading only.", "",
          "| Action loops | Video loop read by each action loop | Spatial | Object | Goal | Long | Overall | Successes | Seed 42 / 43 / 44 | Record |",
          "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    link = {1: "[this folder](breakdown.json)", 2: "[this folder](breakdown.json)", 3: "[matched baselines](../../matched_baselines/README.md)",
            5: "[matched baselines](../../matched_baselines/README.md)", 6: "[matched baselines](../../matched_baselines/README.md)",
            4: "[final campaign](../../final/results.json), trained depth"}
    for r in L:
        o.append(f"| {r['action_loops']} | {ALIGN[r['action_loops']]} | " + " | ".join(f"{r['suites'][s][1]:.2f}%" for s in SUITES)
                 + f" | **{r['rate']:.2f}%** | {r['successes']}/{r['episodes']} | " + " / ".join(f"{r['seeds'][k]:.2f}" for k in SEEDS) + f" | {link[r['action_loops']]} |")
    by = {r["action_loops"]: r for r in L}
    o += ["", "[loop_sweep_all.csv](loop_sweep_all.csv) holds the same rows with the per-suite success counts.", "",
          "Read from the table:", "",
          f"- Overall success by action loops 1 → 6: " + " / ".join(f"{by[k]['rate']:.2f}" for k in range(1, 7)) + "%.",
          f"- Long is the lowest suite at every loop count: " + " / ".join(f"{by[k]['suites']['libero_10'][1]:.2f}" for k in range(1, 7)) + "%.",
          f"- Against the trained 4 loops, 5 loops differs by {by[5]['rate'] - by[4]['rate']:+.2f} points overall, 3 loops by {by[3]['rate'] - by[4]['rate']:+.2f}, "
          f"6 loops by {by[6]['rate'] - by[4]['rate']:+.2f}, 2 loops by {by[2]['rate'] - by[4]['rate']:+.2f} and 1 loop by {by[1]['rate'] - by[4]['rate']:+.2f}.", ""]
    assert all(min(by[k]["suites"], key=lambda s: by[k]["suites"][s][1]) == "libero_10" for k in range(1, 7))

    o += ["## Source and checks", "",
          "- Source: `/data14t/anhdh35/loopwam_v0_ka_sweep_20261027/results/ka1` and `ka2` on the workstation where the queue ran. Copied on "
          f"2026-11-14, read-only: the {R['1']['group_files']} + {R['2']['group_files']} task/seed group files (one record per scored episode), each run's "
          "`summary.json`, `manifest.json` and `progress.json`, and the queue's two state files. "
          f"[source/ka12_groups.tar.gz](source/ka12_groups.tar.gz) ({B['archive_files']} files, SHA-256 `{B['archive_sha256']}`) with the workstation's own "
          "`sha256sum` listing in [source/workstation_sha256.txt](source/workstation_sha256.txt); every copied file matches that listing.",
          "- Link to the existing report: the workstation's `queue_state.json` is byte-identical to [../queue_status.json](../queue_status.json), and each "
          "run's `summary.json` equals the summary stored there.",
          "- Checks made by [analyze_ka12.py](analyze_ka12.py) before counting: 120 group files per run with 50 episodes each (episodes and initial states "
          "0–49), the full grid of 4 suites × 10 tasks × 3 seeds, every episode recorded with this checkpoint, action loops 1 or 2, video loops 4 and "
          "10 denoising steps, and the episode totals equal to the run's own `summary.json` "
          f"({R['1']['successes']}/{R['1']['episodes']} and {R['2']['successes']}/{R['2']['episodes']}).",
          "- [Independent check](../../verify_v0_ka_breakdown_20261114.py) (separate code): recounts every table of this page from the archive and "
          "compares the copied rows for 3, 4, 5 and 6 loops with their records. Result in [independent_check.json](independent_check.json).",
          "- Not stored here: the rollout videos (2.6 GB and 2.2 GB), the per-query logs and the per-episode copies of the same records. They "
          "remain on the workstation under the path above.", "",
          "## Files", "",
          "| File | Content |", "| --- | --- |",
          "| [breakdown.json](breakdown.json) | Both runs: totals, suites, seeds, tasks, checks |",
          "| [suite_results.csv](suite_results.csv) | One row per run and suite, with the three seeds |",
          "| [per_task.csv](per_task.csv) | One row per run, suite and task, with per-seed success counts |",
          "| [episodes_ka1.csv](episodes_ka1.csv) · [episodes_ka2.csv](episodes_ka2.csv) | One row per scored episode (6,000 each): suite, task, seed, initial state, success, policy steps, queries |",
          "| [loop_sweep_all.csv](loop_sweep_all.csv) | The six action-loop counts side by side, each row with its record |",
          "| [analyze_ka12.py](analyze_ka12.py) · [make_report.py](make_report.py) | The scripts that produced the files above and this page |",
          "| [source/](source/ka12_groups.tar.gz) | The copied run records and the workstation's hash listing |", ""]
    (HERE / "README.md").write_text("\n".join(o))
    print("\n".join(o[:14])); print("..."); print("\n".join(x for x in o if x.startswith(("| 1 |", "| 2 |", "| 3 |", "| 4 |", "| 5 |", "| 6 |")) and "Record" not in x and "[" in x))
    print("\n".join(x for x in o if x.startswith(("Seed 42", "Tasks with", "- Overall", "- Long", "- Against"))))


if __name__ == "__main__":
    main()
