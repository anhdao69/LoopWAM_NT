# v0 4/4 at 1 and 2 action loops: breakdown by suite, seed and task

Added 2026-11-14. Status: **complete**. Nothing was re-evaluated: these are the two scored runs of the [original action-depth queue](../report.md) (27 October 2026), whose report had the totals only. The numbers below are counted from the runs' saved per-episode records.

Setup of both runs ([setup report](../setup_report.md)): the released v0 checkpoint trained with video/action loops 4/4 (policy SHA-256 `1c659a7c2f6970c3ea5972a8fba4abd9db35857c83a67a053c0ee35555fb602b`), video loops kept at 4, only the number of action loops changed at inference, 10 denoising steps, standard LIBERO, 4 suites × 10 tasks × 50 initial states × seeds 42/43/44 = 6,000 episodes per run.

## Result by suite

| Action loops | Spatial | Object | Goal | Long | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.73% (281/1500) | 25.13% (377/1500) | 5.07% (76/1500) | 0.27% (4/1500) | **12.30%** (738/6000) |
| 2 | 65.07% (976/1500) | 72.13% (1082/1500) | 54.60% (819/1500) | 30.47% (457/1500) | **55.57%** (3334/6000) |

[suite_results.csv](suite_results.csv)

## Result by seed

Success rate in percent; each suite cell is 500 episodes, each overall cell 2,000. The last column is the mean of the three seeds ± their sample standard deviation.

| Action loops | Suite | Seed 42 | Seed 43 | Seed 44 | Mean ± std |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Spatial | 23.20 | 17.20 | 15.80 | 18.73 ± 3.93 |
| 1 | Object | 42.60 | 19.20 | 13.60 | 25.13 ± 15.38 |
| 1 | Goal | 9.00 | 4.80 | 1.40 | 5.07 ± 3.81 |
| 1 | Long | 0.80 | 0.00 | 0.00 | 0.27 ± 0.46 |
| 1 | **Overall** | **18.90** | **10.30** | **7.70** | **12.30 ± 5.86** |
| 2 | Spatial | 68.00 | 64.40 | 62.80 | 65.07 ± 2.66 |
| 2 | Object | 83.80 | 68.80 | 63.80 | 72.13 ± 10.41 |
| 2 | Goal | 60.00 | 56.00 | 47.80 | 54.60 ± 6.22 |
| 2 | Long | 41.60 | 27.80 | 22.00 | 30.47 ± 10.07 |
| 2 | **Overall** | **63.35** | **54.25** | **49.10** | **55.57 ± 7.22** |

Seed 42 is the highest and seed 44 the lowest (or tied lowest) in every suite of both runs: overall 18.90 / 10.30 / 7.70% at 1 loop and 63.35 / 54.25 / 49.10% at 2 loops. The three seeds start every task from the same 50 initial states (identical initial-state hashes in the records). The cause of the seed spread was not investigated here.

## Result by task

Successes out of 150 episodes per task (3 seeds × 50 initial states). [per_task.csv](per_task.csv) has the per-seed counts.

### Spatial

| Task | Instruction | 1 loop | 2 loops |
| ---: | --- | ---: | ---: |
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 2 (1.3%) | 75 (50.0%) |
| 1 | pick up the black bowl next to the ramekin and place it on the plate | 2 (1.3%) | 66 (44.0%) |
| 2 | pick up the black bowl from table center and place it on the plate | 13 (8.7%) | 126 (84.0%) |
| 3 | pick up the black bowl on the cookie box and place it on the plate | 112 (74.7%) | 129 (86.0%) |
| 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 2 (1.3%) | 82 (54.7%) |
| 5 | pick up the black bowl on the ramekin and place it on the plate | 0 (0.0%) | 66 (44.0%) |
| 6 | pick up the black bowl next to the cookie box and place it on the plate | 66 (44.0%) | 143 (95.3%) |
| 7 | pick up the black bowl on the stove and place it on the plate | 15 (10.0%) | 101 (67.3%) |
| 8 | pick up the black bowl next to the plate and place it on the plate | 0 (0.0%) | 99 (66.0%) |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 69 (46.0%) | 89 (59.3%) |
| | **Spatial total** | **281/1500** | **976/1500** |

### Object

| Task | Instruction | 1 loop | 2 loops |
| ---: | --- | ---: | ---: |
| 0 | pick up the alphabet soup and place it in the basket | 65 (43.3%) | 98 (65.3%) |
| 1 | pick up the cream cheese and place it in the basket | 129 (86.0%) | 147 (98.0%) |
| 2 | pick up the salad dressing and place it in the basket | 27 (18.0%) | 145 (96.7%) |
| 3 | pick up the bbq sauce and place it in the basket | 8 (5.3%) | 107 (71.3%) |
| 4 | pick up the ketchup and place it in the basket | 18 (12.0%) | 44 (29.3%) |
| 5 | pick up the tomato sauce and place it in the basket | 14 (9.3%) | 74 (49.3%) |
| 6 | pick up the butter and place it in the basket | 32 (21.3%) | 114 (76.0%) |
| 7 | pick up the milk and place it in the basket | 28 (18.7%) | 114 (76.0%) |
| 8 | pick up the chocolate pudding and place it in the basket | 52 (34.7%) | 138 (92.0%) |
| 9 | pick up the orange juice and place it in the basket | 4 (2.7%) | 101 (67.3%) |
| | **Object total** | **377/1500** | **1082/1500** |

### Goal

| Task | Instruction | 1 loop | 2 loops |
| ---: | --- | ---: | ---: |
| 0 | open the middle drawer of the cabinet | 31 (20.7%) | 115 (76.7%) |
| 1 | put the bowl on the stove | 22 (14.7%) | 145 (96.7%) |
| 2 | put the wine bottle on top of the cabinet | 0 (0.0%) | 67 (44.7%) |
| 3 | open the top drawer and put the bowl inside | 0 (0.0%) | 65 (43.3%) |
| 4 | put the bowl on top of the cabinet | 0 (0.0%) | 58 (38.7%) |
| 5 | push the plate to the front of the stove | 0 (0.0%) | 107 (71.3%) |
| 6 | put the cream cheese in the bowl | 0 (0.0%) | 85 (56.7%) |
| 7 | turn on the stove | 15 (10.0%) | 40 (26.7%) |
| 8 | put the bowl on the plate | 8 (5.3%) | 133 (88.7%) |
| 9 | put the wine bottle on the rack | 0 (0.0%) | 4 (2.7%) |
| | **Goal total** | **76/1500** | **819/1500** |

### Long

| Task | Instruction | 1 loop | 2 loops |
| ---: | --- | ---: | ---: |
| 0 | put both the alphabet soup and the tomato sauce in the basket | 0 (0.0%) | 50 (33.3%) |
| 1 | put both the cream cheese box and the butter in the basket | 0 (0.0%) | 110 (73.3%) |
| 2 | turn on the stove and put the moka pot on it | 0 (0.0%) | 24 (16.0%) |
| 3 | put the black bowl in the bottom drawer of the cabinet and close it | 1 (0.7%) | 52 (34.7%) |
| 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 0 (0.0%) | 4 (2.7%) |
| 5 | pick up the book and place it in the back compartment of the caddy | 0 (0.0%) | 46 (30.7%) |
| 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 0 (0.0%) | 0 (0.0%) |
| 7 | put both the alphabet soup and the cream cheese box in the basket | 3 (2.0%) | 123 (82.0%) |
| 8 | put both moka pots on the stove | 0 (0.0%) | 2 (1.3%) |
| 9 | put the yellow and white mug in the microwave and close it | 0 (0.0%) | 46 (30.7%) |
| | **Long total** | **4/1500** | **457/1500** |

Tasks with no success in 150 episodes: at 1 loop 2 of 10 in Spatial, 0 of 10 in Object, 6 of 10 in Goal, 8 of 10 in Long; at 2 loops 0 in Spatial, 0 in Object, 0 in Goal, 1 in Long.

## All action-loop counts of this checkpoint in one view

The same checkpoint has been scored at six action-loop counts, in three separate records. The rows for 3, 4, 5 and 6 loops are copied from those records, not from this folder's data. All rows use the same checkpoint, video loops 4, 10 denoising steps, seeds 42/43/44 and 6,000 episodes, but they were not run as one queue: 1 and 2 loops are the original queue (cuda-graph back end); 4 loops is the earlier final campaign (cuda-graph back end); 3, 5 and 6 loops are the later matched-baseline campaign, which its README describes as run on a "local 2x RTX PRO 6000 Blackwell host (eager native inference)". The consolidated report says the original queue and the matched-baseline campaign must not be conflated; the table puts them side by side for reading only.

| Action loops | Video loop read by each action loop | Spatial | Object | Goal | Long | Overall | Successes | Seed 42 / 43 / 44 | Record |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 4 | 18.73% | 25.13% | 5.07% | 0.27% | **12.30%** | 738/6000 | 18.90 / 10.30 / 7.70 | [this folder](breakdown.json) |
| 2 | 3,4 | 65.07% | 72.13% | 54.60% | 30.47% | **55.57%** | 3334/6000 | 63.35 / 54.25 / 49.10 | [this folder](breakdown.json) |
| 3 | 2,3,4 | 95.53% | 92.60% | 92.93% | 76.80% | **89.47%** | 5368/6000 | 92.00 / 89.35 / 87.05 | [matched baselines](../../matched_baselines/README.md) |
| 4 | 1,2,3,4 | 96.93% | 98.73% | 96.13% | 87.87% | **94.92%** | 5695/6000 | 95.55 / 94.95 / 94.25 | [final campaign](../../final/results.json), trained depth |
| 5 | 1,1,2,3,4 | 94.47% | 98.87% | 96.20% | 88.87% | **94.60%** | 5676/6000 | 94.90 / 94.50 / 94.40 | [matched baselines](../../matched_baselines/README.md) |
| 6 | 1,1,1,2,3,4 | 88.13% | 96.93% | 93.73% | 78.00% | **89.20%** | 5352/6000 | 87.95 / 90.80 / 88.85 | [matched baselines](../../matched_baselines/README.md) |

[loop_sweep_all.csv](loop_sweep_all.csv) holds the same rows with the per-suite success counts.

Read from the table:

- Overall success by action loops 1 → 6: 12.30 / 55.57 / 89.47 / 94.92 / 94.60 / 89.20%.
- Long is the lowest suite at every loop count: 0.27 / 30.47 / 76.80 / 87.87 / 88.87 / 78.00%.
- Against the trained 4 loops, 5 loops differs by -0.32 points overall, 3 loops by -5.45, 6 loops by -5.72, 2 loops by -39.35 and 1 loop by -82.62.

## Source and checks

- Source: `/data14t/anhdh35/loopwam_v0_ka_sweep_20261027/results/ka1` and `ka2` on the workstation where the queue ran. Copied on 2026-11-14, read-only: the 120 + 120 task/seed group files (one record per scored episode), each run's `summary.json`, `manifest.json` and `progress.json`, and the queue's two state files. [source/ka12_groups.tar.gz](source/ka12_groups.tar.gz) (248 files, SHA-256 `d092af2a3875d98d95491761ede27a40d8012c57867b24698a755784d39bf839`) with the workstation's own `sha256sum` listing in [source/workstation_sha256.txt](source/workstation_sha256.txt); every copied file matches that listing.
- Link to the existing report: the workstation's `queue_state.json` is byte-identical to [../queue_status.json](../queue_status.json), and each run's `summary.json` equals the summary stored there.
- Checks made by [analyze_ka12.py](analyze_ka12.py) before counting: 120 group files per run with 50 episodes each (episodes and initial states 0–49), the full grid of 4 suites × 10 tasks × 3 seeds, every episode recorded with this checkpoint, action loops 1 or 2, video loops 4 and 10 denoising steps, and the episode totals equal to the run's own `summary.json` (738/6000 and 3334/6000).
- [Independent check](../../verify_v0_ka_breakdown_20261114.py) (separate code): recounts every table of this page from the archive and compares the copied rows for 3, 4, 5 and 6 loops with their records. Result in [independent_check.json](independent_check.json).
- Not stored here: the rollout videos (2.6 GB and 2.2 GB), the per-query logs and the per-episode copies of the same records. They remain on the workstation under the path above.

## Files

| File | Content |
| --- | --- |
| [breakdown.json](breakdown.json) | Both runs: totals, suites, seeds, tasks, checks |
| [suite_results.csv](suite_results.csv) | One row per run and suite, with the three seeds |
| [per_task.csv](per_task.csv) | One row per run, suite and task, with per-seed success counts |
| [episodes_ka1.csv](episodes_ka1.csv) · [episodes_ka2.csv](episodes_ka2.csv) | One row per scored episode (6,000 each): suite, task, seed, initial state, success, policy steps, queries |
| [loop_sweep_all.csv](loop_sweep_all.csv) | The six action-loop counts side by side, each row with its record |
| [analyze_ka12.py](analyze_ka12.py) · [make_report.py](make_report.py) | The scripts that produced the files above and this page |
| [source/](source/ka12_groups.tar.gz) | The copied run records and the workstation's hash listing |
