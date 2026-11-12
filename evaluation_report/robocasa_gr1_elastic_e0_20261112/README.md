# RoboCasa-GR1 elastic student GR1-E0: final results

**Status: complete.** GR1-E0 trained for 5 epochs (235,160 updates) from 2026-11-08 11:50 UTC to 2026-11-11 02:59 UTC and was evaluated 25 times with the official protocol; the last evaluation finished 2026-11-11 23:26 UTC. This is the first report with GR1-E0 results. The teacher's report, [robocasa_gr1_continuation_20261108](../robocasa_gr1_continuation_20261108/README.md), says GR1-E0 was in progress; it is kept unchanged.

GR1-E0 is one student network distilled from the RoboCasa-GR1 v0 teacher (continuation epoch 9, 626/1200 = 52.17%). The teacher always runs 10 denoising steps with 4 loops of its 6 shared core blocks. The student also runs with fewer loops and fewer steps, chosen at inference time without retraining. A setting is written as the loop count of each step: `[4]x10` is 10 steps of 4 loops (the teacher's setting), `[2,2]` is 2 steps of 2 loops, `[1,1]` is 2 steps of 1 loop. Cost is counted in action-block passes per policy query, steps x (6 + 6 x loops): 300 for `[4]x10`, 60 for `[4,4]`, 36 for `[2,2]`, 24 for `[1,1]`.

## Results

Official success: successes out of 1,200 episodes (24 tasks x 50). Teacher, same protocol and host: **626 (52.17%)**.

| setting | loops | steps | block passes | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 |
|---|---|---|---|---|---|---|---|---|
| `[1,1]` | 1 | 2 | 24 | 474 (39.50%) * | 533 (44.42%) | 505 (42.08%) | 518 (43.17%) | 533 (44.42%) |
| `[1]x5` | 1 | 5 | 60 | not run | not run | not run | 569 (47.42%) | 558 (46.50%) * |
| `[1]x10` | 1 | 10 | 120 | not run | not run | not run | 561 (46.75%) | 593 (49.42%) |
| `[2,2]` | 2 | 2 | 36 | not run | 627 (52.25%) | 608 (50.67%) | 619 (51.58%) | 598 (49.83%) |
| `[2]x5` | 2 | 5 | 90 | not run | not run | not run | 609 (50.75%) | 618 (51.50%) |
| `[2]x10` | 2 | 10 | 180 | not run | not run | not run | 609 (50.75%) | 607 (50.58%) |
| `[4,4]` | 4 | 2 | 60 | not run | 577 (48.08%) | 586 (48.83%) | 629 (52.42%) | 621 (51.75%) |
| `[4]x5` | 4 | 5 | 150 | not run | not run | not run | 642 (53.50%) | 653 (54.42%) |
| `[4]x10` | 4 | 10 | 300 | not run | not run | not run | 620 (51.67%) | 628 (52.33%) |

\* The validator flagged one episode of this evaluation; the count is the official one (see [Validator flags](#validator-flags)).

One run of 1,200 episodes has a 95% interval of about +/-2.8 points. The comparisons below pair episodes by task and episode index (every run uses the same environment seed and the same policy noise seed); `net` is the difference in successful episodes and `z` the McNemar statistic (|z| >= 1.96 is p < 0.05, not corrected for the number of comparisons). Each checkpoint was evaluated once per setting, so run-to-run variation is not measured. All tables: [analysis/results.md](analysis/results.md); per task: [analysis/per_task.md](analysis/per_task.md), [analysis/per_task_successes.csv](analysis/per_task_successes.csv).

### Each setting against the teacher (epochs 4 and 5)

| setting | block passes | epoch 4 | net vs teacher (z) | epoch 5 | net vs teacher (z) | epoch 4 -> 5 (z) |
|---|---|---|---|---|---|---|
| `[1,1]` | 24 | 518 (43.17%) | -108 (z -4.69) | 533 (44.42%) | -93 (z -4.17) | +15 (z +0.68) |
| `[1]x5` | 60 | 569 (47.42%) | -57 (z -2.54) | 558 (46.50%) | -68 (z -3.03) | -11 (z -0.50) |
| `[1]x10` | 120 | 561 (46.75%) | -65 (z -2.83) | 593 (49.42%) | -33 (z -1.45) | +32 (z +1.41) |
| `[2,2]` | 36 | 619 (51.58%) | -7 (z -0.31) | 598 (49.83%) | -28 (z -1.25) | -21 (z -0.94) |
| `[2]x5` | 90 | 609 (50.75%) | -17 (z -0.75) | 618 (51.50%) | -8 (z -0.36) | +9 (z +0.40) |
| `[2]x10` | 180 | 609 (50.75%) | -17 (z -0.75) | 607 (50.58%) | -19 (z -0.88) | -2 (z -0.09) |
| `[4,4]` | 60 | 629 (52.42%) | +3 (z +0.13) | 621 (51.75%) | -5 (z -0.23) | -8 (z -0.36) |
| `[4]x5` | 150 | 642 (53.50%) | +16 (z +0.74) | 653 (54.42%) | +27 (z +1.19) | +11 (z +0.50) |
| `[4]x10` | 300 | 620 (51.67%) | -6 (z -0.27) | 628 (52.33%) | +2 (z +0.09) | +8 (z +0.35) |

### Loop count, pooled over the three step counts and over epochs 4 and 5

| loops | successes of 7,200 | paired change from the row above |
|---|---|---|
| 1 | 3332 (46.28%) | |
| 2 | 3660 (50.83%) | +328 (z +5.95) |
| 4 | 3793 (52.68%) | +133 (z +2.46) |

### What the results show

- **4 loops match the teacher at every step count.** Against the teacher's 626: epoch 5 `[4,4]` 621, `[4]x5` 653, `[4]x10` 628; epoch 4 629, 642, 620. The paired z of these six runs lies between -0.27 and +1.19. `[4,4]` uses 60 block passes per query, a fifth of the teacher's 300.
- **2 loops are about 2 points below 4 loops.** No single 2-loop run differs significantly from the teacher (z between -1.25 and -0.31 at epochs 4 and 5), but pooled as in the table above, 2 loops are 1.85 points below 4 loops (z +2.46).
- **1 loop is clearly below the teacher.** `[1,1]` reaches 518 and 533 at epochs 4 and 5 (against the teacher: net -108, z -4.69 and net -93, z -4.17). More steps recover part of the gap: `[1]x10` reaches 593 at epoch 5 (net -33, z -1.45) and 561 at epoch 4 (net -65, z -2.83). The `[1,1]` deficit is spread over the tasks: its mean count over epochs 2-5 is below the teacher's on 21 of 24 tasks.
- **The step count matters only at 1 loop.** At 2 and 4 loops no paired change between 2, 5 and 10 steps is significant in either epoch (largest |z| 1.46). At 1 loop: epoch 4 `[1,1]` -> `[1]x5` net +51, z +2.27; epoch 5 `[1,1]` -> `[1]x10` net +60, z +2.72.
- **Epochs 4 and 5 do not differ.** Over all nine settings they total 5,376 and 5,409 of 10,800 (paired net +33, z +0.49); no single setting changes significantly (largest |z| 1.41). Epoch 5 is the final checkpoint and the one published on Hugging Face.
- **Earlier epochs.** `[1,1]` did not improve after epoch 2 (533, 505, 518, 533 for epochs 2-5; epoch 1 474). `[2,2]` shows no significant change over epochs 2-5 (627, 608, 619, 598). `[4,4]` is lower at epochs 2 and 3 (577, 586) than at epochs 4 and 5 (629, 621): 48.46% against 52.08% of 2,400 episodes each (z +2.51).

Related reports: the teacher's training and evaluations are in [robocasa_gr1_v0_20261101](../robocasa_gr1_v0_20261101/README.md) and [robocasa_gr1_continuation_20261108](../robocasa_gr1_continuation_20261108/README.md); its re-evaluation on other hosts is in [robocasa_gr1_epochs09_10_20261108](../robocasa_gr1_epochs09_10_20261108/README.md).

## Evaluation protocol

The same protocol as the teacher's evaluations: starVLA `examples/simBenchmarks/Robocasa_tabletop`, 24 tasks x 50 rollouts, any-step success, 720 steps, 12 actions executed per query, n_envs 1, environment seed 0, policy noise seed 42, EGL rendering; H100 GPUs 4-7 (job 5758), 2 policy servers per GPU, 24 parallel clients. In every task file the policy server reports the checkpoint it loaded (path, step, sha256), the schedule and its loop list; all evaluations of one epoch served the same file. Every evaluation completed at its first attempt.

| when (UTC) | evaluations | why |
|---|---|---|
| 2026-11-08 to 2026-11-10 | `[1,1]` after each of epochs 1-4, with training paused | planned per-epoch check |
| 2026-11-10 12:50-16:37 | epoch 4 at `[4,4]`, `[4]x5`, `[2,2]` | diagnostic, with epoch 5 stopped on request: is the `[1,1]` gap due to loops or to steps? |
| 2026-11-11 03:01-13:15 | epoch 5 at all nine settings | final checkpoint |
| 2026-11-11 13:19-19:12 | epoch 4 at the five remaining settings | full epoch 4 / epoch 5 comparison |
| 2026-11-11 19:14-23:26 | epochs 2 and 3 at `[2,2]` and `[4,4]` | epoch trend at 2 and 4 loops |

The mean policy round trip and wall time of each evaluation are in section E of [analysis/results.md](analysis/results.md). They were measured under the evaluation's own load (24 simulations on 8 servers) and are not a latency measurement; single-query latency of the settings has not been measured on RoboCasa-GR1.

## Training

- **Teacher and initialization.** RoboCasa-GR1 v0 continuation epoch 9 (update 211,644, sha256 `694a679b4c07`), chosen by the rule fixed before the run: "epoch 10 if its official success count (24 tasks x 50 episodes) is strictly greater than epoch 9's 626/1200, else epoch 9"; epoch 10 had 620/1200, so epoch 9 was used ([record/teacher_selection.json](record/teacher_selection.json)). The student starts from the teacher's weights.
- **What is trained.** The action expert and the step-length embedding (105,701,021 parameters); the rest of the model stays frozen at the teacher's weights. The recipe is E0 of the LIBERO elastic students (`elastic-v1`) with two embodiment settings changed: the endpoint loss covers the 12 actions executed per query, and the gripper term is off (GR1's 29 action dimensions are continuous joint targets).
- **Loss.** `total = local + 0.5 x end + 0.5 x anchor`: `local` matches the student's velocity to the teacher's at every step of the sampled setting, `end` is the squared error between the student's and the teacher's final actions, `anchor` is the data flow-matching loss at 4 loops. The nine settings (loops 1, 2, 4 x steps 2, 5, 10) are sampled evenly, one setting per update.
- **Optimization.** AdamW, learning rate 3e-5 for the action expert and 1e-3 for the step-length embedding, linear warmup over 7,055 updates, cosine decay to 1%, gradient clipping at 1.0; global batch 128 (4 H100 x 32); 6,020,058 windows per epoch = 47,032 updates per epoch; seed 42. Data, normalization and latent cache are the teacher's.
- **Timeline.** Epochs 1-4 each trained in one launch and were followed by a `[1,1]` evaluation. Epoch 5 started 2026-11-10 12:02 UTC and was stopped on request at update 191,359 (12:49 UTC) to run the diagnostic evaluations of epoch 4; it resumed from the saved training state at 16:49 UTC (the sixth launch) and finished at update 235,160 on 2026-11-11 02:59 UTC. `training/metrics.jsonl.gz` holds every update exactly once, in order; time inside updates 53.77 h (10.71, 10.69, 10.97, 10.70, 10.70 h for epochs 1-5).
- **Checkpoints.** Each epoch export passed the export verification before it was evaluated (`record/validation/verify_GR1-E0_e01.json` ... `e05.json`). Hashes: [training/checkpoints_sha256.json](training/checkpoints_sha256.json).
- **Training loss.** The endpoint loss fell from epoch 1 to epoch 5 for every setting and is ordered by loop count, not by step count: at the end of epoch 5 it is 0.00201 for `[1,1]`, 0.00115 for `[2,2]` and 0.00052 for `[4,4]` ([analysis/training_curves.md](analysis/training_curves.md)). A lower loss did not bring more success at 1 loop: the `[1,1]` loss fell 25% from epoch 2 to epoch 5 (0.00267 to 0.00201) while its success count was 533 both times.

## Validator flags

Two evaluations are marked `valid: false` by the run's validator, each for one episode: epoch 1 `[1,1]` (task PnPMilkToMicrowaveClose, episode index 47) and epoch 5 `[1]x5` (task PosttrainPnPNovelFromPlateToCardboardboxSplitA, episode index 46). The official evaluator records both episodes as failures, while the name of their rollout video carries the success flag of the last rendered frame ([analysis/video_flag_check_output.md](analysis/video_flag_check_output.md)). The validator assumed that such a video always belongs to a successful episode. That does not hold: the evaluator reads the success state after each executed chunk of 12 actions, and the video flag is the state at the last rendered frame, one step before the end of the episode. An episode that is in the success state at that frame but at none of the chunk ends is an official failure with a success-flagged video.

The counts in this report are the official evaluator's: 474/1200 and 558/1200. Counting the two episodes as successes instead would give 475 and 559 (+0.08 points each). Every other check of the validator passed for both evaluations, and the other 23 evaluations are valid. The validator output is stored unchanged (`eval/e01_flat_k1_n2_validation.json`, `eval/e05_flat_k1_n5_validation.json`); this is also why `record/STATUS.json` ends as `done_with_problems`.

## Publication and code

- **Hugging Face `anhdao69/LoopWAM_NT`:** `Robocasa-GR1/Elastic/E0/epoch_05.pt` (sha256 `942959275057`, 2,340,640,237 bytes, update 235,160) was uploaded on 2026-11-11 (commit `c94c25e3`) with its training config, metrics and export verification; the remote hash and size of every file were checked against the local bytes ([hf/](hf/)). Only the epoch-5 `[1,1]` result existed at upload time, so the model card there lists that result alone; the other 24 evaluations in this report have not been added to Hugging Face. Epochs 1-4 are not published.
- **Code:** revision `7c5a5db` of branch `elastic_gr1` (pinned in `record/REVISION` and in `training/config.json`). The branch is not on GitHub at the time of this report; the shell tools that drove the run are in `record/tools/`.

**Independent check** ([independent_check.json](independent_check.json), script [../verify_gr1_elastic_e0_20261112.py](../verify_gr1_elastic_e0_20261112.py)): for each of the 25 evaluations, per-episode successes == summary.json == validation JSON == the run's RESULTS.md == the orchestrator log == the tables above; the evaluation config and every policy server name the right checkpoint, step, hash and loop list; the paired statistics and pooled counts are recomputed separately from `analysis/analyze_gr1_e0.py` and match `analysis/analysis.json` and this README; training metrics cover every update once; the teacher selection, checkpoint hashes and the Hugging Face receipt agree.

## Contents

- `analysis/`: `analyze_gr1_e0.py` (builds every table from the stored evidence), `results.md`, `per_task.md`, `per_task_successes.csv`, `training_curves.md`, `analysis.json`, and the validator-flag check (`video_flag_check.py`, `video_flag_check_output.md`).
- `eval/`: 25 evaluation folders `eEE_flat_kK_nN` (epoch, loops, steps): per-task episode JSON, summary, config, driver, client and server logs; one `_validation.json` each; `media_manifest.tsv.gz` lists the 30,004 rollout videos (166.4 GB) kept on the H100.
- `training/`: config, final status, launches, CPU binding, data manifest and normalization, the schedule sequence, `metrics.jsonl.gz` (one row per update), `train.log.gz`, `checkpoints_sha256.json`.
- `record/`: the run's own RESULTS.md and STATUS files, orchestrator and keeper logs, teacher selection, initialization record, pre-launch validation, self tests, export verifications, tools, and logs (training, evaluation and validation logs of every step).
- `hf/`: Hugging Face upload receipt, dry-run and upload logs, model-card drafts, and the upload script.
- `SOURCES.tsv`: every stored file with its source path and the SHA-256 and size of the original bytes (also for the `.gz` files). `excluded_files.tsv`: the files of the run folder that are not stored (checkpoints and lock files), with sizes and checkpoint hashes.

Copied 2026-11-12 06:12 UTC from the H100 without modifying any source; the server-side SHA-256 listing was identical before and after the archive was made, and every local file was checked against it. Not copied: model checkpoints (`*.pt`) and rollout videos.
