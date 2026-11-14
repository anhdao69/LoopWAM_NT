# All recorded evaluation results

Original broad snapshot: **2026-11-07T02:51:27+07:00** (Vietnam time).

**November 15 targeted update:** [RoboTwin-10 epoch-10 task details](robotwin10_epoch10_results_20261115/README.md) and the new section below add the requested completed results. Earlier dated scope notes describe their original updates; other benchmark sections retain their existing contents.

**RT-50 Elastic addendum, November 13, 2026:** the [detailed final E0 teacher10/student5ep comparison](robotwin_elastic_e0_teacher10_5ep_20261111/final_report.md) now records **[1,1] 89.28%, [2,2] 90.60%, [4,4] 91.49%**. All use NFE 2, seed 42 and 10,000 episodes per schedule, and all final audits passed. The linked report contains all 50 tasks, runtimes, exact checkpoint identity, original-host record checks and the historical epoch-1 E0 comparison. The remainder of this broad catalogue retains its November 7 scope and is not a new audit of other benchmarks.

This report reconciles the saved evaluation records in this checkout. It includes the latest H100 RoboTwin-10 retrains, all retained RoboTwin reruns, full RoboTwin epoch comparisons, matched LIBERO, CL/RT, Plus/Pro and action-depth results. No evaluations were rerun during this update. Original logs, immutable scored artifacts and historical reports are retained.

RoboTwin score arithmetic was recomputed from accepted episode records and checked against final tables. Where a final audit exists, its COMPLETE.json digest was checked. Existing video audits are cited; this update did not rehash every historical video or re-run historical simulations. Epoch6 has complete records but lacks a saved final_audit.json. Historical LIBERO/Plus/Pro counts were cross-checked at their saved aggregation levels, not freshly re-audited at video level.

No pooled total is calculated across sections: historical campaign and branch catalogues overlap. Different benchmarks, training task sets, seeds, reruns and hosts remain separate. Unseeded RoboTwin instruction-template sampling prevents asserting exact trajectory pairing.

## RoboTwin-10 — selected epoch-10 results, updated November 15

This requested update covers five model versions and six completed runs. Each uses seed 42, NFE 10 and 2,000 accepted episodes. The newer Dense-12 continuation and earlier local run remain separate. All six saved final audits and their episode totals were reconciled from 12,000 records.

| Model / evaluation | Clean | Randomized | All successes | Overall SR | Execution |
| --- | --- | --- | --- | --- | --- |
| [v0 4/2](robotwin10_epoch10_queue_20261113/v0_4x2/final_report.md) | 825/1000 (82.50%) | 823/1000 (82.30%) | 1648/2000 | **82.40%** | Local RTX PRO 6000 |
| [v0 4/1](robotwin10_v0_4x1_epoch10_20261108/final_report.md) | 800/1000 (80.00%) | 819/1000 (81.90%) | 1619/2000 | **80.95%** | Local RTX PRO 6000 |
| [v0 2/2](robotwin10_epoch10_queue_20261113/v0_2x2/final_report.md) | 786/1000 (78.60%) | 792/1000 (79.20%) | 1578/2000 | **78.90%** | Local RTX PRO 6000 |
| [v0 1/4](robotwin10_epoch10_remote189_20261113/v0_1x4/final_report.md) | 802/1000 (80.20%) | 789/1000 (78.90%) | 1591/2000 | **79.55%** | aLinh RTX PRO 5000 |
| [Dense-12 — latest continuation](robotwin10_epoch10_remote189_20261113/dense_s12/final_report.md) | 769/1000 (76.90%) | 776/1000 (77.60%) | 1545/2000 | **77.25%** | 400 local + 1,600 aLinh episodes |
| [Dense-12 — earlier local run](robotwin10_dense_s12_epoch10_20261109/final_report.md) | 754/1000 (75.40%) | 763/1000 (76.30%) | 1517/2000 | **75.85%** | Local RTX PRO 6000 |

The following per-task table uses the latest completed run for each included model. Each task percentage covers 100 clean + 100 randomized episodes. Detailed clean/randomized success counts for **every task of all six evaluations**, exclusion accounting and checkpoint identities are in the [dated detailed report](robotwin10_epoch10_results_20261115/README.md).

| Task | v0 4/2 | v0 4/1 | v0 2/2 | v0 1/4 | Dense-12 — latest continuation |
| --- | --- | --- | --- | --- | --- |
| adjust_bottle | 100.00% | 99.00% | 99.50% | 99.50% | 99.50% |
| grab_roller | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| hanging_mug | 51.00% | 45.00% | 40.00% | 45.50% | 44.50% |
| move_stapler_pad | 69.50% | 75.50% | 63.50% | 63.50% | 61.00% |
| open_microwave | 77.50% | 70.50% | 58.00% | 73.50% | 53.00% |
| press_stapler | 85.00% | 84.50% | 91.00% | 88.50% | 90.50% |
| scan_object | 85.50% | 83.00% | 87.00% | 83.00% | 79.50% |
| stack_blocks_two | 97.00% | 99.00% | 98.00% | 97.50% | 98.00% |
| stamp_seal | 91.50% | 89.00% | 90.00% | 83.00% | 87.00% |
| turn_switch | 67.00% | 64.00% | 62.00% | 61.50% | 59.50% |
| **All 10 tasks** | **82.40%** | **80.95%** | **78.90%** | **79.55%** | **77.25%** |

Exports for this update: [overall CSV](robotwin10_epoch10_results_20261115/summary.csv), [60-row task CSV](robotwin10_epoch10_results_20261115/task_results.csv), [full JSON](robotwin10_epoch10_results_20261115/results.json). Earlier CSV/JSON snapshots under `all_results/` retain their original scope. The latest Dense-12 run includes 400 verified local episodes and 1,600 aLinh episodes. Reruns are not pooled or selected task by task; unseeded instruction-template RNG prevents claiming exact rollout pairing.

## RoboTwin-10 — historical epoch-5 full ten-task runs, seed 42

Each row is 10 tasks × clean/randomized ×100 accepted episodes =2,000 episodes, NFE 10. All 12 runs have saved passing final audits. Reruns are separate observations; no best-seed or best-task selection. H100 means training hardware, not evaluation hardware.

| Model / execution | Clean | Randomized | Combined | Successes | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Dense-S30 — local | 83.80% | 83.40% | **83.60%** | 1672/2000 | [report](robotwin10_dense_s30/final_report.md) |
| v0 4/1 — server189 | 82.30% | 80.40% | **81.35%** | 1627/2000 | [report](robotwin10_v0_4x1/final_report.md) |
| v0 4/1 — local rerun | 82.10% | 79.60% | **80.85%** | 1617/2000 | [report](robotwin10_v0_4x1_local_20261105/final_report.md) |
| v0 4/2 — H100 retrain, server189 evaluation | 80.10% | 81.50% | **80.80%** | 1616/2000 | [report](robotwin10_h100_v0_4x2_remote189_20261106/final_report.md) |
| v0 4/2 — original local | 80.40% | 79.50% | **79.95%** | 1599/2000 | [report](robotwin10_v0_4x2/final_report.md) |
| v0 4/4 — original local | 77.40% | 80.30% | **78.85%** | 1577/2000 | [report](robotwin10_v0_4x4/final_report.md) |
| v0 4/4 — H100 retrain, local evaluation | 79.90% | 77.40% | **78.65%** | 1573/2000 | [report](robotwin10_h100_v0_4x4_20261106/final_report.md) |
| v0 4/4 — rerun server189 | 78.40% | 78.00% | **78.20%** | 1564/2000 | [report](robotwin10_v0_4x4_remote189/final_report.md) |
| v0 4/4 — rerun server137 | 76.60% | 79.30% | **77.95%** | 1559/2000 | [report](robotwin10_v0_4x4_remote137_20261105/final_report.md) |
| v0 2/2 — local | 74.50% | 72.10% | **73.30%** | 1466/2000 | [report](robotwin10_v0_2x2/final_report.md) |
| v0 1/4 — server137, valid four-worker rerun | 73.90% | 72.40% | **73.15%** | 1463/2000 | [report](robotwin10_v0_1x4_remote137_20261105/final_report.md) |
| Dense-S12 — local, valid four-worker rerun | 71.90% | 72.50% | **72.20%** | 1444/2000 | [report](robotwin10_dense_s12_20261105/final_report.md) |

### H100 retrains versus the original local variants

| Variant | Original | H100 retrain | Difference | Success-count change |
| --- | ---: | ---: | ---: | ---: |
| v0 4/4 | 78.85% | 78.65% | -0.20 pp | -4 |
| v0 4/2 | 79.95% | 80.80% | +0.85 pp | +17 |

The retrains declare matching model/trainer source hashes and normalization values. Their donor-artifact and data-manifest hashes differ from the earlier releases; see their setup reports. Evaluation hosts differ. The differences above are observed single-seed outcomes, not a significance claim.

## Full RoboTwin — all 50 tasks, seed 42

Each epoch row uses 10,000 episodes: 100 per task per setting; NFE 10. These are the full 50-task-trained checkpoint family, separate from RoboTwin-10 training.

| Epoch | Clean | Randomized | Combined | Successes | Evidence status |
| --- | ---: | ---: | ---: | ---: | --- |
| 5 | 91.60% | 91.86% | **91.73%** | 9173/10000 | [Saved final audit passed](robotwin_v0/report.md) |
| 6 | 90.08% | 89.62% | **89.85%** | 8985/10000 | [Complete records reconciled; no saved final artifact audit](robotwin_v0_epoch06/report.md) |
| 10 | 92.06% | 91.68% | **91.87%** | 9187/10000 | [Saved final audit passed](robotwin_v0_epoch10/final_report.md) |

Epoch10 versus epoch5: +0.46pp clean, −0.18pp randomized, **+0.14pp combined (+14 successes)**. Epoch10 and the recent ten-task reruns include an explicit renderer-log guard; older historical audits did not all record that same check.

## Seed43 two-task experiments — separate from the main table

Only move_stapler_pad and open_microwave;100 episodes each, clean/randomized;400 episodes/model. Never substituted into seed 42 results.

| Model | Clean | Randomized | Combined | Successes |
| --- | ---: | ---: | ---: | ---: |
| [Dense-S30 — seed 43, two tasks only](robotwin10_seed43_two_tasks/dense_s30/final_report.md) | 73.50% | 63.50% | 68.50% | 274/400 |
| [v0 4/4 — seed 43, two tasks only](robotwin10_seed43_two_tasks/v0_4x4/final_report.md) | 46.00% | 62.50% | 54.25% | 217/400 |

## Matched standard LIBERO and Long-only results

Source: [matched-baselines records](matched_baselines/README.md). Four-suite entries use 6,000 episodes, seeds42/43/44,50 episodes/task/seed. Long-only entries use 1,500 episodes and are explicitly marked by their denominator. The first 14 entries are the later matched campaign; the remaining entries retain the earlier campaign provenance.

| Model | Spatial | Object | Goal | Long | Overall / evaluated scope | Successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FasterWAM | 99.00% | 99.60% | 98.20% | 98.07% | 98.72% | 5923/6000 |
| Light-WAM | 96.67% | 98.87% | 96.60% | 90.93% | 95.77% | 5746/6000 |
| LoopWAM Dense-S12 | 94.80% | 93.27% | 92.13% | 79.73% | 89.98% | 5399/6000 |
| LoopWAM Dense-S30 | 97.27% | 98.60% | 95.87% | 90.80% | 95.63% | 5738/6000 |
| LoopWAM v0 Ka3 | 95.53% | 92.60% | 92.93% | 76.80% | 89.47% | 5368/6000 |
| LoopWAM v0 Ka5 | 94.47% | 98.87% | 96.20% | 88.87% | 94.60% | 5676/6000 |
| LoopWAM v0 Ka6 | 88.13% | 96.93% | 93.73% | 78.00% | 89.20% | 5352/6000 |
| LoopWAM v0 partition 0/6/6 | 97.20% | 98.73% | 95.27% | 86.47% | 94.42% | 5665/6000 |
| LoopWAM v0 partition 1/6/5 | 97.27% | 98.13% | 95.93% | 83.33% | 93.67% | 5620/6000 |
| LoopWAM v0 init core16_21 | 96.20% | 97.27% | 94.60% | 84.40% | 93.12% | 5587/6000 |
| LoopWAM v0 partition 5/6/1 | 96.73% | 97.40% | 93.60% | 84.93% | 93.17% | 5590/6000 |
| LoopWAM v0 partition 6/6/0 | 95.07% | 92.67% | 91.27% | 82.07% | 90.27% | 5416/6000 |
| LoopWAM v0 partition 1/9/2 k3 | 97.53% | 98.33% | 93.47% | 77.80% | 91.78% | 5507/6000 |
| LoopWAM v0 partition 4/3/5 k7 | 93.33% | 97.87% | 92.47% | 79.33% | 90.75% | 5445/6000 |
| Light-WAM Long | — | — | — | 90.87% | 90.87% | 1363/1500 |
| LoopWAM full 4/4 | 96.93% | 98.73% | 96.13% | 87.87% | 94.92% | 5695/6000 |
| LoopWAM Long 4/4 original | — | — | — | 77.73% | 77.73% | 1166/1500 |
| LoopWAM Long 4/4 repeat | — | — | — | 79.47% | 79.47% | 1192/1500 |
| LoopWAM Long KV mix 4/1 | — | — | — | 75.53% | 75.53% | 1133/1500 |
| LoopWAM Long aligned 4/1 | — | — | — | 74.00% | 74.00% | 1110/1500 |
| LoopWAM Long KV concat 4/1 | — | — | — | 74.53% | 74.53% | 1118/1500 |
| Fast-WAM original all suites | 97.47% | 99.27% | 96.27% | 93.53% | 96.63% | 5798/6000 |
| LoopWAM full video1/action4 | 96.00% | 97.80% | 94.73% | 78.20% | 91.68% | 5501/6000 |
| LoopWAM full video4/action1 | 96.27% | 98.07% | 95.40% | 86.27% | 94.00% | 5640/6000 |
| LoopWAM full video2/action2 | 96.93% | 96.53% | 94.27% | 82.07% | 92.45% | 5547/6000 |
| LoopWAM full KV mix 4/1 | 97.40% | 98.33% | 96.93% | 85.13% | 94.45% | 5667/6000 |

## ChronoLoop and Residual Transport — historical campaign

Seeds42/43/44;6,000 episodes per completed condition. `external_complete` retains its source provenance and does not mean a new local execution. Counts agree between detailed JSON and the existing run-summary CSV. [Full campaign evidence](campaign/detailed_report.html).

| Condition | Source status | Successes | SR |
| --- | --- | ---: | ---: |
| CL-0 | complete | 5766/6000 | 96.10% |
| RT-A S1 | complete | 5668/6000 | 94.47% |
| RT-A S2 | complete | 5647/6000 | 94.12% |
| RT-A S3 | complete | 5643/6000 | 94.05% |
| RT-A S4 | external_complete | 5632/6000 | 93.87% |
| RT-B4 S1 | complete | 5659/6000 | 94.32% |
| RT-B4 S2 | complete | 5566/6000 | 92.77% |
| CL-REG | complete | 5758/6000 | 95.97% |
| RT-T1ft S1 | external_complete | 5674/6000 | 94.57% |
| RT-T1ft S2 | external_complete | 5651/6000 | 94.18% |
| RT-Pa S1 | external_complete | 5637/6000 | 93.95% |
| RT-Pa S2 | complete | 5575/6000 | 92.92% |
| CL-W2 | complete | 5761/6000 | 96.02% |
| RT-A2 S1 | external_complete | 5646/6000 | 94.10% |
| RT-A2 S2 | external_complete | 5593/6000 | 93.22% |
| RT-A2 S5 | external_complete | 5699/6000 | 94.98% |
| RT-A2 S6 | external_complete | 5669/6000 | 94.48% |
| RT-A2 S7 | external_complete | 5696/6000 | 94.93% |
| RT-A2 S8 | external_complete | 5672/6000 | 94.53% |
| CL-A | complete | 5771/6000 | 96.18% |
| CL-0@1 | complete | 5742/6000 | 95.70% |
| RT-TF S1 | external_complete | 5665/6000 | 94.42% |
| RT-TF S2 | external_complete | 5648/6000 | 94.13% |
| RT-B2a parent2 | complete | 5674/6000 | 94.57% |
| CL-A@1 | complete | 5720/6000 | 95.33% |
| CL-FRAME | complete | 5754/6000 | 95.90% |

## LIBERO-Plus and LIBERO-Pro

Seven CL checkpoints, seed 42 only. Plus follows the released FasterWAM 15% selection, one episode per selected task,1,506 episodes/model. Pro uses Position and Task separately, four suites,10 episodes/task: 400 episodes/perturbation, 800/model. All 14 jobs / 16,142 episodes completed in the saved campaign. Low Pro Position SR is retained as measured; completion alone does not establish its cause. [Detailed protocol and results](robustness/report.md).

| Model | Plus /1506 | Pro Position /400 | Pro Task /400 | Pro combined /800 |
| --- | ---: | ---: | ---: | ---: |
| CL-0 | 708/1506 (47.01%) | 3/400 (0.75%) | 55/400 (13.75%) | 58/800 (7.25%) |
| CL-A | 719/1506 (47.74%) | 3/400 (0.75%) | 57/400 (14.25%) | 60/800 (7.50%) |
| CL-REG | 721/1506 (47.88%) | 2/400 (0.50%) | 58/400 (14.50%) | 60/800 (7.50%) |
| CL-W2 | 729/1506 (48.41%) | 1/400 (0.25%) | 52/400 (13.00%) | 53/800 (6.62%) |
| CL-0@1 | 689/1506 (45.75%) | 2/400 (0.50%) | 54/400 (13.50%) | 56/800 (7.00%) |
| CL-A@1 | 693/1506 (46.02%) | 4/400 (1.00%) | 56/400 (14.00%) | 60/800 (7.50%) |
| CL-FRAME | 719/1506 (47.74%) | 3/400 (0.75%) | 59/400 (14.75%) | 62/800 (7.75%) |

## Original local v0 action-depth queue

Kv4, NFE 10, seeds42/43/44,6,000 episodes per condition. Ka1/Ka2 completed;Ka3/5/6 were removed from this queue at user request. Separate later matched-baseline Ka3/5/6 results above remain valid records of that campaign. These two histories must not be conflated. [Original queue](v0_ka_sweep/report.md).

| Ka | Successes | SR |
| --- | ---: | ---: |
| 1 | 738/6000 | 12.30% |
| 2 | 3334/6000 | 55.57% |

## Exclusions and missing-result history

- Dense-S12 and v0 1/4 ten-worker RoboTwin attempts were invalidated for GPU/renderer errors. Only their fresh four-worker audited reruns appear above; archived attempts are not additional model results.
- The waiting duplicate RoboTwin4/4 queue on server137 was cancelled; the separate later completed rerun is listed by its dated run identity.
- The original `dense_full` waiting queue was cancelled without scored episodes. Separate matched-baseline Dense-S12/S30 results are listed above; RoboTwin dense models are another benchmark.
- The historical CL/RT discovery snapshot has 29 `not_published` conditions. This is its saved status, not a fresh assertion about Hugging Face availability today. They have no numeric result in that snapshot and are retained in [the unscored inventory](all_results/unscored_inventory.json).
- No seed 43 best-task substitutions, smoke/calibration results, partial failed attempts or archived invalid runs are included in final SR.

## Branch-reported matched result catalogue

The branch snapshot also records Elastic/RT/Chrono/init-variant results beyond the local campaign. The following catalogue preserves the original source labels from the October 30 matched report. It overlaps the tables above; it is not another set of independently rerun models and must not be summed with them. [Branch evidence index](branch_reports/index.json).

| # | Model | Success % | Evaluated |
|---:|---|---:|---|
| 1 | **FasterWAM** | **98.72** | this evaluation (local RTX PRO 6000 Blackwell) |
| 2 | FastWAM (released, libero_uncond_2cam224) | 96.63 | H100, branch `elastic` all-models report (baseline) |
| 3 | ChronoLoop CL-A_mem16-learned-loopwrite_a4 | 96.18 | H100, branch `elastic` all-models report (chronoloop) |
| 4 | ChronoLoop CL-0_no-memory_a4 | 96.10 | H100, branch `elastic` all-models report (chronoloop) |
| 5 | ChronoLoop CL-W2_mem16-learned-external-updater_a4 | 96.02 | H100, branch `elastic` all-models report (chronoloop) |
| 6 | ChronoLoop CL-REG_mem16-reset-registers_a4 | 95.97 | H100, branch `elastic` all-models report (chronoloop) |
| 7 | ChronoLoop CL-FRAME_history-frame-k3_a4 | 95.90 | H100, branch `elastic` all-models report (chronoloop) |
| 8 | elastic F0-PB parent2 [4,4] | 95.88 | H100, branch `elastic` all-models report (elastic) |
| 9 | **Light-WAM** | **95.77** | this evaluation (local RTX PRO 6000 Blackwell) |
| 10 | elastic F0-PB parent5 [4]x5 | 95.75 | H100, branch `elastic` all-models report (elastic) |
| 11 | ChronoLoop CL-0-at1_no-memory_a1 | 95.70 | H100, branch `elastic` all-models report (chronoloop) |
| 12 | elastic E1 flat_k4_n5 [4]x5 | 95.65 | H100, branch `elastic` all-models report (elastic) |
| 13 | **LoopWAM Dense-S30** | **95.63** | this evaluation (local RTX PRO 6000 Blackwell) |
| 14 | elastic E0 flat_k4_n5 [4]x5 | 95.45 | H100, branch `elastic` all-models report (elastic) |
| 15 | ChronoLoop CL-A-at1_mem16-learned-loopwrite_a1 | 95.33 | H100, branch `elastic` all-models report (chronoloop) |
| 16 | elastic E2 flat_k4_n5 [4]x5 | 95.28 | H100, branch `elastic` all-models report (elastic) |
| 17 | elastic E2 flat_k4_n2 [4,4] | 95.27 | H100, branch `elastic` all-models report (elastic) |
| 18 | elastic E1 flat_k4_n2 [4,4] | 95.22 | H100, branch `elastic` all-models report (elastic) |
| 19 | elastic E2 flat_k2_n5 [2]x5 | 95.07 | H100, branch `elastic` all-models report (elastic) |
| 20 | RT-A2 S5 [4,2,2,2,2] | 94.98 | H100, branch `elastic` all-models report (residual-transport) |
| 21 | elastic E0 flat_k4_n2 [4,4] | 94.97 | H100, branch `elastic` all-models report (elastic) |
| 22 | RT-A2 S7 [4,2] | 94.93 | H100, branch `elastic` all-models report (residual-transport) |
| 23 | all-suites v0 4/4 | 94.92 | H100, branch `elastic` all-models report (all-suites) |
| 24 | elastic E0 flat_k4_n10 [4]x10 | 94.87 | H100, branch `elastic` all-models report (elastic) |
| 25 | elastic E0 flat_k2_n5 [2]x5 | 94.85 | H100, branch `elastic` all-models report (elastic) |
| 26 | elastic E2 flat_k2_n2 [2,2] | 94.75 | H100, branch `elastic` all-models report (elastic) |
| 27 | elastic E2 flat_k4_n10 [4]x10 | 94.70 | H100, branch `elastic` all-models report (elastic) |
| 28 | elastic F0-PB S7 [4,2] | 94.70 | H100, branch `elastic` all-models report (elastic) |
| 29 | elastic F0-PB parent10 [4]x10 | 94.68 | H100, branch `elastic` all-models report (elastic) |
| 30 | **LoopWAM v0 Ka5** | **94.60** | this evaluation (local RTX PRO 6000 Blackwell) |
| 31 | elastic E1 flat_k2_n2 [2,2] | 94.58 | H100, branch `elastic` all-models report (elastic) |
| 32 | RT-B2a parent2 [4,4] | 94.57 | H100, branch `elastic` all-models report (residual-transport) |
| 33 | RT-T1ft S1 [4]+[2]x9 | 94.57 | H100, branch `elastic` all-models report (residual-transport) |
| 34 | elastic E0 flat_k1_n5 [1]x5 | 94.53 | H100, branch `elastic` all-models report (elastic) |
| 35 | elastic E0 flat_k2_n2 [2,2] | 94.53 | H100, branch `elastic` all-models report (elastic) |
| 36 | RT-A2 S8 [4,1] | 94.53 | H100, branch `elastic` all-models report (residual-transport) |
| 37 | elastic E1 flat_k2_n5 [2]x5 | 94.52 | H100, branch `elastic` all-models report (elastic) |
| 38 | elastic F0-PB S8 [4,1] | 94.50 | H100, branch `elastic` all-models report (elastic) |
| 39 | RT-A2 S6 [4,1,1,1,1] | 94.48 | H100, branch `elastic` all-models report (residual-transport) |
| 40 | RT-A S1 [4]+[2]x9 | 94.47 | H100, branch `elastic` all-models report (residual-transport) |
| 41 | elastic E1 flat_k4_n10 [4]x10 | 94.45 | H100, branch `elastic` all-models report (elastic) |
| 42 | all-suites v0 4/1 kv-mix | 94.45 | H100, branch `elastic` all-models report (all-suites) |
| 43 | RT-TF S1 [4]+[2]x9 | 94.42 | H100, branch `elastic` all-models report (residual-transport) |
| 44 | **LoopWAM v0 partition 0/6/6** | **94.42** | this evaluation (local RTX PRO 6000 Blackwell) |
| 45 | elastic E0 flat_k1_n2 [1,1] | 94.33 | H100, branch `elastic` all-models report (elastic) |
| 46 | RT-B4 S1 [4]+[2]x9 | 94.32 | H100, branch `elastic` all-models report (residual-transport) |
| 47 | elastic E1 flat_k1_n5 [1]x5 | 94.30 | H100, branch `elastic` all-models report (elastic) |
| 48 | elastic E2 flat_k1_n5 [1]x5 | 94.25 | H100, branch `elastic` all-models report (elastic) |
| 49 | elastic F0-PB S5 [4,2,2,2,2] | 94.25 | H100, branch `elastic` all-models report (elastic) |
| 50 | RT-T1ft S2 [4]+[1]x9 | 94.18 | H100, branch `elastic` all-models report (residual-transport) |
| 51 | RT-TF S2 [4]+[1]x9 | 94.13 | H100, branch `elastic` all-models report (residual-transport) |
| 52 | RT-A S2 [4]+[1]x9 | 94.12 | H100, branch `elastic` all-models report (residual-transport) |
| 53 | RT-A2 S1 [4]+[2]x9 | 94.10 | H100, branch `elastic` all-models report (residual-transport) |
| 54 | elastic E2 flat_k1_n2 [1,1] | 94.07 | H100, branch `elastic` all-models report (elastic) |
| 55 | elastic E2 flat_k2_n10 [2]x10 | 94.07 | H100, branch `elastic` all-models report (elastic) |
| 56 | RT-A S3 [4,3,2,2,2,1,1,1,1,1] | 94.05 | H100, branch `elastic` all-models report (residual-transport) |
| 57 | elastic E1 flat_k2_n10 [2]x10 | 94.00 | H100, branch `elastic` all-models report (elastic) |
| 58 | elastic E2 flat_k1_n10 [1]x10 | 94.00 | H100, branch `elastic` all-models report (elastic) |
| 59 | all-suites v0 4/1 | 94.00 | H100, branch `elastic` all-models report (all-suites) |
| 60 | RT-Pa S1 [4]+[2]x9 | 93.95 | H100, branch `elastic` all-models report (residual-transport) |
| 61 | RT-A S4 [4,4,1,1,1,1,1,1,1,1] | 93.87 | H100, branch `elastic` all-models report (residual-transport) |
| 62 | elastic E0 flat_k2_n10 [2]x10 | 93.83 | H100, branch `elastic` all-models report (elastic) |
| 63 | elastic SP11 flat_k1_n2 [1,1] | 93.77 | H100, branch `elastic` all-models report (elastic) |
| 64 | elastic F0-PB S6 [4,1,1,1,1] | 93.75 | H100, branch `elastic` all-models report (elastic) |
| 65 | elastic E1 flat_k1_n2 [1,1] | 93.73 | H100, branch `elastic` all-models report (elastic) |
| 66 | elastic E0 flat_k1_n10 [1]x10 | 93.68 | H100, branch `elastic` all-models report (elastic) |
| 67 | **LoopWAM v0 partition 1/6/5** | **93.67** | this evaluation (local RTX PRO 6000 Blackwell) |
| 68 | elastic E1 flat_k1_n10 [1]x10 | 93.48 | H100, branch `elastic` all-models report (elastic) |
| 69 | elastic F0-PB S1 [4]+[2]x9 | 93.38 | H100, branch `elastic` all-models report (elastic) |
| 70 | RT-A2 S2 [4]+[1]x9 | 93.22 | H100, branch `elastic` all-models report (residual-transport) |
| 71 | **LoopWAM v0 partition 5/6/1** | **93.17** | this evaluation (local RTX PRO 6000 Blackwell) |
| 72 | **LoopWAM v0 init core16_21** | **93.12** | this evaluation (local RTX PRO 6000 Blackwell) |
| 73 | RT-Pa S2 [4]+[1]x9 | 92.92 | H100, branch `elastic` all-models report (residual-transport) |
| 74 | init variant core_mean | 92.82 | H100, branch `elastic` all-models report (init-variants) |
| 75 | RT-B4 S2 [4]+[1]x9 | 92.77 | H100, branch `elastic` all-models report (residual-transport) |
| 76 | all-suites v0 2/2 | 92.45 | H100, branch `elastic` all-models report (all-suites) |
| 77 | init variant ffn_magnitude | 92.12 | H100, branch `elastic` all-models report (init-variants) |
| 78 | elastic F0-PB S2 [4]+[1]x9 | 92.05 | H100, branch `elastic` all-models report (elastic) |
| 79 | **LoopWAM v0 partition 1/9/2 k3** | **91.78** | this evaluation (local RTX PRO 6000 Blackwell) |
| 80 | all-suites v0 1/4 | 91.68 | H100, branch `elastic` all-models report (all-suites) |
| 81 | init variant stepwise | 91.47 | H100, branch `elastic` all-models report (init-variants) |
| 82 | **LoopWAM v0 partition 4/3/5 k7** | **90.75** | this evaluation (local RTX PRO 6000 Blackwell) |
| 83 | **LoopWAM v0 partition 6/6/0** | **90.27** | this evaluation (local RTX PRO 6000 Blackwell) |
| 84 | **LoopWAM Dense-S12** | **89.98** | this evaluation (local RTX PRO 6000 Blackwell) |
| 85 | **LoopWAM v0 Ka3** | **89.47** | this evaluation (local RTX PRO 6000 Blackwell) |
| 86 | **LoopWAM v0 Ka6** | **89.20** | this evaluation (local RTX PRO 6000 Blackwell) |
| 87 | init variant no_video_pretraining | 85.65 | H100, branch `elastic` all-models report (init-variants) |
| 88 | init variant core_random | 82.20 | H100, branch `elastic` all-models report (init-variants) |

## Downloads and provenance

- [All RoboTwin summary rows](all_results/robotwin_results.csv) and [every RoboTwin task/setting result](all_results/robotwin_tasks.csv), including seed and run identity.
- [Matched LIBERO](all_results/libero_matched.csv), [CL/RT](all_results/cl_rt_results.csv), [Plus/Pro](all_results/robustness_results.csv), [Ka sweep](all_results/v0_ka_results.csv).
- [Machine-readable consolidated snapshot](all_results/results.json) and [source hashes / reconciliation record](all_results/reconciliation.json).

This is a local report update. It does not push commits or synchronize every repository branch. Historical branch documents retain their original dates and provenance.
