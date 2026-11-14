# RoboTwin-10 epoch 10: detailed task results

Updated **2026-11-15T02:09:56+07:00** (Vietnam time). Scope: **v0 4/2, v0 4/1, v0 2/2, v0 1/4 and Dense-12**. **v0 4/4 is excluded from this update; its existing results and files are unchanged.**

Five model versions and six completed evaluations are documented. The latest Dense-12 continuation is used in the five-model task comparison, and the earlier Dense-12 evaluation is retained separately. All six runs have passing saved final audits. This report freshly reconciles **12,000 accepted episode records**, 120 task/setting cells, both CSV exports, aggregate JSON results and completion-file hashes. No simulation or video-content audit was rerun for this documentation update.

## Overall scores

Every row uses seed **42**, **NFE 10**, and **2,000 accepted episodes**: 10 tasks × clean/randomized × 100 episodes. Clean and randomized denominators are 1,000 each.

| Model / evaluation | Clean | Randomized | All successes | Overall SR | Execution |
| --- | --- | --- | --- | --- | --- |
| [v0 4/2](../robotwin10_epoch10_queue_20261113/v0_4x2/final_report.md) | 825/1000 (82.50%) | 823/1000 (82.30%) | 1648/2000 | **82.40%** | Local RTX PRO 6000 |
| [v0 4/1](../robotwin10_v0_4x1_epoch10_20261108/final_report.md) | 800/1000 (80.00%) | 819/1000 (81.90%) | 1619/2000 | **80.95%** | Local RTX PRO 6000 |
| [v0 2/2](../robotwin10_epoch10_queue_20261113/v0_2x2/final_report.md) | 786/1000 (78.60%) | 792/1000 (79.20%) | 1578/2000 | **78.90%** | Local RTX PRO 6000 |
| [v0 1/4](../robotwin10_epoch10_remote189_20261113/v0_1x4/final_report.md) | 802/1000 (80.20%) | 789/1000 (78.90%) | 1591/2000 | **79.55%** | aLinh RTX PRO 5000 |
| [Dense-12 — latest continuation](../robotwin10_epoch10_remote189_20261113/dense_s12/final_report.md) | 769/1000 (76.90%) | 776/1000 (77.60%) | 1545/2000 | **77.25%** | 400 local + 1,600 aLinh episodes |
| [Dense-12 — earlier local run](../robotwin10_dense_s12_epoch10_20261109/final_report.md) | 754/1000 (75.40%) | 763/1000 (76.30%) | 1517/2000 | **75.85%** | Local RTX PRO 6000 |

The two Dense-12 rows use the same epoch-10 checkpoint SHA256. They are separate evaluations; their episodes are not pooled, and tasks are not selected from whichever run scored better. “Latest” follows execution history, not score ranking. No completed Dense-30 epoch-10 evaluation was found in the saved RT-10 result inventory; its epoch-5 result is not substituted here.

## Task comparison — latest completed run per included model

Each entry below is the task's combined SR across **200 episodes** (100 clean + 100 randomized). The final row is the full 2,000-episode SR. Equal task denominators make the mean of task SRs equal to overall SR.

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

## Detailed clean and randomized results

Setting SRs are shown explicitly with their denominators. Each model's combined task SR is `(clean successes + randomized successes) / 200 × 100`; it is not a score from a separate set of episodes.

### v0 4/2

Epoch 10 / update 90,890; Local RTX PRO 6000. [Original final report](../robotwin10_epoch10_queue_20261113/v0_4x2/final_report.md) · [Task CSV](../robotwin10_epoch10_queue_20261113/v0_4x2/task_results.csv) · [Episode CSV](../robotwin10_epoch10_queue_20261113/v0_4x2/episodes.csv) · [Passed audit](../robotwin10_epoch10_queue_20261113/v0_4x2/final_audit.json).

| Task | Clean successes | Clean SR | Randomized successes | Randomized SR | Combined successes | Combined SR |
| --- | --- | --- | --- | --- | --- | --- |
| adjust_bottle | 100/100 | 100.00% | 100/100 | 100.00% | 200/200 | 100.00% |
| grab_roller | 100/100 | 100.00% | 100/100 | 100.00% | 200/200 | 100.00% |
| hanging_mug | 54/100 | 54.00% | 48/100 | 48.00% | 102/200 | 51.00% |
| move_stapler_pad | 69/100 | 69.00% | 70/100 | 70.00% | 139/200 | 69.50% |
| open_microwave | 80/100 | 80.00% | 75/100 | 75.00% | 155/200 | 77.50% |
| press_stapler | 81/100 | 81.00% | 89/100 | 89.00% | 170/200 | 85.00% |
| scan_object | 88/100 | 88.00% | 83/100 | 83.00% | 171/200 | 85.50% |
| stack_blocks_two | 97/100 | 97.00% | 97/100 | 97.00% | 194/200 | 97.00% |
| stamp_seal | 91/100 | 91.00% | 92/100 | 92.00% | 183/200 | 91.50% |
| turn_switch | 65/100 | 65.00% | 69/100 | 69.00% | 134/200 | 67.00% |
| **Total** | 825/1000 | 82.50% | 823/1000 | 82.30% | 1648/2000 | **82.40%** |

### v0 4/1

Epoch 10 / update 90,890; Local RTX PRO 6000. [Original final report](../robotwin10_v0_4x1_epoch10_20261108/final_report.md) · [Task CSV](../robotwin10_v0_4x1_epoch10_20261108/task_results.csv) · [Episode CSV](../robotwin10_v0_4x1_epoch10_20261108/episodes.csv) · [Passed audit](../robotwin10_v0_4x1_epoch10_20261108/final_audit.json).

| Task | Clean successes | Clean SR | Randomized successes | Randomized SR | Combined successes | Combined SR |
| --- | --- | --- | --- | --- | --- | --- |
| adjust_bottle | 99/100 | 99.00% | 99/100 | 99.00% | 198/200 | 99.00% |
| grab_roller | 100/100 | 100.00% | 100/100 | 100.00% | 200/200 | 100.00% |
| hanging_mug | 51/100 | 51.00% | 39/100 | 39.00% | 90/200 | 45.00% |
| move_stapler_pad | 78/100 | 78.00% | 73/100 | 73.00% | 151/200 | 75.50% |
| open_microwave | 71/100 | 71.00% | 70/100 | 70.00% | 141/200 | 70.50% |
| press_stapler | 83/100 | 83.00% | 86/100 | 86.00% | 169/200 | 84.50% |
| scan_object | 81/100 | 81.00% | 85/100 | 85.00% | 166/200 | 83.00% |
| stack_blocks_two | 99/100 | 99.00% | 99/100 | 99.00% | 198/200 | 99.00% |
| stamp_seal | 83/100 | 83.00% | 95/100 | 95.00% | 178/200 | 89.00% |
| turn_switch | 55/100 | 55.00% | 73/100 | 73.00% | 128/200 | 64.00% |
| **Total** | 800/1000 | 80.00% | 819/1000 | 81.90% | 1619/2000 | **80.95%** |

### v0 2/2

Epoch 10 / update 90,890; Local RTX PRO 6000. [Original final report](../robotwin10_epoch10_queue_20261113/v0_2x2/final_report.md) · [Task CSV](../robotwin10_epoch10_queue_20261113/v0_2x2/task_results.csv) · [Episode CSV](../robotwin10_epoch10_queue_20261113/v0_2x2/episodes.csv) · [Passed audit](../robotwin10_epoch10_queue_20261113/v0_2x2/final_audit.json).

| Task | Clean successes | Clean SR | Randomized successes | Randomized SR | Combined successes | Combined SR |
| --- | --- | --- | --- | --- | --- | --- |
| adjust_bottle | 100/100 | 100.00% | 99/100 | 99.00% | 199/200 | 99.50% |
| grab_roller | 100/100 | 100.00% | 100/100 | 100.00% | 200/200 | 100.00% |
| hanging_mug | 42/100 | 42.00% | 38/100 | 38.00% | 80/200 | 40.00% |
| move_stapler_pad | 60/100 | 60.00% | 67/100 | 67.00% | 127/200 | 63.50% |
| open_microwave | 55/100 | 55.00% | 61/100 | 61.00% | 116/200 | 58.00% |
| press_stapler | 90/100 | 90.00% | 92/100 | 92.00% | 182/200 | 91.00% |
| scan_object | 90/100 | 90.00% | 84/100 | 84.00% | 174/200 | 87.00% |
| stack_blocks_two | 99/100 | 99.00% | 97/100 | 97.00% | 196/200 | 98.00% |
| stamp_seal | 91/100 | 91.00% | 89/100 | 89.00% | 180/200 | 90.00% |
| turn_switch | 59/100 | 59.00% | 65/100 | 65.00% | 124/200 | 62.00% |
| **Total** | 786/1000 | 78.60% | 792/1000 | 79.20% | 1578/2000 | **78.90%** |

### v0 1/4

Epoch 10 / update 90,890; aLinh RTX PRO 5000. [Original final report](../robotwin10_epoch10_remote189_20261113/v0_1x4/final_report.md) · [Task CSV](../robotwin10_epoch10_remote189_20261113/v0_1x4/task_results.csv) · [Episode CSV](../robotwin10_epoch10_remote189_20261113/v0_1x4/episodes.csv) · [Passed audit](../robotwin10_epoch10_remote189_20261113/v0_1x4/final_audit.json).

| Task | Clean successes | Clean SR | Randomized successes | Randomized SR | Combined successes | Combined SR |
| --- | --- | --- | --- | --- | --- | --- |
| adjust_bottle | 100/100 | 100.00% | 99/100 | 99.00% | 199/200 | 99.50% |
| grab_roller | 100/100 | 100.00% | 100/100 | 100.00% | 200/200 | 100.00% |
| hanging_mug | 49/100 | 49.00% | 42/100 | 42.00% | 91/200 | 45.50% |
| move_stapler_pad | 65/100 | 65.00% | 62/100 | 62.00% | 127/200 | 63.50% |
| open_microwave | 74/100 | 74.00% | 73/100 | 73.00% | 147/200 | 73.50% |
| press_stapler | 89/100 | 89.00% | 88/100 | 88.00% | 177/200 | 88.50% |
| scan_object | 87/100 | 87.00% | 79/100 | 79.00% | 166/200 | 83.00% |
| stack_blocks_two | 98/100 | 98.00% | 97/100 | 97.00% | 195/200 | 97.50% |
| stamp_seal | 82/100 | 82.00% | 84/100 | 84.00% | 166/200 | 83.00% |
| turn_switch | 58/100 | 58.00% | 65/100 | 65.00% | 123/200 | 61.50% |
| **Total** | 802/1000 | 80.20% | 789/1000 | 78.90% | 1591/2000 | **79.55%** |

### Dense-12 — latest continuation

Epoch 10 / update 90,890; 400 local + 1,600 aLinh episodes. [Original final report](../robotwin10_epoch10_remote189_20261113/dense_s12/final_report.md) · [Task CSV](../robotwin10_epoch10_remote189_20261113/dense_s12/task_results.csv) · [Episode CSV](../robotwin10_epoch10_remote189_20261113/dense_s12/episodes.csv) · [Passed audit](../robotwin10_epoch10_remote189_20261113/dense_s12/final_audit.json).

| Task | Clean successes | Clean SR | Randomized successes | Randomized SR | Combined successes | Combined SR |
| --- | --- | --- | --- | --- | --- | --- |
| adjust_bottle | 100/100 | 100.00% | 99/100 | 99.00% | 199/200 | 99.50% |
| grab_roller | 100/100 | 100.00% | 100/100 | 100.00% | 200/200 | 100.00% |
| hanging_mug | 46/100 | 46.00% | 43/100 | 43.00% | 89/200 | 44.50% |
| move_stapler_pad | 54/100 | 54.00% | 68/100 | 68.00% | 122/200 | 61.00% |
| open_microwave | 64/100 | 64.00% | 42/100 | 42.00% | 106/200 | 53.00% |
| press_stapler | 91/100 | 91.00% | 90/100 | 90.00% | 181/200 | 90.50% |
| scan_object | 79/100 | 79.00% | 80/100 | 80.00% | 159/200 | 79.50% |
| stack_blocks_two | 97/100 | 97.00% | 99/100 | 99.00% | 196/200 | 98.00% |
| stamp_seal | 84/100 | 84.00% | 90/100 | 90.00% | 174/200 | 87.00% |
| turn_switch | 54/100 | 54.00% | 65/100 | 65.00% | 119/200 | 59.50% |
| **Total** | 769/1000 | 76.90% | 776/1000 | 77.60% | 1545/2000 | **77.25%** |

### Dense-12 — earlier local run

Epoch 10 / update 90,890; Local RTX PRO 6000. [Original final report](../robotwin10_dense_s12_epoch10_20261109/final_report.md) · [Task CSV](../robotwin10_dense_s12_epoch10_20261109/task_results.csv) · [Episode CSV](../robotwin10_dense_s12_epoch10_20261109/episodes.csv) · [Passed audit](../robotwin10_dense_s12_epoch10_20261109/final_audit.json).

| Task | Clean successes | Clean SR | Randomized successes | Randomized SR | Combined successes | Combined SR |
| --- | --- | --- | --- | --- | --- | --- |
| adjust_bottle | 100/100 | 100.00% | 99/100 | 99.00% | 199/200 | 99.50% |
| grab_roller | 100/100 | 100.00% | 100/100 | 100.00% | 200/200 | 100.00% |
| hanging_mug | 45/100 | 45.00% | 43/100 | 43.00% | 88/200 | 44.00% |
| move_stapler_pad | 53/100 | 53.00% | 58/100 | 58.00% | 111/200 | 55.50% |
| open_microwave | 58/100 | 58.00% | 51/100 | 51.00% | 109/200 | 54.50% |
| press_stapler | 91/100 | 91.00% | 90/100 | 90.00% | 181/200 | 90.50% |
| scan_object | 72/100 | 72.00% | 77/100 | 77.00% | 149/200 | 74.50% |
| stack_blocks_two | 97/100 | 97.00% | 97/100 | 97.00% | 194/200 | 97.00% |
| stamp_seal | 86/100 | 86.00% | 86/100 | 86.00% | 172/200 | 86.00% |
| turn_switch | 52/100 | 52.00% | 62/100 | 62.00% | 114/200 | 57.00% |
| **Total** | 754/1000 | 75.40% | 763/1000 | 76.30% | 1517/2000 | **75.85%** |

## Protocol, provenance and exclusions

All included exports are trained for **10 epochs / 90,890 optimizer updates** on the ten-task RoboTwin set. Core settings match: seed42; unseen instructions; NFE10; predict32 / execute24; original 14-D state/action normalization and task horizons; original scene admission; FP32 policy weights/BF16 compute; shared T5 and CUDA graphs; original camera preprocessing and rendering. The 4/2, 4/1, 2/2 and 1/4 labels describe video/action loop counts, not NFE or chunk length. Dense-12 is the dense 12-block architecture.

The accepted denominators exclude scene-admission failures, smoke/calibration episodes, and failed or interrupted attempts. The final audits retain those exclusions:

| Evaluation | Saved audit | Accepted cells | Scored episodes | Excluded attempts | Excluded logged partial episodes | Workers in final controller |
| --- | --- | --- | --- | --- | --- | --- |
| v0 4/2 | Passed | 20/20 | 2000 | 0 | 0 | 4 |
| v0 4/1 | Passed | 20/20 | 2000 | 12 | 486 | 4 |
| v0 2/2 | Passed | 20/20 | 2000 | 0 | 0 | 4 |
| v0 1/4 | Passed | 20/20 | 2000 | 2 | 112 | 2 |
| Dense-12 — latest continuation | Passed | 20/20 | 2000 | 8 | 266 | 2 |
| Dense-12 — earlier local run | Passed | 20/20 | 2000 | 0 | 0 | 4 |

- **Dense-12 latest:** 400 fully verified episodes (`adjust_bottle` and `grab_roller`, both settings) originated locally and were transferred to aLinh. The other 1,600 were evaluated on aLinh. Eight interrupted local attempts / 266 logged episodes remain excluded.
- **v0 1/4:** all 2,000 accepted episodes were generated on aLinh. On resume, 800 already completed episodes were retained, and the remaining 1,200 were evaluated. Two interrupted `open_microwave` attempts / 112 logged episodes remain excluded.
- **v0 4/1:** the accepted final result contains 2,000 episodes. Its saved exclusion ledger records 12 failed/interrupted attempts and 486 logged partial episodes following earlier resource failures. Those attempts do not contribute to the displayed SR.
- **4/2, 2/2 and the earlier Dense-12:** their saved audits record zero failed attempts and zero excluded partial episodes.

Final-controller worker counts do not describe every earlier attempt. Preserved launches may describe only the last execution segment, so their durations should not be treated as total time across stops, retries and migration.

The original Python instruction-template RNG is unseeded. Equal seed42 and matched benchmark settings do not guarantee identical scenes/instructions across independent runs, and hosts/concurrency differ. These are observed aggregate scores, not claims of perfectly paired rollouts or statistically significant differences.

### Checkpoint identity

| Evaluation | Native variant | Epoch / update | Checkpoint SHA256 |
| --- | --- | --- | --- |
| v0 4/2 | v0-video4-action2 | 10 / 90890 | `f36997a9de1a147a787960de9f5317964a03c45f54648ef343e63a2578ed2755` |
| v0 4/1 | v0-video4-action1 | 10 / 90890 | `2407441a5f15bc19c5327cf79454d96cc8097b15a194b2ac0c0a50e7b3404af8` |
| v0 2/2 | v0-video2-action2 | 10 / 90890 | `89a87a875e92a9e19ab255609116aca3a86a31f654d93b88a5dd63c630a09786` |
| v0 1/4 | v0-video1-action4 | 10 / 90890 | `bd15ddb617cf78014582deb4787890d05c6519e2017a0c0204a9db553ec442d6` |
| Dense-12 — latest continuation | dense-s12 | 10 / 90890 | `a222704c10688bf42a531a4a4175fe45a1b172c437cf9b9efa435c4b4fae1ac8` |
| Dense-12 — earlier local run | dense-s12 | 10 / 90890 | `a222704c10688bf42a531a4a4175fe45a1b172c437cf9b9efa435c4b4fae1ac8` |

## Verification and exports

This update recalculated success counts from all 12,000 saved episode flags and checked them against task/episode CSVs and final JSON totals. It verified each COMPLETE.json SHA256 against its passed audit, checked 20 unique task/setting cells per run and 100 unique accepted scene seeds within each cell, and verified the epoch/update, native variant, seed, NFE, action horizon and exclusion counts. It also checked that the shared protocol fields match across the six records.

This is a fresh report/record reconciliation. Media bytes, simulator packages and GPU numerical gates were not re-audited here; their verification is the scope of each saved final audit. All original completion records, audits, per-episode CSVs and task CSVs remain unchanged. The v0 4/4 epoch-10 directory is hash-protected and is outside the updated result set.

- [Overall results CSV](summary.csv): six evaluation rows, with run identity and checkpoint hashes.
- [Detailed task CSV](task_results.csv): 60 rows, one row per task per evaluation, with separate clean/randomized/combined counts and SR.
- [Latest-run task comparison CSV](task_comparison.csv): ten rows covering the five included model versions.
- [Full JSON results and input hashes](results.json).
- [Reconciliation receipt](evidence/reconciliation.json) and [final report verification](evidence/report_validation.json).
- [Reconciliation/report generator](reconcile_and_render.py); it reads source records and writes only derived files in this report directory.

The main [results catalogue](../all_results.md) and [report index](../README.md) link this dated update. Their older exports and other benchmark sections retain their original scope. Prior Markdown files are backed up under `evidence/before_update/`.
