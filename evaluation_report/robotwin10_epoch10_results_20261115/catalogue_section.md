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
