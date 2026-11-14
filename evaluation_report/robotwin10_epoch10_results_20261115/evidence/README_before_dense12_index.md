## Elastic E0 full RoboTwin — 2026-11-07

[Archived result and exact protocol](robotwin_elastic_e0_20261107/report.md): epoch 1, update 47,462; schedules `[1,1]`, `[4,4]`, `[4,4,4,4,4]`; seed 42; 10,000 episodes each (50 tasks, clean + randomized). This historical campaign is complete. The newer teacher-epoch10/student-five-epoch comparison is linked below.

# Evaluation reports

**RT-10 epoch-10 task update — November 15, 2026:** [Detailed report](robotwin10_epoch10_results_20261115/README.md) for v0 4/2 **82.40%**, v0 4/1 **80.95%**, v0 2/2 **78.90%**, v0 1/4 **79.55%** and latest Dense-12 **77.25%**. Includes all task clean/randomized counts, a task comparison, and the earlier Dense-12 run **75.85%** separately. [Task CSV](robotwin10_epoch10_results_20261115/task_results.csv).

**RT-50 Elastic update — November 13, 2026:** [Final detailed [1,1] / [2,2] / [4,4] report](robotwin_elastic_e0_teacher10_5ep_20261111/final_report.md): **89.28% / 90.60% / 91.49%**, all complete and audited; seed 42, NFE 2, 10,000 episodes each. Includes all 50 tasks, checkpoint/protocol verification, actual runtimes, epoch-1 comparison, CSV/JSON exports and reproducibility limitations. Other historical sections retain their own dates.

Original broad index snapshot: **2026-11-07T02:51:27+07:00**; targeted updates are dated above. The last repository-wide branch report synchronization was October 30; these local updates have not been pushed or synchronized across branches.

**[All results — consolidated report](all_results.md)** includes all retained RoboTwin runs, full 50-task epoch comparisons, matched LIBERO, CL/RT, Plus/Pro, the Ka sweep, exclusions and historical branch-reported results. [CSV/JSON exports and reconciliation evidence](all_results/).

| Path | Current content |
| --- | --- |
| [RT-10 epoch 10 — detailed task results](robotwin10_epoch10_results_20261115/README.md) | Five requested model versions, six audited evaluations; all clean/randomized task counts; latest Dense-12 and earlier run reported separately; updated November 15 |
| [Elastic E0 teacher10/student5ep — full RT-50](robotwin_elastic_e0_teacher10_5ep_20261111/final_report.md) | Final [1,1] **89.28%**, [2,2] **90.60%**, [4,4] **91.49%**; 30,000 scored episodes across three audited NFE-2 evaluations; fresh report reconciliation November 13 |
| [Elastic E2 — [1,1] / NFE2](robotwin_elastic_e2_20261109/report.md) | Started 2026-11-09 on server137; full50tasks,10,000episodes,seed42; see live report for stage and SR |
| [RoboCasa-GR1 — epochs 9 and 10](robocasa_gr1_epochs09_10_20261108/final_report.md) | Final audit updated 2026-11-09: epoch 9 **628/1200 = 52.33%**, epoch 10 **600/1200 = 50.00%**; 24 tasks; environment seed 0, policy noise seed 42; execute 12 / predict 32 |
| [All results](all_results.md) | Current overview with source links; distinguishes saved audits, record-only reconciliation, external results and unscored history |
| [H100 RoboTwin-10 retrains](robotwin10_h100_20261106/report.md) | Both complete and audited:4/4 78.65%,4/2 80.80%;seed 42; 2,000 episodes each |
| [Dense-S30 — local](robotwin10_dense_s30/final_report.md) | Complete: 1672/2000 = **83.60%**; seed 42; saved final audit passed |
| [Dense-S12 — local, valid four-worker rerun](robotwin10_dense_s12_20261105/final_report.md) | Complete: 1444/2000 = **72.20%**; seed 42; saved final audit passed |
| [v0 4/4 — original local](robotwin10_v0_4x4/final_report.md) | Complete: 1577/2000 = **78.85%**; seed 42; saved final audit passed |
| [v0 4/4 — rerun server189](robotwin10_v0_4x4_remote189/final_report.md) | Complete: 1564/2000 = **78.20%**; seed 42; saved final audit passed |
| [v0 4/4 — rerun server137](robotwin10_v0_4x4_remote137_20261105/final_report.md) | Complete: 1559/2000 = **77.95%**; seed 42; saved final audit passed |
| [v0 4/2 — original local](robotwin10_v0_4x2/final_report.md) | Complete: 1599/2000 = **79.95%**; seed 42; saved final audit passed |
| [v0 2/2 — local](robotwin10_v0_2x2/final_report.md) | Complete: 1466/2000 = **73.30%**; seed 42; saved final audit passed |
| [v0 4/1 — server189](robotwin10_v0_4x1/final_report.md) | Complete: 1627/2000 = **81.35%**; seed 42; saved final audit passed |
| [v0 4/1 — local rerun](robotwin10_v0_4x1_local_20261105/final_report.md) | Complete: 1617/2000 = **80.85%**; seed 42; saved final audit passed |
| [v0 1/4 — server137, valid four-worker rerun](robotwin10_v0_1x4_remote137_20261105/final_report.md) | Complete: 1463/2000 = **73.15%**; seed 42; saved final audit passed |
| [v0 4/4 — H100 retrain, local evaluation](robotwin10_h100_v0_4x4_20261106/final_report.md) | Complete: 1573/2000 = **78.65%**; seed 42; saved final audit passed |
| [v0 4/2 — H100 retrain, server189 evaluation](robotwin10_h100_v0_4x2_remote189_20261106/final_report.md) | Complete: 1616/2000 = **80.80%**; seed 42; saved final audit passed |
| [v0 4/4 — epoch5, full 50 training](robotwin_v0/report.md) | Complete: 9173/10000 = **91.73%**; seed 42; saved final audit passed |
| [v0 4/4 — epoch10, full 50 training](robotwin_v0_epoch10/final_report.md) | Complete: 9187/10000 = **91.87%**; seed 42; saved final audit passed |
| [Dense-S30 — seed 43, two tasks only](robotwin10_seed43_two_tasks/dense_s30/final_report.md) | Complete: 274/400 = **68.50%**; seed 43; saved final audit passed |
| [v0 4/4 — seed 43, two tasks only](robotwin10_seed43_two_tasks/v0_4x4/final_report.md) | Complete: 217/400 = **54.25%**; seed 43; saved final audit passed |
| [v0 4/4 — epoch6, full 50 training](robotwin_v0_epoch06/report.md) | Complete: 8985/10000 = **89.85%**; seed 42; completion records reconciled; no saved final audit |
| [Matched LIBERO](matched_baselines/README.md) | Public FasterWAM/Light-WAM, LoopWAM/dense/partition/init/Ka results and historical branch catalogue; source provenance retained |
| [Original campaign](final/report.html) | Original LoopWAM/FastWAM/Light-WAM results, seeds42/43/44, task and seed exports |
| [CL/RT campaign](campaign/detailed_report.html) | 26 completed conditions in the saved ledger, including 12 external-complete conditions; 29 historical not-published entries are not scored results |
| [LIBERO-Plus/Pro](robustness/report.md) | Complete:14 jobs,16,142 episodes, seven CL models,seed 42;Plus15% protocol and Pro Position+Task |
| [Original Ka queue](v0_ka_sweep/report.md) | Ka1 12.30%,Ka2 55.57%;6,000 episodes each. Ka3/5/6 removed from that queue; separate later matched results are in matched_baselines |
| [Original dense queue](dense_full/report.md) | Cancelled without scored episodes. Separate completed LIBERO dense evaluations are in matched_baselines; RoboTwin dense results above |
| [Branch report snapshots](branch_reports/) | Historical per-branch Markdown and source commit index; not newly executed or revalidated simulations |

Historical draft/final and campaign records are preserved. The consolidated report does not sum overlapping ledgers, combine different benchmarks, substitute seed 43 task results into seed 42, or count invalidated attempts.
