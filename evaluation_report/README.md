# Evaluation reports

Shared snapshot: **2026-11-06**. Report contents are synchronized across all GitHub branches. Each report retains its own run date and completion status.

## Start here

- Consolidated exports: [LIBERO matched](all_results/libero_matched.csv), [RoboTwin](all_results/robotwin_results.csv), [robustness](all_results/robustness_results.csv), and [all results JSON](all_results/results.json).
- Recent comparisons: [H100 retrains](robotwin10_h100_20261106/report.md) and [Dense-s12 versus v0 1/4](robotwin10_dense12_v014_20261105/report.md).
- Added in the second 2026-11-06 sync (H100 and ws results not previously in this snapshot): [CL-0 elastic distillation, matched LIBERO](elastic_cl0_libero_20261104/README.md) (complete), [RoboCasa-GR1 v0 training and official evaluation](robocasa_gr1_v0_20261101/README.md) (complete), [RoboCasa-GR1 epochs 6-10 continuation](robocasa_gr1_continuation_20261106/README.md) (in progress at snapshot) and [RoboTwin-50 elastic distillation](elastic_robotwin_20261106/README.md) (in progress at snapshot). These four folders are not part of the consolidated `all_results` exports, which are built on ws by `all_results/build_snapshot.py`.
- [Complete file index](FILE_INDEX.md) lists every current report and evidence file.

## Find a report

| Report group | Open | Contents |
|---|---|---|
| `all_results` | [results.json](all_results/results.json) | Consolidated exports with reconciliation, validation, and source snapshots. |
| `branch_reports` | [index.json](branch_reports/index.json) | Branch-specific report documents with source commits in index.json. |
| `campaign` | [report.html](campaign/report.html) · [detailed_report.html](campaign/detailed_report.html) | Campaign results, queue status, and per-condition evidence. |
| `dense_full` | [report.md](dense_full/report.md) · [queue_status.json](dense_full/queue_status.json) | Dense full-model evaluation setup, status, and cancellation records. |
| `draft` | [report.html](draft/report.html) · [results.json](draft/results.json) | Earlier draft consolidated campaign snapshot; retained for reference. |
| `elastic_cl0_libero_20261104` | [README.md](elastic_cl0_libero_20261104/README.md) · [cl0_eval_report.md](elastic_cl0_libero_20261104/eval/cl0_eval_report.md) | CL-0 elastic E0/E1/E2 distillation: 27 matched LIBERO conditions x 6,000 episodes, training records, independent check. |
| `elastic_robotwin_20261106` | [README.md](elastic_robotwin_20261106/README.md) · [REPORT.md](elastic_robotwin_20261106/REPORT.md) | RoboTwin-50 elastic port: 8/4-GPU timing test and the epoch-5 E0/E1/E2 queue (in progress at snapshot). |
| `final` | [report.html](final/report.html) · [results.json](final/results.json) | Final consolidated campaign snapshot. |
| `matched_baselines` | [README.md](matched_baselines/README.md) | Matched LIBERO comparisons and supporting evidence. |
| `matched_pro6000_20261101` | [README.md](matched_pro6000_20261101/README.md) · [results.json](matched_pro6000_20261101/results.json) | Matched LIBERO batch from November 1, including condition comparisons. |
| `robocasa_gr1_continuation_20261106` | [README.md](robocasa_gr1_continuation_20261106/README.md) · [STATUS.json](robocasa_gr1_continuation_20261106/record/STATUS.json) | RoboCasa-GR1 epochs 6-10 continuation with per-epoch evaluation (in progress at snapshot). |
| `robocasa_gr1_v0_20261101` | [README.md](robocasa_gr1_v0_20261101/README.md) · [official_results_all.md](robocasa_gr1_v0_20261101/eval_ws/results/official_results_all.md) | RoboCasa-GR1 v0 training, official 24-task evaluation of epochs 3/4/5 on H100 and ws, epoch-4 action-step sweep. |
| `robotwin10_dense12_v014_20261105` | [report.md](robotwin10_dense12_v014_20261105/report.md) | Run reports, status, and supporting evidence. |
| `robotwin10_dense_s12_20261105` | [report.md](robotwin10_dense_s12_20261105/report.md) · [final_results.json](robotwin10_dense_s12_20261105/final_results.json) | Run reports, status, and supporting evidence. |
| `robotwin10_dense_s30` | [report.md](robotwin10_dense_s30/report.md) · [final_results.json](robotwin10_dense_s30/final_results.json) | Run reports, status, and supporting evidence. |
| `robotwin10_h100_20261106` | [report.md](robotwin10_h100_20261106/report.md) | Run reports, status, and supporting evidence. |
| `robotwin10_h100_v0_4x2_remote189_20261106` | [report.md](robotwin10_h100_v0_4x2_remote189_20261106/report.md) · [final_results.json](robotwin10_h100_v0_4x2_remote189_20261106/final_results.json) | Run reports, status, and supporting evidence. |
| `robotwin10_h100_v0_4x4_20261106` | [report.md](robotwin10_h100_v0_4x4_20261106/report.md) · [final_results.json](robotwin10_h100_v0_4x4_20261106/final_results.json) | Run reports, status, and supporting evidence. |
| `robotwin10_seed43_two_tasks` | [queue_status.json](robotwin10_seed43_two_tasks/queue_status.json) | Run reports, status, and supporting evidence. |
| `robotwin10_v0_1x4_remote137_20261105` | [report.md](robotwin10_v0_1x4_remote137_20261105/report.md) · [final_results.json](robotwin10_v0_1x4_remote137_20261105/final_results.json) | Run reports, status, and supporting evidence. |
| `robotwin10_v0_2x2` | [report.md](robotwin10_v0_2x2/report.md) · [final_results.json](robotwin10_v0_2x2/final_results.json) | Run reports, status, and supporting evidence. |
| `robotwin10_v0_4x1` | [report.md](robotwin10_v0_4x1/report.md) · [final_results.json](robotwin10_v0_4x1/final_results.json) | Run reports, status, and supporting evidence. |
| `robotwin10_v0_4x1_local_20261105` | [report.md](robotwin10_v0_4x1_local_20261105/report.md) · [final_results.json](robotwin10_v0_4x1_local_20261105/final_results.json) | Run reports, status, and supporting evidence. |
| `robotwin10_v0_4x2` | [report.md](robotwin10_v0_4x2/report.md) · [final_results.json](robotwin10_v0_4x2/final_results.json) | Run reports, status, and supporting evidence. |
| `robotwin10_v0_4x4` | [report.md](robotwin10_v0_4x4/report.md) · [final_results.json](robotwin10_v0_4x4/final_results.json) | Run reports, status, and supporting evidence. |
| `robotwin10_v0_4x4_remote137` | [report.md](robotwin10_v0_4x4_remote137/report.md) · [queue_status.json](robotwin10_v0_4x4_remote137/queue_status.json) | Run reports, status, and supporting evidence. |
| `robotwin10_v0_4x4_remote137_20261105` | [report.md](robotwin10_v0_4x4_remote137_20261105/report.md) · [final_results.json](robotwin10_v0_4x4_remote137_20261105/final_results.json) | Run reports, status, and supporting evidence. |
| `robotwin10_v0_4x4_remote189` | [report.md](robotwin10_v0_4x4_remote189/report.md) · [final_results.json](robotwin10_v0_4x4_remote189/final_results.json) | Run reports, status, and supporting evidence. |
| `robotwin_v0` | [report.md](robotwin_v0/report.md) · [final_results.json](robotwin_v0/final_results.json) | RoboTwin v0 results, task and episode exports, retries, and audits. |
| `robotwin_v0_epoch06` | [report.md](robotwin_v0_epoch06/report.md) | Run reports, status, and supporting evidence. |
| `robotwin_v0_epoch10` | [report.md](robotwin_v0_epoch10/report.md) · [final_results.json](robotwin_v0_epoch10/final_results.json) | Run reports, status, and supporting evidence. |
| `robustness` | [report.md](robustness/report.md) · [report.html](robustness/report.html) | LIBERO-Plus and LIBERO-Pro robustness results and provenance. |
| `v0_ka_sweep` | [report.md](v0_ka_sweep/report.md) · [queue_status.json](v0_ka_sweep/queue_status.json) | v0 action-horizon sweep setup and status. |

## Evidence and history

- [Complete file index](FILE_INDEX.md): reports, exports, status files, scripts, logs, and provenance grouped by folder.
- [Preserved versions](ARCHIVE_INDEX.md): previous or differing copies retained during synchronization and cleanup.
- [Integrity inventory](integrity_inventory_20261106b.json): SHA-256 hashes for the snapshot files (previous: [first 2026-11-06 sync](integrity_inventory_20261106.json)).
- [Sync manifest](sync_manifest_20261106b.json): sources, branch heads, additions and preserved versions of the second 2026-11-06 sync (previous: [first 2026-11-06 sync](sync_manifest_20261106.json)).
- [Independent check of the added LIBERO and RoboCasa-GR1 tables](verify_h100_additions_20261106.py): recomputes them from the stored per-episode evidence.

Result data and original paths are retained. A draft, cancelled run, or partial queue is not promoted to a completed result by this index. Completion and metrics should be read from the corresponding report.
