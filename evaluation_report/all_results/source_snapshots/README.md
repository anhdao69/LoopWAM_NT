# Evaluation reports

The shared report snapshot was synchronized across repository branches on 2026-10-30. This local checkout also includes the completed RoboTwin report updated on 2026-10-31; local additions have not been pushed or synchronized to other branches.

| Path | Content |
|---|---|
| [`robotwin10_dense_s30/report.md`](robotwin10_dense_s30/report.md) | RoboTwin-10 dense-s30: running with verified checkpoint/source; 2,000 episodes, seed 42, clean/randomized |
| [`robotwin_v0_epoch06/report.md`](robotwin_v0_epoch06/report.md) | LoopWAM v0 RoboTwin epoch 6; running with the same 50-task clean/randomized protocol, seed 42, 10,000 episodes; four workers; see live progress and setup evidence |
| [`robotwin_v0/report.md`](robotwin_v0/report.md) | **Completed 2026-10-31.** LoopWAM v0 RoboTwin, all 50 tasks, clean + randomized, seed 42, 10,000 episodes: 91.60% clean / 91.86% randomized / 91.73% combined; full artifact audit, per-task and per-episode exports, retry history |
| [`matched_baselines/`](matched_baselines) | **Latest.** All four-suite matched LIBERO results (protocol `lightwam_20261009_matched`, 6,000 episodes per model) in one table: the public FasterWAM and Light-WAM releases and 12 LoopWAM exports evaluated and verified 2026-10-27..30, plus every matched result reported on the other branches (Elastic, Residual Transport, ChronoLoop, init variants) |
| [`branch_reports/`](branch_reports) | Snapshot of every branch's report documents (Markdown under `plans/` that differs from `LoopWAM_NT`), with `index.json` giving the source branch and commit; rebuilt by `sync_branch_reports.py` |
| [`final/`](final), [`draft/`](draft) | Consolidated LoopWAM campaign report (audited metrics, per-episode records); built by `build_report.py` |
| [`campaign/`](campaign) | Campaign queue, per-condition and per-episode results |
| [`robustness/`](robustness) | LIBERO-Plus / LIBERO-Pro robustness evaluation |
