# Evaluation reports

This directory is kept identical on every branch of the repository (last synced 2026-10-30).

| Path | Content |
|---|---|
| [`matched_baselines/`](matched_baselines) | **Latest.** All four-suite matched LIBERO results (protocol `lightwam_20261009_matched`, 6,000 episodes per model) in one table: the public FasterWAM and Light-WAM releases and 12 LoopWAM exports evaluated and verified 2026-10-27..30, plus every matched result reported on the other branches (Elastic, Residual Transport, ChronoLoop, init variants) |
| [`branch_reports/`](branch_reports) | Snapshot of every branch's report documents (Markdown under `plans/` that differs from `LoopWAM_NT`), with `index.json` giving the source branch and commit; rebuilt by `sync_branch_reports.py` |
| [`final/`](final), [`draft/`](draft) | Consolidated LoopWAM campaign report (audited metrics, per-episode records); built by `build_report.py` |
| [`campaign/`](campaign) | Campaign queue, per-condition and per-episode results |
| [`robustness/`](robustness) | LIBERO-Plus / LIBERO-Pro robustness evaluation |
