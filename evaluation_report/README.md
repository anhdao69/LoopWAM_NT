# Evaluation reports

The shared report snapshot was synchronized across all repository branches on 2026-11-03; this `evaluation_report/` tree is identical on every branch.

| Path | Content |
|---|---|
| [`robotwin_v0/report.md`](robotwin_v0/report.md) | **Completed 2026-10-31.** LoopWAM v0 RoboTwin, all 50 tasks, clean + randomized, seed 42, 10,000 episodes: 91.60% clean / 91.86% randomized / 91.73% combined; full artifact audit, per-task and per-episode exports, retry history |
| [`matched_pro6000_20261101/`](matched_pro6000_20261101) | **Completed 2026-11-02.** 2026-11-01 batch on the 2x RTX PRO 6000 host, same matched protocol (6,000 episodes each): v0 4/2 94.53; E0 at K_a 3/5/6 (NFE 10) 94.15 / 95.03 / 94.78; base v0 4/4 without distillation at NFE 2 [4,4] 95.78, NFE 5 [4]x5 95.73 and [1,1] 11.92; project-status summary, contrasts, gates, per-episode exports |
| [`matched_baselines/`](matched_baselines) | **Latest.** All four-suite matched LIBERO results (protocol `lightwam_20261009_matched`, 6,000 episodes per model) in one table: the public FasterWAM and Light-WAM releases and 12 LoopWAM exports evaluated and verified 2026-10-27..30, plus every matched result reported on the other branches (Elastic, Residual Transport, ChronoLoop, init variants) and the seven conditions of the 2026-11-01 batch |
| [`branch_reports/`](branch_reports) | Snapshot of every branch's report documents (Markdown under `plans/` that differs from `LoopWAM_NT`), with `index.json` giving the source branch and commit; rebuilt by `sync_branch_reports.py` |
| [`final/`](final), [`draft/`](draft) | Consolidated LoopWAM campaign report (audited metrics, per-episode records); built by `build_report.py` |
| [`campaign/`](campaign) | Campaign queue, per-condition and per-episode results |
| [`robustness/`](robustness) | LIBERO-Plus / LIBERO-Pro robustness evaluation |
