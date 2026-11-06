<!-- results-reconciled-20261107 -->
> **Complete:** 274/400 = **68.50%**. Clean **73.50%**, randomized **63.50%**. Saved final audit passed; outcome arithmetic and COMPLETE digest rechecked. Updated 2026-11-07T02:51:27+07:00.
> [Final detailed report](final_report.md).

# LoopWAM dense-s30 RoboTwin-10 two-task seed-43 evaluation

Scored episodes: **400/400**; successes: **274**; completed task/settings: **4/4**.

Seed43;2tasks;clean+randomized;100acceptedepisodes/task/setting;unseen instructions;NFE10;predict32/execute24;dense-s30 one-pass checkpoint at epoch5. Smoke/calibration excluded. These results are complete.

| Task | Setting | Status | Episodes | Successes | SR |
| --- | --- | --- | --- | --- | --- |
| open_microwave | demo_clean | complete | 100/100 | 75 | 75.00% |
| open_microwave | demo_randomized | complete | 100/100 | 53 | 53.00% |
| move_stapler_pad | demo_clean | complete | 100/100 | 72 | 72.00% |
| move_stapler_pad | demo_randomized | complete | 100/100 | 74 | 74.00% |

[Detailed setup and comparability](setup_report.md) · [Progress JSON](progress.json)
Rendering uses RT32/depth8/OIDN2.3.3 (Blackwell compatibility upgrade). Input preprocessing and instruction generation follow released FastWAM; the same settings were used for the preceding local v0 evaluations.
