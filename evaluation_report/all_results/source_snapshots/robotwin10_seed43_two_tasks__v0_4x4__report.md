# LoopWAM v0-video4-action4 RoboTwin-10 two-task seed-43 evaluation

Scored episodes: **400/400**; successes: **217**; completed task/settings: **4/4**.

Seed43;2tasks;clean+randomized;100acceptedepisodes/task/setting;unseen instructions;NFE10;predict32/execute24;v0-video4-action4 checkpoint at epoch5. Smoke/calibration excluded. Partial SR is provisional.

| Task | Setting | Status | Episodes | Successes | SR |
| --- | --- | --- | --- | --- | --- |
| open_microwave | demo_clean | complete | 100/100 | 33 | 33.00% |
| open_microwave | demo_randomized | complete | 100/100 | 60 | 60.00% |
| move_stapler_pad | demo_clean | complete | 100/100 | 59 | 59.00% |
| move_stapler_pad | demo_randomized | complete | 100/100 | 65 | 65.00% |

[Detailed setup and comparability](setup_report.md) · [Progress JSON](progress.json)
Rendering uses RT32/depth8/OIDN2.3.3 (Blackwell compatibility upgrade). Input preprocessing and instruction generation follow released FastWAM; the same settings were used for the preceding local v0 evaluations.
