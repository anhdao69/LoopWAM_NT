# LoopWAM v0 1/4 RoboTwin-10 — server 10.148.182.137

Scored episodes: **239/2,000**; successes: **153**; completed task/settings: **1/20**.

Seed42;10tasks;clean+randomized;100acceptedepisodes/task/setting;unseen instructions;NFE10;predict32/execute24;v0-video1-action4 checkpoint at epoch5. Smoke/calibration excluded. Partial SR is provisional.

| Task | Setting | Status | Episodes | Successes | SR |
| --- | --- | --- | --- | --- | --- |
| adjust_bottle | demo_clean | failed | 0/100 | 0 | — |
| adjust_bottle | demo_randomized | failed | 0/100 | 0 | — |
| grab_roller | demo_clean | failed | 0/100 | 0 | — |
| grab_roller | demo_randomized | failed | 0/100 | 0 | — |
| hanging_mug | demo_clean | running | 25/100 | 16 | 64.00% |
| hanging_mug | demo_randomized | running | 0/100 | 0 | — |
| move_stapler_pad | demo_clean | failed | 0/100 | 0 | — |
| move_stapler_pad | demo_randomized | complete | 100/100 | 58 | 58.00% |
| open_microwave | demo_clean | running | 43/100 | 17 | 39.53% |
| open_microwave | demo_randomized | failed | 0/100 | 0 | — |
| press_stapler | demo_clean | failed | 0/100 | 0 | — |
| press_stapler | demo_randomized | failed | 0/100 | 0 | — |
| scan_object | demo_clean | running | 12/100 | 10 | 83.33% |
| scan_object | demo_randomized | failed | 0/100 | 0 | — |
| stack_blocks_two | demo_clean | failed | 0/100 | 0 | — |
| stack_blocks_two | demo_randomized | running | 17/100 | 17 | 100.00% |
| stamp_seal | demo_clean | running | 42/100 | 35 | 83.33% |
| stamp_seal | demo_randomized | running | 0/100 | 0 | — |
| turn_switch | demo_clean | failed | 0/100 | 0 | — |
| turn_switch | demo_randomized | failed | 0/100 | 0 | — |

[Detailed setup and comparability](setup_report.md) · [Progress JSON](progress.json)
Rendering uses RT32/depth8/OIDN2.3.3 (Blackwell compatibility upgrade). Input preprocessing and instruction generation follow released FastWAM; the same settings were used for the preceding local v0 evaluations.
