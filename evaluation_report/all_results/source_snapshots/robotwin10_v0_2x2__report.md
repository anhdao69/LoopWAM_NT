# LoopWAM v0-video2-action2 RoboTwin-10 evaluation

Scored episodes: **2000/2,000**; successes: **1466**; completed task/settings: **20/20**.

Seed42;10tasks;clean+randomized;100acceptedepisodes/task/setting;unseen instructions;NFE10;predict32/execute24;v0-video2-action2 checkpoint at epoch5. Smoke/calibration excluded. Partial SR is provisional.

| Task | Setting | Status | Episodes | Successes | SR |
| --- | --- | --- | --- | --- | --- |
| adjust_bottle | demo_clean | complete | 100/100 | 100 | 100.00% |
| adjust_bottle | demo_randomized | complete | 100/100 | 99 | 99.00% |
| grab_roller | demo_clean | complete | 100/100 | 100 | 100.00% |
| grab_roller | demo_randomized | complete | 100/100 | 100 | 100.00% |
| hanging_mug | demo_clean | complete | 100/100 | 38 | 38.00% |
| hanging_mug | demo_randomized | complete | 100/100 | 19 | 19.00% |
| move_stapler_pad | demo_clean | complete | 100/100 | 54 | 54.00% |
| move_stapler_pad | demo_randomized | complete | 100/100 | 62 | 62.00% |
| open_microwave | demo_clean | complete | 100/100 | 47 | 47.00% |
| open_microwave | demo_randomized | complete | 100/100 | 40 | 40.00% |
| press_stapler | demo_clean | complete | 100/100 | 88 | 88.00% |
| press_stapler | demo_randomized | complete | 100/100 | 92 | 92.00% |
| scan_object | demo_clean | complete | 100/100 | 86 | 86.00% |
| scan_object | demo_randomized | complete | 100/100 | 77 | 77.00% |
| stack_blocks_two | demo_clean | complete | 100/100 | 94 | 94.00% |
| stack_blocks_two | demo_randomized | complete | 100/100 | 91 | 91.00% |
| stamp_seal | demo_clean | complete | 100/100 | 88 | 88.00% |
| stamp_seal | demo_randomized | complete | 100/100 | 86 | 86.00% |
| turn_switch | demo_clean | complete | 100/100 | 50 | 50.00% |
| turn_switch | demo_randomized | complete | 100/100 | 55 | 55.00% |

[Detailed setup and comparability](setup_report.md) · [Progress JSON](progress.json)
Rendering uses RT32/depth8/OIDN2.3.3 (Blackwell compatibility upgrade). Input preprocessing and instruction generation follow released FastWAM; the same settings were used for the preceding local v0 evaluations.
