# LoopWAM dense-s12 RoboTwin-10 evaluation

Scored episodes: **2000/2,000**; successes: **1444**; completed task/settings: **20/20**.

Seed42;10tasks;clean+randomized;100acceptedepisodes/task/setting;unseen instructions;NFE10;predict32/execute24;dense-s12 checkpoint at epoch5. Smoke/calibration excluded. Partial SR is provisional.

| Task | Setting | Status | Episodes | Successes | SR |
| --- | --- | --- | --- | --- | --- |
| adjust_bottle | demo_clean | complete | 100/100 | 100 | 100.00% |
| adjust_bottle | demo_randomized | complete | 100/100 | 99 | 99.00% |
| grab_roller | demo_clean | complete | 100/100 | 100 | 100.00% |
| grab_roller | demo_randomized | complete | 100/100 | 100 | 100.00% |
| hanging_mug | demo_clean | complete | 100/100 | 41 | 41.00% |
| hanging_mug | demo_randomized | complete | 100/100 | 27 | 27.00% |
| move_stapler_pad | demo_clean | complete | 100/100 | 49 | 49.00% |
| move_stapler_pad | demo_randomized | complete | 100/100 | 60 | 60.00% |
| open_microwave | demo_clean | complete | 100/100 | 39 | 39.00% |
| open_microwave | demo_randomized | complete | 100/100 | 44 | 44.00% |
| press_stapler | demo_clean | complete | 100/100 | 85 | 85.00% |
| press_stapler | demo_randomized | complete | 100/100 | 81 | 81.00% |
| scan_object | demo_clean | complete | 100/100 | 78 | 78.00% |
| scan_object | demo_randomized | complete | 100/100 | 77 | 77.00% |
| stack_blocks_two | demo_clean | complete | 100/100 | 98 | 98.00% |
| stack_blocks_two | demo_randomized | complete | 100/100 | 95 | 95.00% |
| stamp_seal | demo_clean | complete | 100/100 | 84 | 84.00% |
| stamp_seal | demo_randomized | complete | 100/100 | 84 | 84.00% |
| turn_switch | demo_clean | complete | 100/100 | 45 | 45.00% |
| turn_switch | demo_randomized | complete | 100/100 | 58 | 58.00% |

[Detailed setup and comparability](setup_report.md) · [Progress JSON](progress.json)
Rendering uses RT32/depth8/OIDN2.3.3 (Blackwell compatibility upgrade). Input preprocessing and instruction generation follow released FastWAM; the same settings were used for the preceding local v0 evaluations.
