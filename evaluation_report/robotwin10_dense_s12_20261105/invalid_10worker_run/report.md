# LoopWAM dense-s12 RoboTwin-10 evaluation

Scored episodes: **824/2,000**; successes: **615**; completed task/settings: **5/20**.

Seed42;10tasks;clean+randomized;100acceptedepisodes/task/setting;unseen instructions;NFE10;predict32/execute24;dense-s12 checkpoint at epoch5. Smoke/calibration excluded. Partial SR is provisional.

| Task | Setting | Status | Episodes | Successes | SR |
| --- | --- | --- | --- | --- | --- |
| adjust_bottle | demo_clean | running | 50/100 | 50 | 100.00% |
| adjust_bottle | demo_randomized | failed | 0/100 | 0 | — |
| grab_roller | demo_clean | complete | 100/100 | 100 | 100.00% |
| grab_roller | demo_randomized | failed | 0/100 | 0 | — |
| hanging_mug | demo_clean | pending | 0/100 | 0 | — |
| hanging_mug | demo_randomized | running | 44/100 | 14 | 31.82% |
| move_stapler_pad | demo_clean | complete | 100/100 | 49 | 49.00% |
| move_stapler_pad | demo_randomized | complete | 100/100 | 57 | 57.00% |
| open_microwave | demo_clean | running | 85/100 | 39 | 45.88% |
| open_microwave | demo_randomized | running | 0/100 | 0 | — |
| press_stapler | demo_clean | pending | 0/100 | 0 | — |
| press_stapler | demo_randomized | running | 0/100 | 0 | — |
| scan_object | demo_clean | running | 11/100 | 9 | 81.82% |
| scan_object | demo_randomized | running | 72/100 | 54 | 75.00% |
| stack_blocks_two | demo_clean | complete | 100/100 | 99 | 99.00% |
| stack_blocks_two | demo_randomized | running | 62/100 | 59 | 95.16% |
| stamp_seal | demo_clean | complete | 100/100 | 85 | 85.00% |
| stamp_seal | demo_randomized | running | 0/100 | 0 | — |
| turn_switch | demo_clean | running | 0/100 | 0 | — |
| turn_switch | demo_randomized | pending | 0/100 | 0 | — |

[Detailed setup and comparability](setup_report.md) · [Progress JSON](progress.json)
Rendering uses RT32/depth8/OIDN2.3.3 (Blackwell compatibility upgrade). Input preprocessing and instruction generation follow released FastWAM; the same settings were used for the preceding local v0 evaluations.
