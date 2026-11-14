# LoopWAM dense-s12 RoboTwin-10 — local RTX PRO 6000

Scored episodes: **638/2,000**; successes: **519**; completed task/settings: **4/20**.

Seed42;10tasks;clean+randomized;100acceptedepisodes/task/setting;unseen instructions;NFE10;predict32/execute24;dense-s12 checkpoint at epoch10. Smoke/calibration excluded. Partial SR is provisional.

| Task | Setting | Status | Episodes | Successes | SR |
| --- | --- | --- | --- | --- | --- |
| adjust_bottle | demo_clean | complete | 100/100 | 100 | 100.00% |
| adjust_bottle | demo_randomized | complete | 100/100 | 99 | 99.00% |
| grab_roller | demo_clean | complete | 100/100 | 100 | 100.00% |
| grab_roller | demo_randomized | complete | 100/100 | 100 | 100.00% |
| hanging_mug | demo_clean | running | 59/100 | 29 | 49.15% |
| hanging_mug | demo_randomized | running | 32/100 | 6 | 18.75% |
| move_stapler_pad | demo_clean | running | 84/100 | 49 | 58.33% |
| move_stapler_pad | demo_randomized | running | 63/100 | 36 | 57.14% |
| open_microwave | demo_clean | pending | 0/100 | 0 | — |
| open_microwave | demo_randomized | pending | 0/100 | 0 | — |
| press_stapler | demo_clean | pending | 0/100 | 0 | — |
| press_stapler | demo_randomized | pending | 0/100 | 0 | — |
| scan_object | demo_clean | pending | 0/100 | 0 | — |
| scan_object | demo_randomized | pending | 0/100 | 0 | — |
| stack_blocks_two | demo_clean | pending | 0/100 | 0 | — |
| stack_blocks_two | demo_randomized | pending | 0/100 | 0 | — |
| stamp_seal | demo_clean | pending | 0/100 | 0 | — |
| stamp_seal | demo_randomized | pending | 0/100 | 0 | — |
| turn_switch | demo_clean | pending | 0/100 | 0 | — |
| turn_switch | demo_randomized | pending | 0/100 | 0 | — |

[Detailed setup and comparability](setup_report.md) · [Progress JSON](progress.json)
Rendering uses RT32/depth8/OIDN2.3.3 (Blackwell compatibility upgrade). Input preprocessing and instruction generation follow released FastWAM; the same settings were used for the preceding local v0 evaluations.
