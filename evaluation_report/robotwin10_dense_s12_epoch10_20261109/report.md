# LoopWAM Dense-S12 RoboTwin-10 — local RTX PRO 6000

Scored episodes: **2000/2,000**; successes: **1517**; completed task/settings: **20/20**.

Seed42;10tasks;clean+randomized;100acceptedepisodes/task/setting;unseen instructions;NFE10;predict32/execute24;dense-s12 checkpoint at epoch10. Smoke/calibration excluded. Partial SR is provisional.

| Task | Setting | Status | Episodes | Successes | SR |
| --- | --- | --- | --- | --- | --- |
| adjust_bottle | demo_clean | complete | 100/100 | 100 | 100.00% |
| adjust_bottle | demo_randomized | complete | 100/100 | 99 | 99.00% |
| grab_roller | demo_clean | complete | 100/100 | 100 | 100.00% |
| grab_roller | demo_randomized | complete | 100/100 | 100 | 100.00% |
| hanging_mug | demo_clean | complete | 100/100 | 45 | 45.00% |
| hanging_mug | demo_randomized | complete | 100/100 | 43 | 43.00% |
| move_stapler_pad | demo_clean | complete | 100/100 | 53 | 53.00% |
| move_stapler_pad | demo_randomized | complete | 100/100 | 58 | 58.00% |
| open_microwave | demo_clean | complete | 100/100 | 58 | 58.00% |
| open_microwave | demo_randomized | complete | 100/100 | 51 | 51.00% |
| press_stapler | demo_clean | complete | 100/100 | 91 | 91.00% |
| press_stapler | demo_randomized | complete | 100/100 | 90 | 90.00% |
| scan_object | demo_clean | complete | 100/100 | 72 | 72.00% |
| scan_object | demo_randomized | complete | 100/100 | 77 | 77.00% |
| stack_blocks_two | demo_clean | complete | 100/100 | 97 | 97.00% |
| stack_blocks_two | demo_randomized | complete | 100/100 | 97 | 97.00% |
| stamp_seal | demo_clean | complete | 100/100 | 86 | 86.00% |
| stamp_seal | demo_randomized | complete | 100/100 | 86 | 86.00% |
| turn_switch | demo_clean | complete | 100/100 | 52 | 52.00% |
| turn_switch | demo_randomized | complete | 100/100 | 62 | 62.00% |

[Detailed setup and comparability](setup_report.md) · [Progress JSON](progress.json)
Rendering uses RT32/depth8/OIDN2.3.3 (Blackwell compatibility upgrade). Input preprocessing and instruction generation follow released FastWAM; the same settings were used for the preceding local v0 evaluations.
