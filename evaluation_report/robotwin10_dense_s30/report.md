<!-- results-reconciled-20261107 -->
> **Complete:** 1672/2000 = **83.60%**. Clean **83.80%**, randomized **83.40%**. Saved final audit passed; outcome arithmetic and COMPLETE digest rechecked. Updated 2026-11-07T02:51:27+07:00.
> [Final detailed report](final_report.md).

# LoopWAM dense-s30 RoboTwin-10 evaluation

Scored episodes: **2000/2,000**; successes: **1672**; completed task/settings: **20/20**.

Seed42;10tasks;clean+randomized;100acceptedepisodes/task/setting;unseen instructions;NFE10;predict32/execute24;dense-s30 one-pass checkpoint at epoch5. Smoke/calibration excluded. These results are complete.

| Task | Setting | Status | Episodes | Successes | SR |
| --- | --- | --- | --- | --- | --- |
| adjust_bottle | demo_clean | complete | 100/100 | 100 | 100.00% |
| adjust_bottle | demo_randomized | complete | 100/100 | 99 | 99.00% |
| grab_roller | demo_clean | complete | 100/100 | 100 | 100.00% |
| grab_roller | demo_randomized | complete | 100/100 | 99 | 99.00% |
| hanging_mug | demo_clean | complete | 100/100 | 63 | 63.00% |
| hanging_mug | demo_randomized | complete | 100/100 | 56 | 56.00% |
| move_stapler_pad | demo_clean | complete | 100/100 | 83 | 83.00% |
| move_stapler_pad | demo_randomized | complete | 100/100 | 77 | 77.00% |
| open_microwave | demo_clean | complete | 100/100 | 61 | 61.00% |
| open_microwave | demo_randomized | complete | 100/100 | 50 | 50.00% |
| press_stapler | demo_clean | complete | 100/100 | 90 | 90.00% |
| press_stapler | demo_randomized | complete | 100/100 | 93 | 93.00% |
| scan_object | demo_clean | complete | 100/100 | 88 | 88.00% |
| scan_object | demo_randomized | complete | 100/100 | 94 | 94.00% |
| stack_blocks_two | demo_clean | complete | 100/100 | 100 | 100.00% |
| stack_blocks_two | demo_randomized | complete | 100/100 | 100 | 100.00% |
| stamp_seal | demo_clean | complete | 100/100 | 91 | 91.00% |
| stamp_seal | demo_randomized | complete | 100/100 | 96 | 96.00% |
| turn_switch | demo_clean | complete | 100/100 | 62 | 62.00% |
| turn_switch | demo_randomized | complete | 100/100 | 70 | 70.00% |

[Detailed setup and comparability](setup_report.md) · [Progress JSON](progress.json)
Rendering uses RT32/depth8/OIDN2.3.3 (Blackwell compatibility upgrade). Input preprocessing and instruction generation follow released FastWAM; the same settings were used for the preceding local v0 evaluations.
