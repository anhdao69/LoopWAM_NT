# Dense-S30 RoboTwin-10 — final evaluation

Completed 2026-11-03T02:29:10.459957+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 838/1000 | 83.80% |
| demo_randomized | 834/1000 | 83.40% |
| combined | 1672/2000 | 83.60% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| grab_roller | 100 | 99 | 99.50% |
| hanging_mug | 63 | 56 | 59.50% |
| move_stapler_pad | 83 | 77 | 80.00% |
| open_microwave | 61 | 50 | 55.50% |
| press_stapler | 90 | 93 | 91.50% |
| scan_object | 88 | 94 | 91.00% |
| stack_blocks_two | 100 | 100 | 100.00% |
| stamp_seal | 91 | 96 | 93.50% |
| turn_switch | 62 | 70 | 66.00% |

Controller elapsed time: 4.90 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native dense-s30 (30 block pairs, one pass). Sixteen smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

Dense-S30 was trained on these ten tasks; prior v0 models were trained on all 50. Equal evaluation settings do not imply equal training data. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).
