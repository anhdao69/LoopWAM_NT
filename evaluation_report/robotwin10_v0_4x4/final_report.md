# v0 4/4 RoboTwin-10 — final evaluation

Completed 2026-11-03T19:48:25.626663+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 774/1000 | 77.40% |
| demo_randomized | 803/1000 | 80.30% |
| combined | 1577/2000 | 78.85% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 50 | 52 | 51.00% |
| move_stapler_pad | 65 | 71 | 68.00% |
| open_microwave | 26 | 37 | 31.50% |
| press_stapler | 90 | 92 | 91.00% |
| scan_object | 88 | 87 | 87.50% |
| stack_blocks_two | 100 | 98 | 99.00% |
| stamp_seal | 92 | 95 | 93.50% |
| turn_switch | 63 | 72 | 67.50% |

Controller elapsed time: 5.93 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0 (12 stored block pairs; video/action core loops 4/4). All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

This v0 and the preceding dense-s30 were both trained on the same ten tasks for five epochs. Prior full-RoboTwin v0 checkpoints were trained on 50 tasks. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Same-task comparison with dense-s30

| Setting | Dense-s30 | v0 4/4 | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 83.80% | 77.40% | -6.40 |
| demo_randomized | 83.40% | 80.30% | -3.10 |
| combined | 83.60% | 78.85% | -4.75 |

Both releases use the same ten-task training dataset and five-epoch schedule. This compares aggregate SR under matched evaluation settings; instruction strings are not guaranteed paired because the original harness leaves Python template sampling unseeded.