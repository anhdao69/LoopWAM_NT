# v0 4/4 RoboTwin-10 — server 10.148.183.189, final evaluation

Completed 2026-11-04T07:31:18.027768+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 784/1000 | 78.40% |
| demo_randomized | 780/1000 | 78.00% |
| combined | 1564/2000 | 78.20% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 48 | 45 | 46.50% |
| move_stapler_pad | 70 | 69 | 69.50% |
| open_microwave | 32 | 35 | 33.50% |
| press_stapler | 91 | 93 | 92.00% |
| scan_object | 86 | 79 | 82.50% |
| stack_blocks_two | 98 | 96 | 97.00% |
| stamp_seal | 94 | 94 | 94.00% |
| turn_switch | 65 | 70 | 67.50% |

Controller elapsed time: 5.34 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0 (12 stored block pairs; video/action core loops 4/4). All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

This v0 and the preceding dense-s30 were both trained on the same ten tasks for five epochs. Prior full-RoboTwin v0 checkpoints were trained on 50 tasks. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Same-task comparison with dense-s30

| Setting | Dense-s30 | v0 4/4 | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 83.80% | 78.40% | -5.40 |
| demo_randomized | 83.40% | 78.00% | -5.40 |
| combined | 83.60% | 78.20% | -5.40 |

Both releases use the same ten-task training dataset and five-epoch schedule. This compares aggregate SR under matched evaluation settings; instruction strings are not guaranteed paired because the original harness leaves Python template sampling unseeded.

## Local versus remote v0 4/4, full seed42

| Setting | Original server | 10.148.183.189 |
| --- | ---: | ---: |
| demo_clean | 77.40% | 78.40% |
| demo_randomized | 80.30% | 78.00% |
| combined | 78.85% | 78.20% |

The checkpoint, model/harness source, data artifacts and scored settings match. Absolute asset paths were relocated; GPU UUID and driver are recorded separately. Original unseeded instruction sampling prevents assuming identical episode instructions between runs.