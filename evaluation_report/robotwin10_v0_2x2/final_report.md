# v0 2/2 RoboTwin-10 — final evaluation

Completed 2026-11-04T08:13:37.078682+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 745/1000 | 74.50% |
| demo_randomized | 721/1000 | 72.10% |
| combined | 1466/2000 | 73.30% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 38 | 19 | 28.50% |
| move_stapler_pad | 54 | 62 | 58.00% |
| open_microwave | 47 | 40 | 43.50% |
| press_stapler | 88 | 92 | 90.00% |
| scan_object | 86 | 77 | 81.50% |
| stack_blocks_two | 94 | 91 | 92.50% |
| stamp_seal | 88 | 86 | 87.00% |
| turn_switch | 50 | 55 | 52.50% |

Controller elapsed time: 3.35 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0 (12 stored block pairs; video/action core loops 2/2). All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

This v0 and the preceding dense-s30 were both trained on the same ten tasks for five epochs. Prior full-RoboTwin v0 checkpoints were trained on 50 tasks. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Same-task comparison with dense-s30

| Setting | Dense-s30 | v0 2/2 | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 83.80% | 74.50% | -9.30 |
| demo_randomized | 83.40% | 72.10% | -11.30 |
| combined | 83.60% | 73.30% | -10.30 |

Both releases use the same ten-task training dataset and five-epoch schedule. This compares aggregate SR under matched evaluation settings; instruction strings are not guaranteed paired because the original harness leaves Python template sampling unseeded.

## Full seed-42 model comparison

| Model | Clean | Randomized | Combined |
| --- | ---: | ---: | ---: |
| v0_2x2 | 74.50% | 72.10% | 73.30% |
| v0_4x2 | 80.40% | 79.50% | 79.95% |
| v0_4x4 | 77.40% | 80.30% | 78.85% |
| dense_s30 | 83.80% | 83.40% | 83.60% |

All entries use seed42 for all ten tasks. No seed43 results or best-seed substitutions enter this comparison.