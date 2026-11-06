# v0 4/4 — server137 rerun RoboTwin-10 — final evaluation

Completed 2026-11-05T05:05:40.128625+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 766/1000 | 76.60% |
| demo_randomized | 793/1000 | 79.30% |
| combined | 1559/2000 | 77.95% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 48 | 51 | 49.50% |
| move_stapler_pad | 66 | 70 | 68.00% |
| open_microwave | 24 | 35 | 29.50% |
| press_stapler | 90 | 91 | 90.50% |
| scan_object | 86 | 82 | 84.00% |
| stack_blocks_two | 99 | 94 | 96.50% |
| stamp_seal | 89 | 96 | 92.50% |
| turn_switch | 64 | 75 | 69.50% |

Controller elapsed time: 3.60 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0 (12 stored block pairs; video/action core loops 4/4). All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

This v0 and the preceding dense-s30 were both trained on the same ten tasks for five epochs. Prior full-RoboTwin v0 checkpoints were trained on 50 tasks. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Same-task comparison with dense-s30

| Setting | Dense-s30 | v0 4/4 — server137 rerun | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 83.80% | 76.60% | -7.20 |
| demo_randomized | 83.40% | 79.30% | -4.10 |
| combined | 83.60% | 77.95% | -5.65 |

Both releases use the same ten-task training dataset and five-epoch schedule. This compares aggregate SR under matched evaluation settings; instruction strings are not guaranteed paired because the original harness leaves Python template sampling unseeded.

## Full seed-42 model comparison

| Model | Clean | Randomized | Combined |
| --- | ---: | ---: | ---: |
| v0_4x4_remote137_20261105 | 76.60% | 79.30% | 77.95% |
| v0_2x2 | 74.50% | 72.10% | 73.30% |
| v0_4x4_remote189 | 78.40% | 78.00% | 78.20% |
| v0_4x2 | 80.40% | 79.50% | 79.95% |
| v0_4x4 | 77.40% | 80.30% | 78.85% |
| dense_s30 | 83.80% | 83.40% | 83.60% |

All entries use seed42 for all ten tasks. No seed43 results or best-seed substitutions enter this comparison.