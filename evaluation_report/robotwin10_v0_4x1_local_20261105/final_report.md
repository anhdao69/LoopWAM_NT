# v0 4/1 — local rerun RoboTwin-10 — final evaluation

Completed 2026-11-05T17:30:25.648309+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 821/1000 | 82.10% |
| demo_randomized | 796/1000 | 79.60% |
| combined | 1617/2000 | 80.85% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 98 | 99.00% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 51 | 46 | 48.50% |
| move_stapler_pad | 75 | 59 | 67.00% |
| open_microwave | 78 | 63 | 70.50% |
| press_stapler | 90 | 92 | 91.00% |
| scan_object | 83 | 81 | 82.00% |
| stack_blocks_two | 98 | 98 | 98.00% |
| stamp_seal | 87 | 95 | 91.00% |
| turn_switch | 59 | 64 | 61.50% |

Controller elapsed time: 5.71 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0 (12 stored block pairs; video/action core loops 4/1). All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

This v0 and the preceding dense-s30 were both trained on the same ten tasks for five epochs. Prior full-RoboTwin v0 checkpoints were trained on 50 tasks. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Same-task comparison with dense-s30

| Setting | Dense-s30 | v0 4/1 — local rerun | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 83.80% | 82.10% | -1.70 |
| demo_randomized | 83.40% | 79.60% | -3.80 |
| combined | 83.60% | 80.85% | -2.75 |

Both releases use the same ten-task training dataset and five-epoch schedule. This compares aggregate SR under matched evaluation settings; instruction strings are not guaranteed paired because the original harness leaves Python template sampling unseeded.

## Full seed-42 model comparison

| Model | Clean | Randomized | Combined |
| --- | ---: | ---: | ---: |
| v0_4x1_local | 82.10% | 79.60% | 80.85% |
| v0_4x1_remote189 | 82.30% | 80.40% | 81.35% |
| v0_2x2 | 74.50% | 72.10% | 73.30% |
| v0_4x4_remote189 | 78.40% | 78.00% | 78.20% |
| v0_4x4_remote137 | 76.60% | 79.30% | 77.95% |
| v0_4x2 | 80.40% | 79.50% | 79.95% |
| v0_4x4 | 77.40% | 80.30% | 78.85% |
| dense_s30 | 83.80% | 83.40% | 83.60% |

All entries use seed42 for all ten tasks. No seed43 results or best-seed substitutions enter this comparison.