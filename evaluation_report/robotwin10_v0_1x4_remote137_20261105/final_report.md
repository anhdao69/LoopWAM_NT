# v0 1/4 — server137 RoboTwin-10 — final evaluation

Completed 2026-11-06T02:04:23.690771+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 739/1000 | 73.90% |
| demo_randomized | 724/1000 | 72.40% |
| combined | 1463/2000 | 73.15% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 38 | 35 | 36.50% |
| move_stapler_pad | 61 | 57 | 59.00% |
| open_microwave | 38 | 33 | 35.50% |
| press_stapler | 88 | 83 | 85.50% |
| scan_object | 84 | 78 | 81.00% |
| stack_blocks_two | 96 | 95 | 95.50% |
| stamp_seal | 78 | 84 | 81.00% |
| turn_switch | 56 | 60 | 58.00% |

Controller elapsed time: 3.53 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0 (12 stored block pairs; video/action core loops 1/4). All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

This v0 and the preceding dense-s30 were both trained on the same ten tasks for five epochs. Prior full-RoboTwin v0 checkpoints were trained on 50 tasks. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Same-task comparison with dense-s30

| Setting | Dense-s30 | v0 1/4 — server137 | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 83.80% | 73.90% | -9.90 |
| demo_randomized | 83.40% | 72.40% | -11.00 |
| combined | 83.60% | 73.15% | -10.45 |

Both releases use the same ten-task training dataset and five-epoch schedule. This compares aggregate SR under matched evaluation settings; instruction strings are not guaranteed paired because the original harness leaves Python template sampling unseeded.

## Full seed-42 model comparison

| Model | Clean | Randomized | Combined |
| --- | ---: | ---: | ---: |
| v0_1x4_remote137 | 73.90% | 72.40% | 73.15% |
| v0_2x2 | 74.50% | 72.10% | 73.30% |
| v0_4x4_remote189 | 78.40% | 78.00% | 78.20% |
| v0_4x2 | 80.40% | 79.50% | 79.95% |
| v0_4x4 | 77.40% | 80.30% | 78.85% |
| dense_s30 | 83.80% | 83.40% | 83.60% |
| v0_4x1_remote189 | 82.30% | 80.40% | 81.35% |
| v0_4x1_local | 82.10% | 79.60% | 80.85% |
| v0_4x4_remote137 | 76.60% | 79.30% | 77.95% |

All entries use seed42 for all ten tasks. No seed43 results or best-seed substitutions enter this comparison.

The earlier ten-worker attempt was invalidated for GPU/renderer errors and is excluded entirely. This is a fresh four-worker run with the same checkpoint, seed, NFE, tasks and episode counts. See [recovery incident](recovery_incident.json).