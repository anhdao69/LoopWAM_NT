# Dense-s12 — local RoboTwin-10 — final evaluation

Completed 2026-11-06T01:44:12.340759+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 719/1000 | 71.90% |
| demo_randomized | 725/1000 | 72.50% |
| combined | 1444/2000 | 72.20% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 41 | 27 | 34.00% |
| move_stapler_pad | 49 | 60 | 54.50% |
| open_microwave | 39 | 44 | 41.50% |
| press_stapler | 85 | 81 | 83.00% |
| scan_object | 78 | 77 | 77.50% |
| stack_blocks_two | 98 | 95 | 96.50% |
| stamp_seal | 84 | 84 | 84.00% |
| turn_switch | 45 | 58 | 51.50% |

Controller elapsed time: 3.20 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native dense-s12 (12 stored block pairs; one video/action pass, no recurrent loops). All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

This dense-s12 and the preceding dense-s30 were both trained on the same ten tasks for five epochs. Prior full-RoboTwin v0 checkpoints were trained on 50 tasks. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Same-task comparison with dense-s30

| Setting | Dense-s30 | Dense-s12 — local | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 83.80% | 71.90% | -11.90 |
| demo_randomized | 83.40% | 72.50% | -10.90 |
| combined | 83.60% | 72.20% | -11.40 |

Both releases use the same ten-task training dataset and five-epoch schedule. This compares aggregate SR under matched evaluation settings; instruction strings are not guaranteed paired because the original harness leaves Python template sampling unseeded.

## Full seed-42 model comparison

| Model | Clean | Randomized | Combined |
| --- | ---: | ---: | ---: |
| dense_s12 | 71.90% | 72.50% | 72.20% |
| v0_4x1_remote189 | 82.30% | 80.40% | 81.35% |
| v0_2x2 | 74.50% | 72.10% | 73.30% |
| v0_4x4_remote189 | 78.40% | 78.00% | 78.20% |
| v0_4x4_remote137 | 76.60% | 79.30% | 77.95% |
| v0_4x2 | 80.40% | 79.50% | 79.95% |
| v0_4x4 | 77.40% | 80.30% | 78.85% |
| dense_s30 | 83.80% | 83.40% | 83.60% |

All entries use seed42 for all ten tasks. No seed43 results or best-seed substitutions enter this comparison.

The earlier ten-worker attempt was invalidated for GPU/renderer errors and is excluded entirely. This is a fresh four-worker run with the same checkpoint, seed, NFE, tasks and episode counts. See [recovery incident](recovery_incident.json).