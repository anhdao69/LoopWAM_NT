# v0 4/2 RoboTwin-10 — final evaluation

Completed 2026-11-04T04:05:11.786470+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 804/1000 | 80.40% |
| demo_randomized | 795/1000 | 79.50% |
| combined | 1599/2000 | 79.95% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 62 | 55 | 58.50% |
| move_stapler_pad | 66 | 61 | 63.50% |
| open_microwave | 58 | 49 | 53.50% |
| press_stapler | 90 | 95 | 92.50% |
| scan_object | 85 | 82 | 83.50% |
| stack_blocks_two | 97 | 96 | 96.50% |
| stamp_seal | 92 | 92 | 92.00% |
| turn_switch | 54 | 66 | 60.00% |

Controller elapsed time: 3.21 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0 (12 stored block pairs; video/action core loops 4/2). All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

This v0 and the preceding dense-s30 were both trained on the same ten tasks for five epochs. Prior full-RoboTwin v0 checkpoints were trained on 50 tasks. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Same-task comparison with dense-s30

| Setting | Dense-s30 | v0 4/2 | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 83.80% | 80.40% | -3.40 |
| demo_randomized | 83.40% | 79.50% | -3.90 |
| combined | 83.60% | 79.95% | -3.65 |

Both releases use the same ten-task training dataset and five-epoch schedule. This compares aggregate SR under matched evaluation settings; instruction strings are not guaranteed paired because the original harness leaves Python template sampling unseeded.

## Full seed-42 model comparison

| Model | Clean | Randomized | Combined |
| --- | ---: | ---: | ---: |
| v0_4x2 | 80.40% | 79.50% | 79.95% |
| v0_4x4 | 77.40% | 80.30% | 78.85% |
| dense_s30 | 83.80% | 83.40% | 83.60% |

All entries use seed42 for all ten tasks. No seed43 results or best-seed substitutions enter this comparison.