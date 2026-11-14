# v0-video4-action1 epoch 10 — RoboTwin-10 — final evaluation

Completed 2026-11-09T04:01:58.505703+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 800/1000 | 80.00% |
| demo_randomized | 819/1000 | 81.90% |
| combined | 1619/2000 | 80.95% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 99 | 99 | 99.00% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 51 | 39 | 45.00% |
| move_stapler_pad | 78 | 73 | 75.50% |
| open_microwave | 71 | 70 | 70.50% |
| press_stapler | 83 | 86 | 84.50% |
| scan_object | 81 | 85 | 83.00% |
| stack_blocks_two | 99 | 99 | 99.00% |
| stamp_seal | 83 | 95 | 89.00% |
| turn_switch | 55 | 73 | 64.00% |

Controller elapsed time: 3.07 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 12; logged partial episodes excluded from scoring: 486.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0 with video/action loops 4/1. All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

This is epoch 10 continued from epoch 5; inference model and normalization are unchanged. See setup for training-contract differences. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Comparison with original robotwin-10 release

| Setting | Epoch 5 (same server) | epoch 10 | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 82.10% | 80.00% | -2.10 |
| demo_randomized | 79.60% | 81.90% | +2.30 |
| combined | 80.85% | 80.95% | +0.10 |

This is an aggregate comparison under matched evaluation settings. Unseeded Python instruction sampling prevent claiming exactly paired trajectories. Training artifact hash differences are recorded in setup and source checks.