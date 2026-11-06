# v0-video4-action4 H100 retrain — RoboTwin-10 — final evaluation

Completed 2026-11-07T00:00:08.341366+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 799/1000 | 79.90% |
| demo_randomized | 774/1000 | 77.40% |
| combined | 1573/2000 | 78.65% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 45 | 42 | 43.50% |
| move_stapler_pad | 72 | 62 | 67.00% |
| open_microwave | 57 | 47 | 52.00% |
| press_stapler | 90 | 95 | 92.50% |
| scan_object | 88 | 82 | 85.00% |
| stack_blocks_two | 97 | 92 | 94.50% |
| stamp_seal | 90 | 86 | 88.00% |
| turn_switch | 60 | 69 | 64.50% |

Controller elapsed time: 3.60 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0 with video/action loops 4/4. All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

This is an independent H100 retrain. Donor-artifact and training data-manifest hashes differ from earlier releases; see setup for evidence. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Comparison with original robotwin-10 release

| Setting | Original | H100 retrain | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 77.40% | 79.90% | +2.50 |
| demo_randomized | 80.30% | 77.40% | -2.90 |
| combined | 78.85% | 78.65% | -0.20 |

This is an aggregate comparison under matched evaluation settings. Unseeded Python instruction sampling and different GPU hosts prevent claiming exactly paired trajectories. Training artifact hash differences are recorded in setup and source checks.