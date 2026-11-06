# v0-video4-action2 H100 retrain — RoboTwin-10 — final evaluation

Completed 2026-11-07T01:43:35.873662+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 801/1000 | 80.10% |
| demo_randomized | 815/1000 | 81.50% |
| combined | 1616/2000 | 80.80% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 100 | 100.00% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 55 | 55 | 55.00% |
| move_stapler_pad | 72 | 63 | 67.50% |
| open_microwave | 62 | 60 | 61.00% |
| press_stapler | 92 | 90 | 91.00% |
| scan_object | 85 | 88 | 86.50% |
| stack_blocks_two | 96 | 97 | 96.50% |
| stamp_seal | 92 | 93 | 92.50% |
| turn_switch | 47 | 69 | 58.00% |

Controller elapsed time: 5.24 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0 with video/action loops 4/2. All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

This is an independent H100 retrain. Donor-artifact and training data-manifest hashes differ from earlier releases; see setup for evidence. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Comparison with original robotwin-10 release

| Setting | Original | H100 retrain | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 80.40% | 80.10% | -0.30 |
| demo_randomized | 79.50% | 81.50% | +2.00 |
| combined | 79.95% | 80.80% | +0.85 |

This is an aggregate comparison under matched evaluation settings. Unseeded Python instruction sampling and different GPU hosts prevent claiming exactly paired trajectories. Training artifact hash differences are recorded in setup and source checks.