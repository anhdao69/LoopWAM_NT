# v0-video4-action2 epoch 10 — RoboTwin-10 — final evaluation

Completed 2026-11-13T06:41:50.228866+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 825/1000 | 82.50% |
| demo_randomized | 823/1000 | 82.30% |
| combined | 1648/2000 | 82.40% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 100 | 100.00% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 54 | 48 | 51.00% |
| move_stapler_pad | 69 | 70 | 69.50% |
| open_microwave | 80 | 75 | 77.50% |
| press_stapler | 81 | 89 | 85.00% |
| scan_object | 88 | 83 | 85.50% |
| stack_blocks_two | 97 | 97 | 97.00% |
| stamp_seal | 91 | 92 | 91.50% |
| turn_switch | 65 | 69 | 67.00% |

Controller elapsed time: 3.36 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0-video4-action2, 12 stored block pairs, video/action loops 4/2. All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

The epoch10 payload and native model variant are strictly validated. See setup for checkpoint provenance. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Comparison with previous local v0 4/1 epoch10

| Setting | v0 4/1 epoch10 | Current model epoch10 | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 80.00% | 82.50% | +2.50 |
| demo_randomized | 81.90% | 82.30% | +0.40 |
| combined | 80.95% | 82.40% | +1.45 |

This is an aggregate comparison under matched evaluation settings. Unseeded Python instruction sampling prevent claiming exactly paired trajectories. Training artifact hash differences are recorded in setup and source checks.

## November 15 report reconciliation

This result was freshly reconciled against its saved episode records, CSV exports and final audit digest. The [epoch-10 task comparison and detailed report](../../robotwin10_epoch10_results_20261115/README.md) includes all clean/randomized task counts and the separately labeled Dense-12 evaluations. Original scored records and audit files are unchanged.
