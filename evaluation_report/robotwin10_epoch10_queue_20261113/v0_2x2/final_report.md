# v0-video2-action2 epoch 10 — RoboTwin-10 — final evaluation

Completed 2026-11-13T10:06:17.113718+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 786/1000 | 78.60% |
| demo_randomized | 792/1000 | 79.20% |
| combined | 1578/2000 | 78.90% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 42 | 38 | 40.00% |
| move_stapler_pad | 60 | 67 | 63.50% |
| open_microwave | 55 | 61 | 58.00% |
| press_stapler | 90 | 92 | 91.00% |
| scan_object | 90 | 84 | 87.00% |
| stack_blocks_two | 99 | 97 | 98.00% |
| stamp_seal | 91 | 89 | 90.00% |
| turn_switch | 59 | 65 | 62.00% |

Controller elapsed time: 3.33 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0-video2-action2, 12 stored block pairs, video/action loops 2/2. All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

The epoch10 payload and native model variant are strictly validated. See setup for checkpoint provenance. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Comparison with previous local v0 4/1 epoch10

| Setting | v0 4/1 epoch10 | Current model epoch10 | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 80.00% | 78.60% | -1.40 |
| demo_randomized | 81.90% | 79.20% | -2.70 |
| combined | 80.95% | 78.90% | -2.05 |

This is an aggregate comparison under matched evaluation settings. Unseeded Python instruction sampling prevent claiming exactly paired trajectories. Training artifact hash differences are recorded in setup and source checks.

## November 15 report reconciliation

This result was freshly reconciled against its saved episode records, CSV exports and final audit digest. The [epoch-10 task comparison and detailed report](../../robotwin10_epoch10_results_20261115/README.md) includes all clean/randomized task counts and the separately labeled Dense-12 evaluations. Original scored records and audit files are unchanged.
