# dense-s12 epoch 10 — RoboTwin-10 — final evaluation

Completed 2026-11-09T10:20:56.272777+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 754/1000 | 75.40% |
| demo_randomized | 763/1000 | 76.30% |
| combined | 1517/2000 | 75.85% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 45 | 43 | 44.00% |
| move_stapler_pad | 53 | 58 | 55.50% |
| open_microwave | 58 | 51 | 54.50% |
| press_stapler | 91 | 90 | 90.50% |
| scan_object | 72 | 77 | 74.50% |
| stack_blocks_two | 97 | 97 | 97.00% |
| stamp_seal | 86 | 86 | 86.00% |
| turn_switch | 52 | 62 | 57.00% |

Controller elapsed time: 3.26 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native Dense-S12 with12storedblockpairs and video/action passes1/1. All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

This is epoch 10 continued from epoch 5; inference model and normalization are unchanged. See setup for training-contract differences. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Comparison with original robotwin-10 release

| Setting | Epoch 5 (same server) | epoch 10 | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 71.90% | 75.40% | +3.50 |
| demo_randomized | 72.50% | 76.30% | +3.80 |
| combined | 72.20% | 75.85% | +3.65 |

This is an aggregate comparison under matched evaluation settings. Unseeded Python instruction sampling prevent claiming exactly paired trajectories. Training artifact hash differences are recorded in setup and source checks.

## November 15 report reconciliation

This result was freshly reconciled against its saved episode records, CSV exports and final audit digest. The [epoch-10 task comparison and detailed report](../robotwin10_epoch10_results_20261115/README.md) includes all clean/randomized task counts and the separately labeled Dense-12 evaluations. Original scored records and audit files are unchanged.
