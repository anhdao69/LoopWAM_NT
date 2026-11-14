# dense-s12 epoch 10 — RoboTwin-10 — final evaluation

Completed 2026-11-13T18:45:37.976714+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 769/1000 | 76.90% |
| demo_randomized | 776/1000 | 77.60% |
| combined | 1545/2000 | 77.25% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 46 | 43 | 44.50% |
| move_stapler_pad | 54 | 68 | 61.00% |
| open_microwave | 64 | 42 | 53.00% |
| press_stapler | 91 | 90 | 90.50% |
| scan_object | 79 | 80 | 79.50% |
| stack_blocks_two | 97 | 99 | 98.00% |
| stamp_seal | 84 | 90 | 87.00% |
| turn_switch | 54 | 65 | 59.50% |

Controller elapsed time: 4.57 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 8; logged partial episodes excluded from scoring: 266.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native dense-s12, 12 stored block pairs, video/action loops 1/1. All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

The epoch10 payload and native model variant are strictly validated. See setup for checkpoint provenance. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Comparison with previous local v0 4/1 epoch10

| Setting | v0 4/1 epoch10 | Current model epoch10 | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 80.00% | 76.90% | -3.10 |
| demo_randomized | 81.90% | 77.60% | -4.30 |
| combined | 80.95% | 77.25% | -3.70 |

This is an aggregate comparison under matched evaluation settings. Unseeded Python instruction sampling prevent claiming exactly paired trajectories. Training artifact hash differences are recorded in setup and source checks.

Migration provenance: 400 complete episodes were generated on the local RTX PRO6000, transferred without changing outcomes, seeds, instructions, action traces or video hashes, and revalidated on aLinh. The remaining 1600 episodes were generated on aLinh RTX PRO5000 using two workers. Original interrupted local attempts remain excluded. Reported controller time covers the remote segment only.

## November 15 report reconciliation

This result was freshly reconciled against its saved episode records, CSV exports and final audit digest. The [epoch-10 task comparison and detailed report](../../robotwin10_epoch10_results_20261115/README.md) includes all clean/randomized task counts and the separately labeled Dense-12 evaluations. Original scored records and audit files are unchanged.
