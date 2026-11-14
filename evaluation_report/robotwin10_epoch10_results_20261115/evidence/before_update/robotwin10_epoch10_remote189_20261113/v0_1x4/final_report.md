# v0-video1-action4 epoch 10 — RoboTwin-10 — final evaluation

Completed 2026-11-14T04:49:05.675193+07:00. All 20 task/setting pairs and 2,000 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 802/1000 | 80.20% |
| demo_randomized | 789/1000 | 78.90% |
| combined | 1591/2000 | 79.55% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% |
| grab_roller | 100 | 100 | 100.00% |
| hanging_mug | 49 | 42 | 45.50% |
| move_stapler_pad | 65 | 62 | 63.50% |
| open_microwave | 74 | 73 | 73.50% |
| press_stapler | 89 | 88 | 88.50% |
| scan_object | 87 | 79 | 83.00% |
| stack_blocks_two | 98 | 97 | 97.50% |
| stamp_seal | 82 | 84 | 83.00% |
| turn_switch | 58 | 65 | 61.50% |

Controller elapsed time: 3.34 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 2; logged partial episodes excluded from scoring: 112.

Protocol: seed 42; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0-video1-action4, 12 stored block pairs, video/action loops 1/4. All smoke episodes and the eight-input graph gate are excluded. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

The epoch10 payload and native model variant are strictly validated. See setup for checkpoint provenance. See [setup](setup_report.md), [source checks](source_check.json), [protocol comparison](protocol_comparison.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).


## Comparison with previous local v0 4/1 epoch10

| Setting | v0 4/1 epoch10 | Current model epoch10 | Difference (percentage points) |
| --- | ---: | ---: | ---: |
| demo_clean | 80.00% | 80.20% | +0.20 |
| demo_randomized | 81.90% | 78.90% | -3.00 |
| combined | 80.95% | 79.55% | -1.40 |

This is an aggregate comparison under matched evaluation settings. Unseeded Python instruction sampling prevent claiming exactly paired trajectories. Training artifact hash differences are recorded in setup and source checks.

Migration provenance: 0 complete episodes were generated on the local RTX PRO6000, transferred without changing outcomes, seeds, instructions, action traces or video hashes, and revalidated on aLinh. The remaining 2000 episodes were generated on aLinh RTX PRO5000 using two workers. Original interrupted local attempts remain excluded. Reported controller time covers the remote segment only.