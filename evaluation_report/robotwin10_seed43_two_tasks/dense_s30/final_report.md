# dense-s30 — two-task seed-43 evaluation

Completed 2026-11-03T23:58:30.240722+07:00. All 4 task/setting pairs and 400 scored episodes passed artifact verification.

| Setting | Successes / episodes | SR |
| --- | ---: | ---: |
| demo_clean | 147/200 | 73.50% |
| demo_randomized | 127/200 | 63.50% |
| combined | 274/400 | 68.50% |

| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |
| --- | ---: | ---: | ---: |
| open_microwave | 75 | 53 | 64.00% |
| move_stapler_pad | 72 | 74 | 73.00% |

Controller elapsed time: 1.28 hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.
Interrupted attempts: 0; logged partial episodes excluded from scoring: 0.

Protocol: seed 43; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native dense-s30 (30 block pairs, one pass). All unscored calibration episodes are excluded. The reused exact-output graph gate included seed 43. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.

Both compared RoboTwin-10 releases were trained on the same ten tasks for five epochs. This report covers only the two requested tasks. See [setup](setup_report.md), [source checks](source_check.json), [protocol](planned_protocol.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).
