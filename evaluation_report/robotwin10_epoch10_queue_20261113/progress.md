# RoboTwin-10 epoch10 — current progress

Execution moved to **aLinh /10.148.183.189**. Queue status: **complete**. The local GPU evaluation remains stopped.

| Model | Stage | Episodes | Successes | SR |
| --- | --- | ---: | ---: | ---: |
| 4/2 | complete | 2000/2000 | 1648 | 82.40% |
| 2/2 | complete | 2000/2000 | 1578 | 78.90% |
| Dense-12 | complete | 2000/2000 | 1545 | 77.25% |
| 1/4 | complete | 2000/2000 | 1591 | 79.55% |

Dense-12 carries400 verified complete local episodes; all interrupted attempts are excluded. Remote runs use two workers on the48GB RTXPRO5000. Protocol remains seed42,NFE10,predict32/execute24.

[Migration setup](/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_epoch10_remote189_20261113/setup_report.md)
[Remote live progress](/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_epoch10_remote189_20261113/progress.md)
