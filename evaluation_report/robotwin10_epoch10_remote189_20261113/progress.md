# RoboTwin-10 epoch 10 queue

Queue status: **complete**. Live partial results are provisional until final audit.

| Model | Stage | Episodes | Successes | SR |
| --- | --- | ---: | ---: | ---: |
| dense-s12 | complete | 2000/2000 | 1545 | 77.25% |
| v0-video1-action4 | complete | 2000/2000 | 1591 | 79.55% |

Seed 42; 10 tasks; clean + randomized; 100 episodes/task/setting; NFE 10; predict 32 / execute 24. Two workers, bit-exact CUDA graphs, shared T5. Gates and smoke episodes excluded.

[Detailed setup](setup_report.md)
