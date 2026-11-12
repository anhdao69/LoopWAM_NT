# GR1-E0 x5 (elastic E0 on RoboCasa-GR1) - official evaluation results

Updated 2026-11-11 23:26 UTC. Protocol: 24 tasks x 50 episodes, n_action_steps 12, max 720 steps, seed 0, EGL (the GR1 v0 continuation protocol). Rates are any-step success over 1200 episodes.

Teacher: GR1 v0 continuation epoch 9 (sha256 694a679b4c07934b...) - epoch 10 620/1200 <= epoch 9 626/1200.
Teacher official results: epoch 9 626/1200, epoch 10 620/1200

| epoch | update | schedule | NFE | success | rate | valid |
|---|---|---|---|---|---|---|
| 1 | 47032 | flat_k1_n2 [1, 1] | 2 | 474/1200 | 39.50% | PROBLEMS |
| 2 | 94064 | flat_k1_n2 [1, 1] | 2 | 533/1200 | 44.42% | VALID |
| 2 | 94064 | flat_k2_n2  | 2 | 627/1200 | 52.25% | VALID |
| 2 | 94064 | flat_k4_n2 [4, 4] | 2 | 577/1200 | 48.08% | VALID |
| 3 | 141096 | flat_k1_n2 [1, 1] | 2 | 505/1200 | 42.08% | VALID |
| 3 | 141096 | flat_k2_n2  | 2 | 608/1200 | 50.67% | VALID |
| 3 | 141096 | flat_k4_n2 [4, 4] | 2 | 586/1200 | 48.83% | VALID |
| 4 | 188128 | flat_k1_n10  | 10 | 561/1200 | 46.75% | VALID |
| 4 | 188128 | flat_k1_n2 [1, 1] | 2 | 518/1200 | 43.17% | VALID |
| 4 | 188128 | flat_k1_n5  | 5 | 569/1200 | 47.42% | VALID |
| 4 | 188128 | flat_k2_n10  | 10 | 609/1200 | 50.75% | VALID |
| 4 | 188128 | flat_k2_n2  | 2 | 619/1200 | 51.58% | VALID |
| 4 | 188128 | flat_k2_n5  | 5 | 609/1200 | 50.75% | VALID |
| 4 | 188128 | flat_k4_n10  | 10 | 620/1200 | 51.67% | VALID |
| 4 | 188128 | flat_k4_n2 [4, 4] | 2 | 629/1200 | 52.42% | VALID |
| 4 | 188128 | flat_k4_n5 [4]x5 | 5 | 642/1200 | 53.50% | VALID |
| 5 | 235160 | flat_k1_n10  | 10 | 593/1200 | 49.42% | VALID |
| 5 | 235160 | flat_k1_n2 [1, 1] | 2 | 533/1200 | 44.42% | VALID |
| 5 | 235160 | flat_k1_n5  | 5 | 558/1200 | 46.50% | PROBLEMS |
| 5 | 235160 | flat_k2_n10  | 10 | 607/1200 | 50.58% | VALID |
| 5 | 235160 | flat_k2_n2  | 2 | 598/1200 | 49.83% | VALID |
| 5 | 235160 | flat_k2_n5  | 5 | 618/1200 | 51.50% | VALID |
| 5 | 235160 | flat_k4_n10  | 10 | 628/1200 | 52.33% | VALID |
| 5 | 235160 | flat_k4_n2 [4, 4] | 2 | 621/1200 | 51.75% | VALID |
| 5 | 235160 | flat_k4_n5 [4]x5 | 5 | 653/1200 | 54.42% | VALID |

Status: done_with_problems - GR1-E0 x5 trained and evaluated (epoch 5: nine FLAT9 schedules); epoch5/flat_k1_n5:eval (RESULTS.md) (2026-11-11T13:15:42Z)
