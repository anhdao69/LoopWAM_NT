# Elastic E0 teacher10 / five-epoch student — [4,4] final result

**Complete and audited on server137.** Updated November 13, 2026.

| Setting | Successes / episodes | Final SR |
| --- | ---: | ---: |
| Clean | 4567/5000 | 91.34% |
| Randomized | 4582/5000 | 91.64% |
| Combined | 9149/10000 | **91.49%** |

All 50 tasks, seed 42, 100 accepted episodes per task/setting, [4,4], NFE 2, four video loops, predict 32 / execute 24, four workers. The original final audit verified 100 cells and 10,000 videos, with zero failed attempts and zero excluded partial scored episodes. Eight unscored smoke episodes are excluded.

Scored evaluator duration: **15h 40m 56s**; post-process finalization: **4m 55s**. Final audit completed at **2026-11-13 01:56:44 ICT**, using the recorded server clock. The original waiting local supervisor produced no scored episodes. The inherited cross-host gate covers [1,1]/[2,2]; [4,4] has its own server137 native/graph parity gate.

**[Detailed final comparison with [1,1] and [2,2], all tasks and epoch-1 E0](../robotwin_elastic_e0_teacher10_5ep_20261111/final_report.md)** · [Per-task final report](flat_k4_n2/final_report.md) · [Audit](flat_k4_n2/final_audit.json) · [Setup and migration](setup_report.md).
