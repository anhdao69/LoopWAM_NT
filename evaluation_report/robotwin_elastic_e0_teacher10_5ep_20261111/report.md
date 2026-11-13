# Elastic E0 teacher epoch10 / student five epochs — final RT-50 results

Updated November 13, 2026. **All three schedules completed and passed their final audits.**

| Schedule | NFE | Clean | Randomized | Overall | Successes / episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| [1,1] | 2 | 89.52% | 89.04% | **89.28%** | 8928/10000 |
| [2,2] | 2 | 90.58% | 90.62% | **90.60%** | 9060/10000 |
| [4,4] | 2 | 91.34% | 91.64% | **91.49%** | 9149/10000 |

**[Detailed final report: all 50 tasks, protocol, runtimes, audits and epoch-1 comparison](final_report.md)** · [CSV](task_comparison.csv) · [Reconciled JSON](final_comparison.json).

Same E0-epoch10-teacher-5ep checkpoint; student epoch 5, update 237,310. Seed 42; 50 tasks × clean/randomized × 100 accepted episodes per task/setting. Four video loops; predict 32 / execute 24; FP32 policy weights and BF16 compute. [1,1] ran locally; [2,2] and [4,4] ran on server137, each with four workers, shared T5 and validated CUDA graphs.

[4,4] exceeds [1,1] by **2.21 percentage points** and [2,2] by **0.89 points**. Versus the earlier epoch-1 E0, [1,1] improves from 86.84% to 89.28% (+2.44 points), and [4,4] from 90.25% to 91.49% (+1.24 points). There is no completed earlier E0 [2,2] baseline in this campaign.

All 30,000 new episode records were reconciled with original-host logs/traces; retained videos, checkpoint/source hashes and saved audit digests were checked. Original full video-content audits were not repeated. Python instruction-template RNG remains unseeded, and the actual records show imperfect scene/instruction pairing; the final report quantifies this limitation.

[Original setup](setup_report.md) · [[4,4] continuation](../robotwin_elastic_e0_teacher10_5ep_44_20261112/report.md). The old `current_results.json` remains the original two-schedule snapshot; the three-schedule view is `final_comparison.json`. No evaluation was resumed for this update.
