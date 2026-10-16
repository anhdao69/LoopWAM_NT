# CL evaluation report: LIBERO-Plus and LIBERO-Pro

Final results for seven epoch-10 CL checkpoints, seed **42 only**. All **14 jobs / 16,142 scored episodes** completed. No scored jobs are marked failed. This report covers the Plus/Pro campaign; prior standard-LIBERO and other model-family campaigns remain in their separate reports. No training was performed.

Queue start: **2026-10-15 11:16:46**. Queue finish: **2026-10-16 19:24:56**. Timezone: Vietnam, UTC+07:00. Total queue wall time: **32h 08m 10s**, including calibration, model loading, rollouts, logging, and controller overhead. Markdown generated 2026-10-16 19:42:58.

## Contents

- [Final success rates](#final-success-rates)
- [Evaluation protocol](#evaluation-protocol)
- [Detailed results by suite](#detailed-results-by-suite)
- [LIBERO-Plus results by category](#libero-plus-results-by-category)
- [Runtime and inference optimization](#runtime-and-inference-optimization)
- [Checkpoint identities and source versions](#checkpoint-identities-and-source-versions)
- [Validation, setup, and limitations](#validation-setup-and-limitations)
- [Logs, recovery, and artifacts](#logs-recovery-and-artifacts)
- [Extended code and artifact audit](#extended-code-and-artifact-audit)

## Final success rates

| Model | Plus successes / 1,506 | Plus SR | Pro Position / 400 | Position SR | Pro Task / 400 | Task SR | Pro combined / 800 | Pro SR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CL-0 | 708 | 47.01% | 3 | 0.75% | 55 | 13.75% | 58 | 7.25% |
| CL-A | 719 | 47.74% | 3 | 0.75% | 57 | 14.25% | 60 | 7.50% |
| CL-REG | 721 | 47.88% | 2 | 0.50% | 58 | 14.50% | 60 | 7.50% |
| CL-W2 | 729 | 48.41% | 1 | 0.25% | 52 | 13.00% | 53 | 6.63% |
| CL-0@1 | 689 | 45.75% | 2 | 0.50% | 54 | 13.50% | 56 | 7.00% |
| CL-A@1 | 693 | 46.02% | 4 | 1.00% | 56 | 14.00% | 60 | 7.50% |
| CL-FRAME | 719 | 47.74% | 3 | 0.75% | 59 | 14.75% | 62 | 7.75% |

SR is successes divided by scored episodes (micro-average). Pro combines two separate perturbations with equal 400-episode denominators. Smoke and calibration episodes are excluded. CL-W2 has the highest observed Plus SR; CL-FRAME has the highest observed combined Pro SR. These single-seed results do not establish statistical significance. The very low Pro Position scores are recorded outcomes; execution completion and setup checks do not by themselves establish the cause or exclude every integration issue.

## Evaluation protocol

### LIBERO-Plus: FasterWAM released configuration

The full Plus benchmark was not used. Each suite independently selects ceil(15% of its catalogue) with the released manager sampling rule, seed 42. One episode is run per selected task, from official initial-state index 0. The exact rule is:

```python
sorted(random.Random(f"42:{suite}").sample(range(n_tasks), math.ceil(0.15 * n_tasks)))
```

| Suite | Episodes per model | Policy-step cap |
| --- | --- | --- |
| libero_10 | 378 | 700 |
| libero_goal | 389 | 400 |
| libero_spatial | 361 | 400 |
| libero_object | 378 | 400 |

`libero_10` is the Long suite. The total is 1,506 episodes/model across all seven native perturbation categories. Instructions use official `Task.language` exactly as passed by FasterWAM, including any suffixes. No language rewriting or perturbation simplification is applied. Thirty settling steps precede the policy-step allowance.

| Category | Selected episodes per model |
| --- | --- |
| Background Textures | 185 |
| Robot Initial States | 198 |
| Camera Viewpoints | 240 |
| Language Instructions | 244 |
| Sensor Noise | 257 |
| Objects Layout | 220 |
| Light Conditions | 162 |

### LIBERO-Pro: Position and Task

Position (`*_swap`) and Task (`*_task`) are evaluated separately. Each perturbation has four suites, 10 tasks/suite, and 10 episodes/task: 400 episodes per perturbation, 800/model. State indices 0–9 are selected from the official initial states. All 80 task groups have ten distinct selected states. Task ordering follows the official base catalogue paired with the matching published perturbation BDDL and state files.

| Suite | Policy-step cap | Position episodes | Task episodes |
| --- | --- | --- | --- |
| libero_spatial | 220 | 100 | 100 |
| libero_object | 280 | 100 | 100 |
| libero_goal | 300 | 100 | 100 |
| libero_10 | 520 | 100 | 100 |

Position preserves original catalogue language. Task uses the changed goal from perturbed BDDL `language_instruction`; all 40 Task instructions change. Native `liberopro.liberopro` environments execute the perturbations. Standard LIBERO remains available solely for the existing CL image/quaternion utility imports.

### Shared model inputs, actions, precision, and RNG

| Setting | Recorded configuration |
| --- | --- |
| settling_steps | 30 |
| settling_action | [0, 0, 0, 0, 0, 0, -1] |
| execute_actions | 10 |
| predicted_horizon | 32 |
| denoising_steps | 10 |
| precision | FP32 native CL model with BF16 autocast/VAE, identical to previous CL campaign |
| observations | Native trained CL adapter: two 256x256 views rotated180, antialiased resize224 each, horizontal concatenation448x224; 8D proprio; verified training normalization |
| gripper | Prior matched CL adapter: sign(-(2*g-1)) including neutral0 at exact midpoint |
| memory | Native Chrono config from each epoch10 checkpoint; state reset every episode; CL-FRAME query history reset every episode |
| text | Actual per-task instruction encoded once with verified original UMT5, batch1 BF16, length128, cached with masks and hashes |
| seeding | 42 once per task group and env.seed(42); query seed42; no episode reseeding |
| recovery | Only complete atomic task groups are recovered; interrupted groups restart at episode0 to preserve RNG sequence |
| video | Every scored and calibration episode saved as MP4; SHA256 and timing logged |
| scoring | Micro-average over scored episodes; category/suite scores separate; no calibration episodes in SR |
| cached_instruction_count | 1579 |
| text_provenance_sha256 | 3622e6a9b35fc99fae191faf3a67703edd1783434d76456415d39f31f4989339 |

Python, NumPy, and Torch are seeded at task-group entry, with environment seed 42 and query seed 42. The ten Pro episodes in a group are not independently reseeded. Camera preprocessing rotates both 256×256 views 180°, resizes each to 224×224 with antialiasing, and concatenates them horizontally to 448×224. The native adapter supplies eight-dimensional proprioception with the training normalization. Memory and CL-FRAME history reset at each episode. The text prompt is `A video recorded from a robot's point of view executing the following instruction: {task}`.

### Implementation and comparison limits

FasterWAM published run_single_episode body executed unchanged via AST; only policy callback replaced by native CL adapter. Pro horizon lookup uses official Pro limits.

- FasterWAM paper describes seven OOD shifts but omits exact subset and evaluator RNG details; the released configuration/manager are authoritative for this campaign. This is the released-config evaluation of CL checkpoints, not reproduction of FasterWAM model weights.

- Published manager seeds once per process chunk; this campaign seeds once per independent task group for reproducibility under worker-count changes. Environment and each policy query use42 in both. This RNG scoping difference is recorded explicitly.

- Pro caps differ from prior standard LIBERO CL campaign: official Pro220/280/300/520, versus prior400/400/400/700.

- Task perturbation filenames retain original task names; language is read from the perturbed BDDL.

- Plus sampled category counts are reported as generated by pinned config and catalogue; exact paper task-list artifacts were not published in the inspected repository.

The same selected checkpoint, native preprocessing, action conversion, horizon, denoising count, and precision are used across benchmarks. These are the previously selected epoch-10 weights; the report does not claim they were the newest upstream weights on its generation date. Plus, Pro, and the prior standard-LIBERO campaign have different tasks and evaluation settings; their scores should not be treated as interchangeable. This campaign is not a rerun of FasterWAM model weights.

## Detailed results by suite

### LIBERO-Plus

| Model | Suite / perturbation | Episodes | Successes | SR |
| --- | --- | --- | --- | --- |
| CL-0 | libero_10 | 378 | 159 | 42.06% |
| CL-0 | libero_goal | 389 | 123 | 31.62% |
| CL-0 | libero_object | 378 | 246 | 65.08% |
| CL-0 | libero_spatial | 361 | 180 | 49.86% |
| CL-A | libero_10 | 378 | 166 | 43.92% |
| CL-A | libero_goal | 389 | 129 | 33.16% |
| CL-A | libero_object | 378 | 252 | 66.67% |
| CL-A | libero_spatial | 361 | 172 | 47.65% |
| CL-REG | libero_10 | 378 | 167 | 44.18% |
| CL-REG | libero_goal | 389 | 121 | 31.11% |
| CL-REG | libero_object | 378 | 257 | 67.99% |
| CL-REG | libero_spatial | 361 | 176 | 48.75% |
| CL-W2 | libero_10 | 378 | 178 | 47.09% |
| CL-W2 | libero_goal | 389 | 128 | 32.90% |
| CL-W2 | libero_object | 378 | 248 | 65.61% |
| CL-W2 | libero_spatial | 361 | 175 | 48.48% |
| CL-0@1 | libero_10 | 378 | 149 | 39.42% |
| CL-0@1 | libero_goal | 389 | 135 | 34.70% |
| CL-0@1 | libero_object | 378 | 236 | 62.43% |
| CL-0@1 | libero_spatial | 361 | 169 | 46.81% |
| CL-A@1 | libero_10 | 378 | 152 | 40.21% |
| CL-A@1 | libero_goal | 389 | 126 | 32.39% |
| CL-A@1 | libero_object | 378 | 244 | 64.55% |
| CL-A@1 | libero_spatial | 361 | 171 | 47.37% |
| CL-FRAME | libero_10 | 378 | 165 | 43.65% |
| CL-FRAME | libero_goal | 389 | 130 | 33.42% |
| CL-FRAME | libero_object | 378 | 239 | 63.23% |
| CL-FRAME | libero_spatial | 361 | 185 | 51.25% |

### LIBERO-Pro

| Model | Suite / perturbation | Episodes | Successes | SR |
| --- | --- | --- | --- | --- |
| CL-0 | libero_10 | 200 | 9 | 4.50% |
| CL-0 | libero_goal | 200 | 12 | 6.00% |
| CL-0 | libero_object | 200 | 10 | 5.00% |
| CL-0 | libero_spatial | 200 | 27 | 13.50% |
| CL-A | libero_10 | 200 | 9 | 4.50% |
| CL-A | libero_goal | 200 | 12 | 6.00% |
| CL-A | libero_object | 200 | 10 | 5.00% |
| CL-A | libero_spatial | 200 | 29 | 14.50% |
| CL-REG | libero_10 | 200 | 10 | 5.00% |
| CL-REG | libero_goal | 200 | 12 | 6.00% |
| CL-REG | libero_object | 200 | 10 | 5.00% |
| CL-REG | libero_spatial | 200 | 28 | 14.00% |
| CL-W2 | libero_10 | 200 | 8 | 4.00% |
| CL-W2 | libero_goal | 200 | 10 | 5.00% |
| CL-W2 | libero_object | 200 | 10 | 5.00% |
| CL-W2 | libero_spatial | 200 | 25 | 12.50% |
| CL-0@1 | libero_10 | 200 | 9 | 4.50% |
| CL-0@1 | libero_goal | 200 | 12 | 6.00% |
| CL-0@1 | libero_object | 200 | 10 | 5.00% |
| CL-0@1 | libero_spatial | 200 | 25 | 12.50% |
| CL-A@1 | libero_10 | 200 | 9 | 4.50% |
| CL-A@1 | libero_goal | 200 | 14 | 7.00% |
| CL-A@1 | libero_object | 200 | 10 | 5.00% |
| CL-A@1 | libero_spatial | 200 | 27 | 13.50% |
| CL-FRAME | libero_10 | 200 | 11 | 5.50% |
| CL-FRAME | libero_goal | 200 | 14 | 7.00% |
| CL-FRAME | libero_object | 200 | 10 | 5.00% |
| CL-FRAME | libero_spatial | 200 | 27 | 13.50% |

## LIBERO-Plus results by category

| Model | Category | Episodes | Successes | SR |
| --- | --- | --- | --- | --- |
| CL-0 | Background Textures | 185 | 65 | 35.14% |
| CL-0 | Robot Initial States | 198 | 77 | 38.89% |
| CL-0 | Camera Viewpoints | 240 | 77 | 32.08% |
| CL-0 | Language Instructions | 244 | 146 | 59.84% |
| CL-0 | Sensor Noise | 257 | 74 | 28.79% |
| CL-0 | Objects Layout | 220 | 126 | 57.27% |
| CL-0 | Light Conditions | 162 | 143 | 88.27% |
| CL-A | Background Textures | 185 | 63 | 34.05% |
| CL-A | Robot Initial States | 198 | 80 | 40.40% |
| CL-A | Camera Viewpoints | 240 | 80 | 33.33% |
| CL-A | Language Instructions | 244 | 149 | 61.07% |
| CL-A | Sensor Noise | 257 | 72 | 28.02% |
| CL-A | Objects Layout | 220 | 132 | 60.00% |
| CL-A | Light Conditions | 162 | 143 | 88.27% |
| CL-REG | Background Textures | 185 | 65 | 35.14% |
| CL-REG | Robot Initial States | 198 | 79 | 39.90% |
| CL-REG | Camera Viewpoints | 240 | 79 | 32.92% |
| CL-REG | Language Instructions | 244 | 148 | 60.66% |
| CL-REG | Sensor Noise | 257 | 74 | 28.79% |
| CL-REG | Objects Layout | 220 | 135 | 61.36% |
| CL-REG | Light Conditions | 162 | 141 | 87.04% |
| CL-W2 | Background Textures | 185 | 66 | 35.68% |
| CL-W2 | Robot Initial States | 198 | 76 | 38.38% |
| CL-W2 | Camera Viewpoints | 240 | 87 | 36.25% |
| CL-W2 | Language Instructions | 244 | 153 | 62.70% |
| CL-W2 | Sensor Noise | 257 | 78 | 30.35% |
| CL-W2 | Objects Layout | 220 | 127 | 57.73% |
| CL-W2 | Light Conditions | 162 | 142 | 87.65% |
| CL-0@1 | Background Textures | 185 | 68 | 36.76% |
| CL-0@1 | Robot Initial States | 198 | 82 | 41.41% |
| CL-0@1 | Camera Viewpoints | 240 | 60 | 25.00% |
| CL-0@1 | Language Instructions | 244 | 153 | 62.70% |
| CL-0@1 | Sensor Noise | 257 | 63 | 24.51% |
| CL-0@1 | Objects Layout | 220 | 129 | 58.64% |
| CL-0@1 | Light Conditions | 162 | 134 | 82.72% |
| CL-A@1 | Background Textures | 185 | 60 | 32.43% |
| CL-A@1 | Robot Initial States | 198 | 73 | 36.87% |
| CL-A@1 | Camera Viewpoints | 240 | 68 | 28.33% |
| CL-A@1 | Language Instructions | 244 | 156 | 63.93% |
| CL-A@1 | Sensor Noise | 257 | 62 | 24.12% |
| CL-A@1 | Objects Layout | 220 | 136 | 61.82% |
| CL-A@1 | Light Conditions | 162 | 138 | 85.19% |
| CL-FRAME | Background Textures | 185 | 61 | 32.97% |
| CL-FRAME | Robot Initial States | 198 | 72 | 36.36% |
| CL-FRAME | Camera Viewpoints | 240 | 92 | 38.33% |
| CL-FRAME | Language Instructions | 244 | 150 | 61.48% |
| CL-FRAME | Sensor Noise | 257 | 71 | 27.63% |
| CL-FRAME | Objects Layout | 220 | 130 | 59.09% |
| CL-FRAME | Light Conditions | 162 | 143 | 88.27% |

## Runtime and inference optimization

The queue runs all seven Plus jobs, followed by the same seven Pro jobs. Models run sequentially; independent model/environment workers parallelize episodes within a job. Every scored job selected four workers, each with two CPU threads. Optimizations include inference mode without gradient tracking, BF16 autocast/VAE, verified cached text embeddings, and parallel rollout execution. FP32 model weights, native eager inference, 10 denoising steps, 32 predicted actions, and 10 executed actions are retained. Videos and query traces remain enabled.

Candidates are 4, 8, and 12 workers. Higher-count calibration must match the four-worker reference in successes, steps, query counts, and every predicted-action-chunk hash. The fastest eligible measured profile is selected. VRAM admission includes measured worker peak, 12% headroom, and a 2 GiB reserve. This finite check does not prove globally maximal speed or equivalence for all future rollouts. GPU sharing varied during the campaign; timings are observed campaign performance, not isolated hardware benchmarks. No CUDA graphs, quantization, or shortened horizons were used.

| Job | Scored wall time | Runner incl. load | Job incl. calibration | Workers | Queries | Finish UTC+07 |
| --- | --- | --- | --- | --- | --- | --- |
| plus__CL-0 | 3h 15m 58s | 3h 16m 04s | 3h 22m 15s | 4 | 51459 | 2026-10-15 14:39:00 |
| plus__CL-A | 2h 57m 17s | 2h 57m 23s | 3h 03m 45s | 4 | 51119 | 2026-10-15 17:42:45 |
| plus__CL-REG | 2h 53m 23s | 2h 53m 29s | 2h 59m 03s | 4 | 51127 | 2026-10-15 20:41:48 |
| plus__CL-W2 | 3h 23m 30s | 3h 23m 36s | 3h 28m 31s | 4 | 51035 | 2026-10-16 00:10:19 |
| plus__CL-0@1 | 2h 35m 47s | 2h 35m 55s | 2h 43m 25s | 4 | 51977 | 2026-10-16 02:53:44 |
| plus__CL-A@1 | 2h 38m 34s | 2h 38m 42s | 2h 46m 35s | 4 | 51816 | 2026-10-16 05:40:19 |
| plus__CL-FRAME | 3h 50m 01s | 3h 50m 09s | 4h 00m 13s | 4 | 51034 | 2026-10-16 09:40:32 |
| pro__CL-0 | 1h 30m 38s | 1h 30m 46s | 1h 33m 33s | 4 | 25602 | 2026-10-16 11:14:06 |
| pro__CL-A | 1h 29m 20s | 1h 29m 28s | 1h 32m 16s | 4 | 25570 | 2026-10-16 12:46:22 |
| pro__CL-REG | 1h 29m 26s | 1h 29m 33s | 1h 32m 23s | 4 | 25549 | 2026-10-16 14:18:45 |
| pro__CL-W2 | 1h 29m 55s | 1h 30m 03s | 1h 32m 49s | 4 | 25621 | 2026-10-16 15:51:35 |
| pro__CL-0@1 | 0h 55m 25s | 0h 55m 32s | 0h 57m 33s | 4 | 25575 | 2026-10-16 16:49:09 |
| pro__CL-A@1 | 0h 55m 31s | 0h 55m 39s | 0h 57m 48s | 4 | 25548 | 2026-10-16 17:46:57 |
| pro__CL-FRAME | 1h 34m 48s | 1h 34m 56s | 1h 37m 58s | 4 | 25559 | 2026-10-16 19:24:55 |

Scored wall time starts after worker readiness and includes rollout and artifact work. Per-query times summed across parallel workers are not queue wall time.

| Job | Candidate workers | Calibration seconds | Outcome |
| --- | --- | --- | --- |
| plus__CL-0 | 4 | 161.22 | reference |
| plus__CL-0 | 8 | 163.56 | signature mismatch; ineligible |
| plus__CL-0 | 12 | — | Insufficient free VRAM with 12% worker headroom and2GiB reserve |
| plus__CL-A | 4 | 160.95 | reference |
| plus__CL-A | 8 | 164.53 | signature mismatch; ineligible |
| plus__CL-A | 12 | — | Insufficient free VRAM with 12% worker headroom and2GiB reserve |
| plus__CL-REG | 4 | 138.03 | reference |
| plus__CL-REG | 8 | 145.15 | signature mismatch; ineligible |
| plus__CL-REG | 12 | — | Insufficient free VRAM with 12% worker headroom and2GiB reserve |
| plus__CL-W2 | 4 | 116.93 | reference |
| plus__CL-W2 | 8 | 128.40 | signature mismatch; ineligible |
| plus__CL-W2 | 12 | — | Insufficient free VRAM with 12% worker headroom and2GiB reserve |
| plus__CL-0@1 | 4 | 135.12 | reference |
| plus__CL-0@1 | 8 | 121.69 | signature mismatch; ineligible |
| plus__CL-0@1 | 12 | 109.01 | signature mismatch; ineligible |
| plus__CL-A@1 | 4 | 138.90 | reference |
| plus__CL-A@1 | 8 | 126.81 | signature mismatch; ineligible |
| plus__CL-A@1 | 12 | 125.09 | signature mismatch; ineligible |
| plus__CL-FRAME | 4 | 195.46 | reference |
| plus__CL-FRAME | 8 | 169.78 | signature mismatch; ineligible |
| plus__CL-FRAME | 12 | 161.96 | signature mismatch; ineligible |
| pro__CL-0 | 4 | 65.35 | reference |
| pro__CL-0 | 8 | 57.49 | signature mismatch; ineligible |
| pro__CL-0 | 12 | — | Insufficient free VRAM with 12% worker headroom and2GiB reserve |
| pro__CL-A | 4 | 64.31 | reference |
| pro__CL-A | 8 | 56.54 | signature mismatch; ineligible |
| pro__CL-A | 12 | — | Insufficient free VRAM with 12% worker headroom and2GiB reserve |
| pro__CL-REG | 4 | 64.01 | reference |
| pro__CL-REG | 8 | 56.73 | signature mismatch; ineligible |
| pro__CL-REG | 12 | — | Insufficient free VRAM with 12% worker headroom and2GiB reserve |
| pro__CL-W2 | 4 | 64.64 | reference |
| pro__CL-W2 | 8 | 57.32 | signature mismatch; ineligible |
| pro__CL-W2 | 12 | — | Insufficient free VRAM with 12% worker headroom and2GiB reserve |
| pro__CL-0@1 | 4 | 41.31 | reference |
| pro__CL-0@1 | 8 | 35.56 | signature mismatch; ineligible |
| pro__CL-0@1 | 12 | — | Insufficient free VRAM with 12% worker headroom and2GiB reserve |
| pro__CL-A@1 | 4 | 42.52 | reference |
| pro__CL-A@1 | 8 | 36.33 | signature mismatch; ineligible |
| pro__CL-A@1 | 12 | — | Insufficient free VRAM with 12% worker headroom and2GiB reserve |
| pro__CL-FRAME | 4 | 70.79 | reference |
| pro__CL-FRAME | 8 | 63.31 | signature mismatch; ineligible |
| pro__CL-FRAME | 12 | — | Insufficient free VRAM with 12% worker headroom and2GiB reserve |

## Checkpoint identities and source versions

### CL-0

Checkpoint: `/data14t/anhdh35/loopwam_campaign/checkpoints/ChronoLoop/fea8b7334294ce67058c44105ec858025f3eca28/CL-0_no-memory_a4/epoch_10/policy.pt`

SHA-256: `137f0a839b5a1e2363621420ab599cd36a5693000f784a04a11b9bea2bea2cae`

### CL-A

Checkpoint: `/data14t/anhdh35/loopwam_campaign/checkpoints/ChronoLoop/8e25a4e7d117c36b50fdad51f83c5c4211d47be5/CL-A_mem16-learned-loopwrite_a4/epoch_10/policy.pt`

SHA-256: `3b480febd1888712efbcaaa9b5d97641d96a6a088a505d86b6ae14c237fcee39`

### CL-REG

Checkpoint: `/data14t/anhdh35/loopwam_campaign/checkpoints/ChronoLoop/f45c862e1e45645eee407c7a26d0e9fd816bc872/CL-REG_mem16-reset-registers_a4/epoch_10/policy.pt`

SHA-256: `6de1bbf80ab0566a52b3820738ddff35673c2476a6c811e638c9422b826f762d`

### CL-W2

Checkpoint: `/data14t/anhdh35/loopwam_campaign/checkpoints/ChronoLoop/8706968184639b57fc12410cad4a89dce3f79836/CL-W2_mem16-learned-external-updater_a4/epoch_10/policy.pt`

SHA-256: `d08096facd67779d4817a491a60c5ad07a33ef80b94eb1b64d64d2445a3d0032`

### CL-0@1

Checkpoint: `/data14t/anhdh35/loopwam_campaign/checkpoints/ChronoLoop/8e25a4e7d117c36b50fdad51f83c5c4211d47be5/CL-0-at1_no-memory_a1/epoch_10/policy.pt`

SHA-256: `77cbd77a521686e968229357b3391241b394ce4ea16d938d03b3ae38e4256930`

### CL-A@1

Checkpoint: `/data14t/anhdh35/loopwam_campaign/checkpoints/ChronoLoop/204137bc56b946ff46816a3f64bd268e03d99d69/CL-A-at1_mem16-learned-loopwrite_a1/epoch_10/policy.pt`

SHA-256: `391d3b59bcd2dc6d5f326fa4e4729b76ab31a6658bac7714cba39761d4234781`

### CL-FRAME

Checkpoint: `/data14t/anhdh35/loopwam_campaign/checkpoints/ChronoLoop/cf52c6364579e45408150cd5f526b573d42eef4d/CL-FRAME_history-frame-k3_a4/epoch_10/policy.pt`

SHA-256: `b322ae52c8166dbe222aeafea03da7138d60b823df73ce0b09c2b6dfc2c27a8a`

| Model | Published source revision | Evaluation revision | Native source matches | Adapter AST matches |
| --- | --- | --- | --- | --- |
| CL-0 | 97d55b1cfc097e846dc679c0763e93b0b4ac0183 | 0065121824bbc904ad615c61ea13a467f21044dd | True | True |
| CL-A | 97d55b1cfc097e846dc679c0763e93b0b4ac0183 | 0065121824bbc904ad615c61ea13a467f21044dd | True | True |
| CL-REG | f74e2e4e2264675005d9d0307f671cc24bba7b61 | 0065121824bbc904ad615c61ea13a467f21044dd | True | True |
| CL-W2 | f74e2e4e2264675005d9d0307f671cc24bba7b61 | 0065121824bbc904ad615c61ea13a467f21044dd | True | True |
| CL-0@1 | f74e2e4e2264675005d9d0307f671cc24bba7b61 | 0065121824bbc904ad615c61ea13a467f21044dd | True | True |
| CL-A@1 | f74e2e4e2264675005d9d0307f671cc24bba7b61 | 0065121824bbc904ad615c61ea13a467f21044dd | True | True |
| CL-FRAME | f74e2e4e2264675005d9d0307f671cc24bba7b61 | 0065121824bbc904ad615c61ea13a467f21044dd | True | True |

The source audit separately records a training-only dataset/cache-staging difference; evaluated native inference code and adapter checks match. Pinned source and asset identities, as recorded during setup:

```json
{
  "fasterwam": {
    "url": "https://github.com/hustvl/FasterWAM/blob/main/configs/sim_libero_plus.yaml",
    "revision": "83667817df0d4f823f39d90700e61ea2f432ac45"
  },
  "paper": "https://arxiv.org/html/2608.04404v1",
  "plus": {
    "repo": "https://github.com/sylvestf/LIBERO-plus",
    "revision": "4976dc30028e805ff8094b55501d532c48fec182",
    "asset_hf_revision": "dd2bd61b7d9a6fef1abc52d606e983b41886a149",
    "archive_sha256": "96764a4bfbdaea98d4411598caeab235458318fe0f549611b93d1a323027b3cf"
  },
  "pro": {
    "repo": "https://github.com/hongyuxiyohung/LIBERO-PRO",
    "revision": "e235a970758ecd78ecbb585fca6b915a8d44d18d",
    "asset_hf_revision": "c86fc3b8293185a6f373677018ff3e37f8391602"
  }
}
```

## Validation, setup, and limitations

Setup records show seven new evaluator tests and 27 existing campaign tests passing. Tests cover official subset matching, duplicate/missing result rejection, checkpoint and state identity, incomplete-group recovery, published settling/replanning behavior, speed-profile eligibility, and Pro instruction routing. All seven native model source/adapter audits passed.

PLUS simulator audit: **1506/1506 tasks passed** construction, reset, selected initial state 0, rendering both cameras, and one dummy step. This setup smoke check is not a policy-success guarantee.

PRO simulator audit: **80/80 tasks passed** construction, reset, selected initial state 0, rendering both cameras, and one dummy step. This setup smoke check is not a policy-success guarantee.

Simulator smoke checks physically exercise initial state 0. Full scored Pro runs exercise all requested indices. Exploratory policy smoke and speed calibration are unscored. The final Pro smoke artifact is linked below.

At Markdown generation, all 16,142 CSV episode records were checked for unique model/benchmark/task/episode identity, seed 42, matching initial-state index, expected checkpoint hash, per-job episode totals, and success totals. Category and suite totals were reconciled with all job summaries and results.json. The initial table-generation check did not rehash video files; the separate fresh integrity audit below records the stronger checks completed afterward.

GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 580.178.04. Python: `/storage/anhdh35/miniconda3/envs/loopwam-libero-eval/bin/python`. Full package freeze, isolated dependencies, and downloaded system-package hashes are in `protocol_audit.json`.

- ImageMagick6 PNG coder module path configured for native Plus motion blur; no perturbation implementation changed.

- Wand0.6.13 and scikit-image0.25.2 plus dependencies installed in isolated python_deps; core evaluation environment unchanged.

- Current official Pro uses liberopro.liberopro namespace and omits convenient swap/task benchmark registrations; use official base task order and the actual published matching BDDL/state files with native Pro environments.

- Pro uses native liberopro.liberopro environments; standard LIBERO remains available solely for existing CL image/quaternion utility imports. No Pro perturbations are implemented through standard LIBERO environments.

- Final instruction audit keeps original catalogue language for Position; only Task receives updated BDDL goal language. Prior exploratory smoke with BDDL wording for Position is unscored and excluded.

## Logs, recovery, and artifacts

Episode logs include task, suite, perturbation, instruction, seed, state index/hash, checkpoint hash, success, policy steps, query count, timings, video path/hash, and query-log path/hash. Every query records the predicted action hash. Completed task groups are saved atomically; interrupted groups restart at episode 0 to preserve RNG sequencing. Runtime manifests identify source, task plans, model/benchmark revisions, cached text, and checkpoint provenance.

- [report.html](report.html)

- [results.json](results.json)

- [run_summary.csv](run_summary.csv)

- [category_results.csv](category_results.csv)

- [suite_results.csv](suite_results.csv)

- [episode_details.csv](episode_details.csv)

- [protocol_audit.json](protocol_audit.json)

- [queue_plan.json](queue_plan.json)

- [queue_status.json](queue_status.json)

- [plus_tasks.json](plus_tasks.json)

- [pro_tasks.json](pro_tasks.json)

- [model_source_audit.json](model_source_audit.json)

- [plus_simulator_audit.json](plus_simulator_audit.json)

- [pro_simulator_audit.json](pro_simulator_audit.json)

- [pro_assets_receipt.json](pro_assets_receipt.json)

- [text_provenance.json](text_provenance.json)

- [final_pro_smoke.json](final_pro_smoke.json)

- [evaluator_source.tar.gz](evaluator_source.tar.gz)

- [source_bundle.json](source_bundle.json)

`queue_plan.json` retains initial pending statuses by design; `queue_status.json` records final execution status. Bulk videos, query traces, manifests, summaries, calibration artifacts, and runner logs are under `/data14t/anhdh35/loopwam_robustness/`. Orchestration code and queue state are under `/storage/anhdh35/LoopWAM-robustness-eval/`.

### Input hashes for this Markdown snapshot

| Input | SHA-256 |
| --- | --- |
| results.json | 34df4507814d8d5e43bd4c0aa180fe6f877d0735a3e061603914305aef2d64a8 |
| queue_status.json | 1ab8ab7c3adfb27a8ea71bc5cdf4a36277f97a5baf5aef9760eb7cc5f6a45d6e |
| episode_details.csv | 346cc0215ffdab28f29a5e27de58dff5e25c6e70e9c0e9174e1058eea882c648 |
| category_results.csv | e441d6d705b93470991ca987d10b693ec702060b624f1fd53c8c924d3386533f |
| suite_results.csv | 43f00e870631fb1fb93ac63564827683cf7491cf7721b37bd0ac0cf7b530433a |
| protocol_audit.json | ebf1bb376a3738458311c17123180a7e717925e00f7687faf8be3c5b39a62e4c |
## Extended code and artifact audit

This section expands the report from the frozen task manifests, individual scored episode records, runtime manifests, and the evaluator code. It distinguishes source inspection, recorded setup checks, fresh artifact integrity checks, and empirical results. No evaluation settings or scored outcomes were modified while writing this report. Percentages use decimal rounding to the nearest 0.01 percentage point, with ties rounded upward; specifically 53/800 = 6.625%, displayed as 6.63%. Some existing machine-generated HTML/CSV displays may use a different tie-rounding convention; integer successes are authoritative.

### Fresh final artifact verification

Completed **2026-10-16 19:45:21 UTC+07**. Result: **PASS** for all **16,142 scored episodes**, **538,591 policy queries**, and **35,330 unique files checked by SHA-256**. The checks read saved artifacts and did not launch any policy evaluation.

| Artifact type | Files rehashed | Total bytes read |
| --- | --- | --- |
| task_plan | 2 | 2,379,283 |
| source | 193 | 12,054,908 |
| bddl | 931 | 2,709,776 |
| initial_states | 340 | 2,603,792 |
| video | 16,142 | 6,731,883,924 |
| query_log | 16,142 | 80,295,122 |
| text_embedding | 1,579 | 1,660,837,031 |
| source_bundle | 1 | 24,046 |

The verifier reconciles CSV rows against atomically committed group JSON and individual episode JSON, requires exact expected task/episode coverage, checks seed and initial-state index/hash, instruction, suite/category, checkpoint hash, step caps, query counts, sequential query indices, finite query times, per-job success totals, and summary identity. Referenced source files must still match each run manifest. Every saved video and query trace is rehashed, as are task plans, BDDL/state files, required cached text, and the archived evaluator source bundle.

The audit does not reload large checkpoint tensors or replay simulations. A state-array hash is matched against the frozen manifest, while the complete state file is rehashed; selected arrays are not newly deserialized in this post-run audit. Action traces store SHA-256 hashes of the complete denormalized, gripper-converted 32×7 action chunks plus query times, not the full numeric action arrays. This supports integrity and exact calibration comparison but does not reconstruct the actions from hashes. Correct artifact identity is evidence of what ran, not proof of benchmark-policy validity in every respect.

- [Machine-readable verification receipt](final_artifact_verification.json)
- [Verification log](final_artifact_verification.log)
- [Read-only verifier source](verify_artifacts.py)

The seven current evaluator regression tests passed again during this report review. The additional 27 existing campaign tests are setup-time evidence recorded in the protocol audit; they were not rerun for this documentation-only update.

### FasterWAM protocol correspondence

| Aspect | Released FasterWAM reference | This CL campaign |
| --- | --- | --- |
| Plus selection | 15% per suite, ceil, sampling seed 42 | Same pinned sampler and task catalogue; explicit selected list included |
| Plus trials / states | One trial per selected task | One episode, state index 0 |
| Settling / replan | 30 settling steps; execute ten actions per query | Same published rollout body and settings |
| Action ensembling / predicted video | Disabled in referenced config | Disabled |
| Policy implementation | FasterWAM checkpoint and inference callback | Native CL checkpoint and callback, documented separately |
| Hardware scheduling | Released config has eight GPUs, two task processes per GPU and chunks of 20 tasks | One shared GPU, four independent workers selected by calibration |
| RNG scope | Seeding per manager process chunk | Seeding per independent task group; environment/query seed 42 |
| Pro applicability | Referenced configuration is for Plus | Separate user-requested Position and Task evaluation with official Pro caps |

This correspondence explains why the report uses “FasterWAM released-config Plus protocol” rather than claiming bitwise reproduction of paper trajectories. A source snapshot can identify the exact experiment even when upstream `main` later changes.

### Model architecture differences

| Model | Video loops | Action loops | Memory tokens | Memory source | Memory write | History-frame setting | Backbone parameters* | Memory parameters* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CL-0 | 4 | 4 | 0 | none | none | 0 | 584536135 | 0 |
| CL-A | 4 | 4 | 16 | learned | loop | 0 | 584536135 | 31008 |
| CL-REG | 4 | 4 | 16 | reset | loop | 0 | 584536135 | 31008 |
| CL-W2 | 4 | 4 | 16 | learned | external | 0 | 584536135 | 37818144 |
| CL-0@1 | 4 | 1 | 0 | none | none | 0 | 584536135 | 0 |
| CL-A@1 | 4 | 1 | 16 | learned | loop | 0 | 584536135 | 31008 |
| CL-FRAME | 4 | 4 | 0 | none | none | 3 | 584536135 | 288 |

*Parameter counts above are values recorded in the released training metadata; they were not freshly recomputed from tensors for this report and do not claim to count the separately loaded text encoder/VAE. All configurations retain four video loops. The `@1` models use one action loop, not one denoising step: all use ten denoising steps per policy query. Checkpoint paths and SHA-256 hashes are listed earlier.

CL-FRAME uses a four-entry deque of query images. On query k, the oldest available image is supplied as the history image; after warm-up this is query k−3. At the first three queries the oldest available early-episode frame is reused. This history is in policy-query units, not simulator-step units. The deque and memory state are reset before each episode. Learned-memory models receive the native returned state on the next query; CL-REG uses the checkpoint’s native reset-register semantics. No external memory rule is substituted.

| Model | HF repository | HF revision | Checkpoint repository path | Bytes |
| --- | --- | --- | --- | --- |
| CL-0 | anhdao69/ChronoLoop | fea8b7334294ce67058c44105ec858025f3eca28 | CL-0_no-memory_a4/epoch_10/policy.pt | 2338624933 |
| CL-A | anhdao69/ChronoLoop | 8e25a4e7d117c36b50fdad51f83c5c4211d47be5 | CL-A_mem16-learned-loopwrite_a4/epoch_10/policy.pt | 2338751283 |
| CL-REG | anhdao69/ChronoLoop | f45c862e1e45645eee407c7a26d0e9fd816bc872 | CL-REG_mem16-reset-registers_a4/epoch_10/policy.pt | 2338751283 |
| CL-W2 | anhdao69/ChronoLoop | 8706968184639b57fc12410cad4a89dce3f79836 | CL-W2_mem16-learned-external-updater_a4/epoch_10/policy.pt | 2489910121 |
| CL-0@1 | anhdao69/ChronoLoop | 8e25a4e7d117c36b50fdad51f83c5c4211d47be5 | CL-0-at1_no-memory_a1/epoch_10/policy.pt | 2338624933 |
| CL-A@1 | anhdao69/ChronoLoop | 204137bc56b946ff46816a3f64bd268e03d99d69 | CL-A-at1_mem16-learned-loopwrite_a1/epoch_10/policy.pt | 2338751283 |
| CL-FRAME | anhdao69/ChronoLoop | cf52c6364579e45408150cd5f526b573d42eef4d | CL-FRAME_history-frame-k3_a4/epoch_10/policy.pt | 2338626793 |

### Exact episode execution and terminal semantics

1. Create the native environment with the selected BDDL and both camera resolutions set to 256×256; call `env.seed(42)` once for the task group.
2. Reset model memory/history before the episode. The published FasterWAM rollout calls `env.reset()` followed by `env.set_init_state(states[episode_index])`.
3. Execute 30 dummy actions `[0, 0, 0, 0, 0, 0, -1]`. These steps do not query the policy.
4. When the pending action list is empty, process current observations and call native `chrono_infer_action` with seed 42 and ten inference steps.
5. Validate a finite 32×7 action chunk, denormalize it, apply the matched gripper conversion, and enqueue its first ten actions.
6. Before each policy action, retain the current two-camera images for the rollout video; step the environment once with the next queued action.
7. Stop on the native environment’s `done` signal or the policy-step allowance. Score the returned `bool(done)`.
8. Save video, query trace, and episode metadata; commit the task group only after all its requested episodes pass integrity checks.

The Pro base-domain `step` assigns `done = self._check_success()`. Thus `done` in this inspected implementation is the task success predicate, rather than an arbitrary generic timeout flag. The outer evaluator imposes the time budget. The executed FasterWAM loop continues through the fixed settling period even if a settling step returns `done`; early termination is tested after policy actions. This is an explicitly preserved behavior of the published rollout, and differs from the unused helper `run_episode` in the CL adapter script, which would terminate during settling. Merely seeing that helper in the source is not evidence that this campaign called it.

`policy_steps` equals the number of saved pre-action frames, excluding settling. Consequently each complete row must satisfy `queries = ceil(policy_steps / 10)` and `1 <= policy_steps <= suite cap`. The final partial action block can contain fewer than ten executed actions. Predicted future-video visualization and action ensembling are disabled. Saved rollout videos show observed frames; they are not future-video predictions, and the final post-terminal frame is not added by this loop.

### Image, state, action, and text path

Camera keys are `agentview_image` and `robot0_eye_in_hand_image`. Both are rotated by reversing their height and width axes. RGB values are converted to floating point, divided by 255, antialiased-resized, concatenated camera-first then wrist-camera horizontally, and mapped to [−1, 1]. The eight-dimensional state concatenates end-effector position (3), quaternion-to-axis-angle orientation (3), and finger joint positions (2). The original dataset min/max normalizer processes state; action denormalization applies `(action - offset) / scale`. The current observation normalization and text reader come from the trained `ObservationAdapter`.

The executed `PolicyAdapter.matched_actions` applies `sign(-(2*g - 1))` to the denormalized gripper channel: g < 0.5 maps to +1 (closed), g > 0.5 to −1 (open), and exactly 0.5 to 0. It requires a finite 32×7 array. Another method in the observation-adapter source maps the midpoint differently, but that method is not the action converter used by this campaign.

Each query builds a new CPU Torch generator with seed 42 for its initial action noise, then transfers that noise to the model device. The query seed is constant, not 42 plus query number. The text cache uses the task’s full prompted instruction, a verified UMT5 encoder/tokenizer, batch-one BF16 encoding, and length-128 context/mask. At inference the cache context is transferred as FP32 before the native model’s own conversions. There are 1,579 required prompt-cache entries. Original training cache bytes were unavailable according to the checkpoint verification metadata; embeddings were regenerated with the verified encoder and original online encoding path. Therefore matching the encoder/procedure is established, while byte-for-byte identity to unavailable original training embeddings is not asserted.

| Shared asset | SHA-256 |
| --- | --- |
| Normalization | d5ad351eb3272f416d34082e8af5483ba1118ceca355a37d9f268fdb269fb3e4 |
| VAE | 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981 |
| UMT5 encoder | 7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d |
| Tokenizer spiece.model | e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458 |
| Tokenizer tokenizer_config.json | ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05 |
| Tokenizer special_tokens_map.json | 7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d |
| Tokenizer tokenizer.json | 6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b |

### Plus catalogue selection and coverage

| Suite | Full catalogue size | Selected count | Actual selected fraction |
| --- | --- | --- | --- |
| libero_10 | 2519 | 378 | 15.01% |
| libero_goal | 2591 | 389 | 15.01% |
| libero_spatial | 2402 | 361 | 15.03% |
| libero_object | 2518 | 378 | 15.01% |

The selected fraction slightly exceeds 15% where rounding upward is required. Sampling is over each suite’s full task catalogue, not independently over each perturbation category or base task. The sorted selected task indices determine execution order. Because suite/category ordering is structured, partial SRs seen during execution were not unbiased estimates of the final SR; all figures in the final tables now use complete runs.

| Suite | Background Textures | Robot Initial States | Camera Viewpoints | Language Instructions | Sensor Noise | Objects Layout | Light Conditions | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| libero_10 | 54 | 31 | 68 | 58 | 77 | 46 | 44 | 378 |
| libero_goal | 53 | 64 | 53 | 68 | 59 | 62 | 30 | 389 |
| libero_spatial | 38 | 44 | 60 | 66 | 55 | 60 | 38 | 361 |
| libero_object | 40 | 59 | 59 | 52 | 66 | 52 | 50 | 378 |

### Pro Position versus Task instruction audit

The following table lists all 40 base-task instruction pairs. Position uses the original instruction; Task uses the perturbed instruction. A repeated filename or task name does not mean the Task goal is unchanged. Each row contributes ten Position episodes and ten Task episodes per model.

| Suite | Task index | Position instruction | Task instruction |
| --- | --- | --- | --- |
| libero_10 | 0 | put both the alphabet soup and the tomato sauce in the basket | put both the cream cheese and the tomato sauce in the basket |
| libero_10 | 1 | put both the cream cheese box and the butter in the basket | put both the alphabet soup and the butter in the basket |
| libero_10 | 2 | turn on the stove and put the moka pot on it | turn on the stove and put the pan on it |
| libero_10 | 3 | put the black bowl in the bottom drawer of the cabinet and close it | put the bottle in the bottom drawer of the cabinet and close it |
| libero_10 | 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | put the yellow and white mug on the left plate and put the white mug on the right plate |
| libero_10 | 5 | pick up the book and place it in the back compartment of the caddy | pick up the cup and place it in the back compartment of the caddy |
| libero_10 | 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | put the red mug on the plate and put the chocolate pudding to the right of the plate |
| libero_10 | 7 | put both the alphabet soup and the cream cheese box in the basket | put both the ketchup and the cream cheese box in the basket |
| libero_10 | 8 | put both moka pots on the stove | put the left moka pot on the stove |
| libero_10 | 9 | put the yellow and white mug in the microwave and close it | put the white mug in the microwave and close it |
| libero_goal | 0 | open the middle drawer of the cabinet | open the bottom drawer of the cabinet |
| libero_goal | 1 | put the bowl on the stove | put the plate on the stove |
| libero_goal | 2 | put the wine bottle on top of the cabinet | put the wine bottle in the bowl |
| libero_goal | 3 | open the top drawer and put the bowl inside | open the top layer of the drawer and put the cream cheese inside |
| libero_goal | 4 | put the bowl on top of the cabinet | put the plate on the top of the drawer |
| libero_goal | 5 | push the plate to the front of the stove | push the cream cheese to the front of the stove |
| libero_goal | 6 | put the cream cheese in the bowl | put the wine bottle in the bowl |
| libero_goal | 7 | turn on the stove | turn off the stove |
| libero_goal | 8 | put the bowl on the plate | put the wine bottle on the plate |
| libero_goal | 9 | put the wine bottle on the rack | put the cream cheese on the rack |
| libero_spatial | 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | pick the akita black bowl not between the plate and the ramekin and place it on the plate |
| libero_spatial | 1 | pick up the black bowl next to the ramekin and place it on the plate | pick the akita black bowl next to the cookie box and place it on the plate |
| libero_spatial | 2 | pick up the black bowl from table center and place it on the plate | pick the akita black bowl next to the plate and place it on the plate |
| libero_spatial | 3 | pick up the black bowl on the cookie box and place it on the plate | pick the akita black bowl on the top of the cabinet and place it on the plate |
| libero_spatial | 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | pick the akita black bowl on the top of the wooden cabinet and place it on the plate |
| libero_spatial | 5 | pick up the black bowl on the ramekin and place it on the plate | pick the akita black bowl on the cookie box and place it on the plate |
| libero_spatial | 6 | pick up the black bowl next to the cookie box and place it on the plate | pick the akita black bowl on the stove and place it on the plate |
| libero_spatial | 7 | pick up the black bowl on the stove and place it on the plate | pick the akita black bowl on the top of the cabinet and place it on the plate |
| libero_spatial | 8 | pick up the black bowl next to the plate and place it on the plate | pick the akita black bowl next to the ramekin and place it on the plate |
| libero_spatial | 9 | pick up the black bowl on the wooden cabinet and place it on the plate | pick the akita black bowl on the stove and place it on the plate |
| libero_object | 0 | pick up the alphabet soup and place it in the basket | pick the cream cheese and place it in the basket |
| libero_object | 1 | pick up the cream cheese and place it in the basket | pick the alphabet soup and place it in the basket |
| libero_object | 2 | pick up the salad dressing and place it in the basket | pick the tomato sauce and place it in the basket |
| libero_object | 3 | pick up the bbq sauce and place it in the basket | pick the ketchup and place it in the basket |
| libero_object | 4 | pick up the ketchup and place it in the basket | pick the milk and place it in the basket |
| libero_object | 5 | pick up the tomato sauce and place it in the basket | pick the bbq sauce and place it in the basket |
| libero_object | 6 | pick up the butter and place it in the basket | pick the orange juice and place it in the basket |
| libero_object | 7 | pick up the milk and place it in the basket | pick the butter and place it in the basket |
| libero_object | 8 | pick up the chocolate pudding and place it in the basket | pick the salad dressing and place it in the basket |
| libero_object | 9 | pick up the orange juice and place it in the basket | pick the chocolate pudding and place it in the basket |

### Recorded runtime interpretation and measured query costs

Policy-query time is measured around the native observation adapter, inference, action denormalization, and transfer to CPU. The CPU transfer synchronizes returned action data, but this is not a CUDA-event kernel benchmark. Mean query time below is summed recorded query seconds divided by query count; it includes contention from concurrent workers. Rollout wall time includes simulation; job wall time includes model loading and calibration. None of these is a training-speed measurement.

| Model | Benchmark | Queries | Mean query ms | Scored episodes/hour | Mean policy steps | Failed episodes at cap |
| --- | --- | --- | --- | --- | --- | --- |
| CL-0 | plus | 51459 | 403.96 | 461.1 | 339.70 | 798 |
| CL-A | plus | 51119 | 341.37 | 509.7 | 337.36 | 787 |
| CL-REG | plus | 51127 | 332.63 | 521.2 | 337.34 | 785 |
| CL-W2 | plus | 51035 | 429.14 | 444.0 | 336.80 | 777 |
| CL-0@1 | plus | 51977 | 211.54 | 580.0 | 343.19 | 817 |
| CL-A@1 | plus | 51816 | 210.07 | 569.8 | 342.21 | 813 |
| CL-FRAME | plus | 51034 | 547.83 | 392.8 | 336.85 | 787 |
| CL-0 | pro | 25602 | 585.07 | 529.6 | 319.71 | 742 |
| CL-A | pro | 25570 | 578.14 | 537.3 | 319.30 | 740 |
| CL-REG | pro | 25549 | 578.44 | 536.8 | 319.06 | 740 |
| CL-W2 | pro | 25621 | 577.32 | 533.8 | 319.99 | 747 |
| CL-0@1 | pro | 25575 | 271.25 | 866.3 | 319.36 | 744 |
| CL-A@1 | pro | 25548 | 271.55 | 864.6 | 319.01 | 740 |
| CL-FRAME | pro | 25559 | 631.56 | 506.3 | 319.17 | 738 |

A step-cap failure describes the terminal behavior recorded by the evaluator; it does not diagnose why the policy failed. GPU occupancy and sharing changed between runs, so small wall-time differences cannot be attributed solely to architecture.

### Aggregation choices and paired comparison

Plus headline SR weights every sampled episode equally. Equal weighting of four suite SRs or seven category SRs gives a different metric because their episode counts differ. Pro gives each task ten episodes and each suite/perturbation 100 episodes, so its combined SR equals the mean of its eight equally sized suite/perturbation cells. The table below reports alternate averages explicitly; they must not replace the labelled headline metric silently.

| Model | Plus episode micro SR | Plus equal-suite macro SR | Plus equal-category macro SR |
| --- | --- | --- | --- |
| CL-0 | 47.01% | 47.16% | 48.61% |
| CL-A | 47.74% | 47.85% | 49.31% |
| CL-REG | 47.88% | 48.01% | 49.40% |
| CL-W2 | 48.41% | 48.52% | 49.82% |
| CL-0@1 | 45.75% | 45.84% | 47.39% |
| CL-A@1 | 46.02% | 46.13% | 47.53% |
| CL-FRAME | 47.74% | 47.89% | 49.16% |

| Model | Benchmark | Matched episodes | Success only in model | Success only in CL-0 | Net SR difference vs CL-0 |
| --- | --- | --- | --- | --- | --- |
| CL-0 | plus | 1506 | 0 | 0 | +0.000 pp |
| CL-A | plus | 1506 | 84 | 73 | +0.730 pp |
| CL-REG | plus | 1506 | 95 | 82 | +0.863 pp |
| CL-W2 | plus | 1506 | 98 | 77 | +1.394 pp |
| CL-0@1 | plus | 1506 | 106 | 125 | -1.262 pp |
| CL-A@1 | plus | 1506 | 93 | 108 | -0.996 pp |
| CL-FRAME | plus | 1506 | 88 | 77 | +0.730 pp |
| CL-0 | pro | 800 | 0 | 0 | +0.000 pp |
| CL-A | pro | 800 | 7 | 5 | +0.250 pp |
| CL-REG | pro | 800 | 6 | 4 | +0.250 pp |
| CL-W2 | pro | 800 | 2 | 7 | -0.625 pp |
| CL-0@1 | pro | 800 | 5 | 7 | -0.250 pp |
| CL-A@1 | pro | 800 | 7 | 5 | +0.250 pp |
| CL-FRAME | pro | 800 | 7 | 3 | +0.500 pp |

Pairing uses the same selected task and initial-state index. It does not imply identical trajectories or identical simulator random streams under every possible execution ordering. No hypothesis test or multi-seed uncertainty estimate is claimed. One Plus success changes its headline score by about 0.0664 percentage points; one Pro success changes combined SR by 0.125 points and a perturbation SR by 0.25 points.

### Interpretation of the low Pro scores

All seven Position scores are 0.25–1.00%, while Task scores are 13.00–14.75%. This pattern is shared across the evaluated CL variants. The logs establish completed simulations, correct requested episode counts, pinned asset identities, recorded language routing, and finite predicted actions. They do not establish whether the low scores arise from distribution shift, model behavior, or an unrecognized integration issue. No independent reference-policy reproduction on the same Pro installation or broad failed-trajectory causal analysis is documented here. Thus these should be cited as results of this pinned implementation and protocol, not as proof that every possible Pro integration issue has been ruled out. The final pre-queue Pro smoke was also 0/8; it validated execution, not performance.

### Reproduction and source map

| File | Responsibility |
| --- | --- |
| /storage/anhdh35/LoopWAM-robustness-eval/protocol.py | Paths, sampler, identity/cap checks, runtime environment, instruction routing |
| /storage/anhdh35/LoopWAM-robustness-eval/prepare_manifest.py | Official task/state selection and hashes |
| /storage/anhdh35/LoopWAM-robustness-eval/reference.py | Load published FasterWAM rollout function bodies via AST |
| /storage/anhdh35/LoopWAM-robustness-eval/runner.py | Model loading, rollout workers, manifests, query/video logging, group recovery |
| /storage/anhdh35/LoopWAM-robustness-eval/queue_run.py | Serial queue, calibration eligibility, VRAM admission, final verification |
| /storage/anhdh35/LoopWAM-campaign-eval/policy_adapter.py | Native inference call, memory/history ownership, action conversion |
| /storage/anhdh35/ChronoLoop-eval/scripts/evaluate_loopwam_libero.py | Imported trained ObservationAdapter |
| /storage/anhdh35/ChronoLoop-eval/src/fastwam/models/wan22/chronoloop.py | Native architecture and chrono_infer_action |
| /storage/anhdh35/LoopWAM-robustness-eval/report.py | HTML, JSON and CSV result aggregation |

Read-only fresh integrity audit command (does not rerun simulations):

```bash
python /storage/anhdh35/LoopWAM_NT/evaluation_report/robustness/verify_artifacts.py
```

Existing CPU regression tests, rerun successfully during this report review:

```bash
cd /storage/anhdh35/LoopWAM-robustness-eval
/storage/anhdh35/miniconda3/envs/loopwam-libero-eval/bin/python -m unittest test_protocol test_rollout test_selection -v
```

The saved run manifest for each job contains its exact CLI arguments and code hashes. For a deliberate independent rerun, use a new output directory and the environment returned by `protocol.environment(benchmark)`, then invoke `runner.py` with the benchmark, exact checkpoint path/hash, four workers and two threads. Reusing completed output would resume/verify existing results rather than produce independent episodes. A rerun is additional compute and has not been launched as part of writing this report. Preserve the pinned source, assets, normalization, and text cache; rerunning dependency setup or fetching latest upstream code would create a different experiment.

### All Pro task results

Each cell is successes out of 10. Task IDs are the frozen manifest IDs. The instruction-pair table above supplies the full goal text for each suite/index.

| Task ID | Category | CL-0 | CL-A | CL-REG | CL-W2 | CL-0@1 | CL-A@1 | CL-FRAME |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| libero_10_swap__0000__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_swap__0001__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_swap__0002__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_swap__0003__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_swap__0004__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_swap__0005__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_swap__0006__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_swap__0007__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_swap__0008__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_swap__0009__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_task__0000__seed42 | Task | 1/10 | 3/10 | 2/10 | 0/10 | 1/10 | 2/10 | 4/10 |
| libero_10_task__0001__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_task__0002__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_task__0003__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_task__0004__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_task__0005__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_task__0006__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_task__0007__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_10_task__0008__seed42 | Task | 8/10 | 6/10 | 8/10 | 8/10 | 8/10 | 7/10 | 7/10 |
| libero_10_task__0009__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_swap__0000__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_swap__0001__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_swap__0002__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_swap__0003__seed42 | Position | 3/10 | 3/10 | 2/10 | 1/10 | 2/10 | 3/10 | 3/10 |
| libero_goal_swap__0004__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_swap__0005__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_swap__0006__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_swap__0007__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_swap__0008__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_swap__0009__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_task__0000__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_task__0001__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_task__0002__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_task__0003__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_task__0004__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_task__0005__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 1/10 | 1/10 | 0/10 |
| libero_goal_task__0006__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_goal_task__0007__seed42 | Task | 8/10 | 9/10 | 10/10 | 9/10 | 9/10 | 10/10 | 9/10 |
| libero_goal_task__0008__seed42 | Task | 1/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 2/10 |
| libero_goal_task__0009__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_swap__0000__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_swap__0001__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_swap__0002__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_swap__0003__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_swap__0004__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_swap__0005__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_swap__0006__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_swap__0007__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_swap__0008__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 1/10 | 0/10 |
| libero_spatial_swap__0009__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_task__0000__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_task__0001__seed42 | Task | 7/10 | 10/10 | 9/10 | 7/10 | 6/10 | 7/10 | 8/10 |
| libero_spatial_task__0002__seed42 | Task | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| libero_spatial_task__0003__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_task__0004__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_task__0005__seed42 | Task | 10/10 | 9/10 | 9/10 | 8/10 | 9/10 | 9/10 | 9/10 |
| libero_spatial_task__0006__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_task__0007__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_task__0008__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_spatial_task__0009__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_swap__0000__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_swap__0001__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_swap__0002__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_swap__0003__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_swap__0004__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_swap__0005__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_swap__0006__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_swap__0007__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_swap__0008__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_swap__0009__seed42 | Position | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_task__0000__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_task__0001__seed42 | Task | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| libero_object_task__0002__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_task__0003__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_task__0004__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_task__0005__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_task__0006__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_task__0007__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_task__0008__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |
| libero_object_task__0009__seed42 | Task | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 |

### All selected Plus task results

This appendix enumerates all 1,506 selected tasks. Each model cell is 1 for success or 0 for failure, from one episode at initial-state index 0. Full BDDL/state paths, hashes, and language are in `plus_tasks.json`; every video/query path is in `episode_details.csv`. The selected task list is provided explicitly so the subset can be reproduced without inferring it from rounded category counts.

<details>
<summary>Expand all 1,506 selected Plus tasks and seven-model outcomes</summary>

| Task ID | Category | Difficulty | Task name | CL-0 | CL-A | CL-REG | CL-W2 | CL-0@1 | CL-A@1 | CL-FRAME |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| libero_10__0009__seed42 | Background Textures | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0011__seed42 | Background Textures | 1 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0015__seed42 | Background Textures | 2 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_27 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_10__0019__seed42 | Background Textures | 2 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_8 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_10__0021__seed42 | Background Textures | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_tb_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0023__seed42 | Background Textures | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_tb_18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0025__seed42 | Background Textures | 3 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_tb_20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0026__seed42 | Background Textures | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_tb_21 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0027__seed42 | Background Textures | 3 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_tb_22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0034__seed42 | Background Textures | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_table_11 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_10__0035__seed42 | Background Textures | 3 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_table_12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0041__seed42 | Background Textures | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_table_18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0046__seed42 | Background Textures | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_table_23 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0049__seed42 | Background Textures | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_table_26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0058__seed42 | Background Textures | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_tb_11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0063__seed42 | Background Textures | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_tb_16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0065__seed42 | Background Textures | 3 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_tb_18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0066__seed42 | Background Textures | 3 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_tb_19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0076__seed42 | Background Textures | 3 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_tb_8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0084__seed42 | Background Textures | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_table_22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0094__seed42 | Background Textures | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_table_6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0110__seed42 | Background Textures | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_table_15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0112__seed42 | Background Textures | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_table_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0125__seed42 | Background Textures | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_tb_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0126__seed42 | Background Textures | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_tb_12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0130__seed42 | Background Textures | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_tb_18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0141__seed42 | Background Textures | 3 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_table_28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0145__seed42 | Background Textures | 2 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_table_1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0160__seed42 | Background Textures | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_table_12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0162__seed42 | Background Textures | 1 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_table_16 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| libero_10__0163__seed42 | Background Textures | 1 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_table_21 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0184__seed42 | Background Textures | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_tb_19 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0186__seed42 | Background Textures | 1 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_tb_21 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0191__seed42 | Background Textures | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_tb_8 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0193__seed42 | Background Textures | 3 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_table_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0196__seed42 | Background Textures | 1 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_table_13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0198__seed42 | Background Textures | 5 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_table_15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0199__seed42 | Background Textures | 2 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_table_16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0213__seed42 | Background Textures | 2 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_table_6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0220__seed42 | Background Textures | 3 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_tb_12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0223__seed42 | Background Textures | 3 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_tb_15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0230__seed42 | Background Textures | 2 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_tb_22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0237__seed42 | Background Textures | 5 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_table_1 | 0 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_10__0238__seed42 | Background Textures | 5 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_table_10 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| libero_10__0241__seed42 | Background Textures | 4 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_table_15 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_10__0248__seed42 | Background Textures | 3 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_table_23 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_10__0250__seed42 | Background Textures | 1 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_table_25 | 1 | 0 | 1 | 1 | 1 | 0 | 1 |
| libero_10__0254__seed42 | Background Textures | 5 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_table_6 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_10__0255__seed42 | Background Textures | 1 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_table_8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0259__seed42 | Background Textures | 4 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_tb_11 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| libero_10__0261__seed42 | Background Textures | 3 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_tb_15 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| libero_10__0265__seed42 | Background Textures | 5 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_tb_2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0268__seed42 | Background Textures | 4 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_tb_22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0280__seed42 | Background Textures | 3 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_table_23 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| libero_10__0291__seed42 | Robot Initial States | 1 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_51 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0300__seed42 | Robot Initial States | 2 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_191 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_10__0332__seed42 | Robot Initial States | 1 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_0_0_100_0_0_initstate_62 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0352__seed42 | Robot Initial States | 3 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_0_0_100_0_0_initstate_362 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0354__seed42 | Robot Initial States | 4 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_0_0_100_0_0_initstate_382 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_10__0391__seed42 | Robot Initial States | 2 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_303 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0393__seed42 | Robot Initial States | 3 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_323 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0405__seed42 | Robot Initial States | 1 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_463 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0407__seed42 | Robot Initial States | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_483 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| libero_10__0410__seed42 | Robot Initial States | 3 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_14 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0414__seed42 | Robot Initial States | 3 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_74 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0415__seed42 | Robot Initial States | 2 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_84 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| libero_10__0416__seed42 | Robot Initial States | 3 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_94 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_10__0422__seed42 | Robot Initial States | 2 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_174 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| libero_10__0425__seed42 | Robot Initial States | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_204 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0446__seed42 | Robot Initial States | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_424 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0488__seed42 | Robot Initial States | 4 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_475 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0492__seed42 | Robot Initial States | 2 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_100_0_0_initstate_116 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0499__seed42 | Robot Initial States | 1 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_100_0_0_initstate_226 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0508__seed42 | Robot Initial States | 4 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_100_0_0_initstate_316 | 0 | 0 | 1 | 1 | 1 | 1 | 0 |
| libero_10__0522__seed42 | Robot Initial States | 5 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_100_0_0_initstate_466 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0528__seed42 | Robot Initial States | 1 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_0_0_initstate_47 | 1 | 0 | 1 | 1 | 0 | 1 | 0 |
| libero_10__0530__seed42 | Robot Initial States | 1 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_0_0_initstate_67 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_10__0570__seed42 | Robot Initial States | 1 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_0_0_initstate_118 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0582__seed42 | Robot Initial States | 3 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_0_0_initstate_248 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0587__seed42 | Robot Initial States | 3 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_0_0_initstate_308 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0603__seed42 | Robot Initial States | 4 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_0_0_initstate_468 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__0615__seed42 | Robot Initial States | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_100_0_0_initstate_249 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| libero_10__0633__seed42 | Robot Initial States | 4 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_100_0_0_initstate_439 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0670__seed42 | Robot Initial States | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_100_0_0_initstate_390 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0678__seed42 | Robot Initial States | 4 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_100_0_0_initstate_470 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0690__seed42 | Camera Viewpoints | 3 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_4_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0693__seed42 | Camera Viewpoints | 2 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_7_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0700__seed42 | Camera Viewpoints | 3 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_14_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0705__seed42 | Camera Viewpoints | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_0_0_100_4_10_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0709__seed42 | Camera Viewpoints | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_16_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_10__0713__seed42 | Camera Viewpoints | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_18_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0716__seed42 | Camera Viewpoints | 1 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_21_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0718__seed42 | Camera Viewpoints | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_22_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0722__seed42 | Camera Viewpoints | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_25_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0740__seed42 | Camera Viewpoints | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_127_0_0_initstate_0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 |
| libero_10__0743__seed42 | Camera Viewpoints | 3 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_130_0_0_initstate_0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |
| libero_10__0747__seed42 | Camera Viewpoints | 1 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_6_8_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0761__seed42 | Camera Viewpoints | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_35_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0763__seed42 | Camera Viewpoints | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_36_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0765__seed42 | Camera Viewpoints | 3 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_37_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0767__seed42 | Camera Viewpoints | 3 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_38_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0775__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_42_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0781__seed42 | Camera Viewpoints | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_131_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0789__seed42 | Camera Viewpoints | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_139_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0790__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_8_2_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0797__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_8_354_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0805__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_48_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0808__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_50_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0809__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_50_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0815__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_54_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0857__seed42 | Camera Viewpoints | 5 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_65_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0860__seed42 | Camera Viewpoints | 5 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_68_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0875__seed42 | Camera Viewpoints | 5 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_75_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0880__seed42 | Camera Viewpoints | 5 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_155_0_0_initstate_0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| libero_10__0882__seed42 | Camera Viewpoints | 5 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_157_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0885__seed42 | Camera Viewpoints | 5 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_160_0_0_initstate_0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 |
| libero_10__0888__seed42 | Camera Viewpoints | 5 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_100_350_8_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0891__seed42 | Camera Viewpoints | 5 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_100_350_352_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0892__seed42 | Camera Viewpoints | 5 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_100_350_354_initstate_0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__0907__seed42 | Camera Viewpoints | 5 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_291_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0909__seed42 | Camera Viewpoints | 5 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_293_0_100_0_0_initstate_0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_10__0918__seed42 | Camera Viewpoints | 5 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_297_15_100_0_0_initstate_0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| libero_10__0921__seed42 | Camera Viewpoints | 5 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_299_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0926__seed42 | Camera Viewpoints | 4 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_165_0_0_initstate_0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| libero_10__0931__seed42 | Camera Viewpoints | 3 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_352_4_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__0940__seed42 | Camera Viewpoints | 5 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_301_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_10__0948__seed42 | Camera Viewpoints | 5 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_305_0_100_0_0_initstate_0 | 1 | 0 | 1 | 1 | 0 | 0 | 1 |
| libero_10__0962__seed42 | Camera Viewpoints | 5 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_312_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_10__0968__seed42 | Camera Viewpoints | 3 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_171_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0975__seed42 | Camera Viewpoints | 3 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_178_0_0_initstate_0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__0976__seed42 | Camera Viewpoints | 3 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_179_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0981__seed42 | Camera Viewpoints | 2 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_354_8_initstate_0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__0987__seed42 | Camera Viewpoints | 1 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_354_358_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__0996__seed42 | Camera Viewpoints | 5 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_319_0_100_0_0_initstate_0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 |
| libero_10__1001__seed42 | Camera Viewpoints | 5 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_321_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| libero_10__1006__seed42 | Camera Viewpoints | 4 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_324_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1010__seed42 | Camera Viewpoints | 5 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_326_0_100_0_0_initstate_0 | 1 | 0 | 1 | 1 | 1 | 0 | 0 |
| libero_10__1013__seed42 | Camera Viewpoints | 5 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_327_15_100_0_0_initstate_0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| libero_10__1025__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_189_0_0_initstate_0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_10__1029__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_100_356_8_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1040__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_333_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1047__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_337_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1048__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_338_0_100_0_0_initstate_0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_10__1055__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_341_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1065__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_195_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1066__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_196_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1067__seed42 | Camera Viewpoints | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_197_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1072__seed42 | Camera Viewpoints | 4 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_100_358_4_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1083__seed42 | Camera Viewpoints | 3 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_346_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1085__seed42 | Camera Viewpoints | 3 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_348_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1090__seed42 | Camera Viewpoints | 2 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_351_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_10__1093__seed42 | Camera Viewpoints | 4 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_352_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1094__seed42 | Camera Viewpoints | 2 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_353_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1105__seed42 | Language Instructions | 2 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_language_5_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1109__seed42 | Language Instructions | 2 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_language_9_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_10__1133__seed42 | Language Instructions | 5 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_language_37_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1161__seed42 | Language Instructions | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_language_40_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1163__seed42 | Language Instructions | 5 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_language_43_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1171__seed42 | Language Instructions | 3 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_language_6_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1180__seed42 | Language Instructions | 3 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_language_17_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1181__seed42 | Language Instructions | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_language_18_view_0_0_100_0_0_initstate_0 | 0 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__1191__seed42 | Language Instructions | 3 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_language_31_view_0_0_100_0_0_initstate_0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 |
| libero_10__1196__seed42 | Language Instructions | 1 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_language_37_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1200__seed42 | Language Instructions | 3 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_language_41_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1203__seed42 | Language Instructions | 1 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_language_44_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_10__1208__seed42 | Language Instructions | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_language_1_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__1210__seed42 | Language Instructions | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_language_3_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1214__seed42 | Language Instructions | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_language_7_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_10__1226__seed42 | Language Instructions | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_language_20_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1231__seed42 | Language Instructions | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_language_25_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1234__seed42 | Language Instructions | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_language_28_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |
| libero_10__1240__seed42 | Language Instructions | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_language_34_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1244__seed42 | Language Instructions | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_language_39_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1245__seed42 | Language Instructions | 2 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_language_41_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__1251__seed42 | Language Instructions | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_language_47_view_0_0_100_0_0_initstate_0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1261__seed42 | Language Instructions | 3 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_language_8_view_0_0_100_0_0_initstate_0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| libero_10__1263__seed42 | Language Instructions | 4 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_language_10_view_0_0_100_0_0_initstate_0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1279__seed42 | Language Instructions | 2 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_language_31_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1292__seed42 | Language Instructions | 1 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_language_45_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| libero_10__1298__seed42 | Language Instructions | 3 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_language_5_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1300__seed42 | Language Instructions | 5 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_language_8_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1301__seed42 | Language Instructions | 1 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_language_9_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1304__seed42 | Language Instructions | 4 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_language_14_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__1307__seed42 | Language Instructions | 2 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_language_18_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1310__seed42 | Language Instructions | 1 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_language_22_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1317__seed42 | Language Instructions | 1 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_language_32_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1321__seed42 | Language Instructions | 4 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_language_43_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1323__seed42 | Language Instructions | 1 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_language_48_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1332__seed42 | Language Instructions | 4 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_language_9_view_0_0_100_0_0_initstate_0 | 1 | 0 | 1 | 0 | 1 | 0 | 0 |
| libero_10__1335__seed42 | Language Instructions | 2 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_language_13_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1336__seed42 | Language Instructions | 2 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_language_14_view_0_0_100_0_0_initstate_0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_10__1339__seed42 | Language Instructions | 4 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_language_17_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1343__seed42 | Language Instructions | 1 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_language_21_view_0_0_100_0_0_initstate_0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1354__seed42 | Language Instructions | 4 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_language_32_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1362__seed42 | Language Instructions | 4 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_language_41_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1368__seed42 | Language Instructions | 1 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_language_47_view_0_0_100_0_0_initstate_0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 |
| libero_10__1372__seed42 | Language Instructions | 2 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_language_2_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1375__seed42 | Language Instructions | 1 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_language_5_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1385__seed42 | Language Instructions | 1 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_language_17_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1402__seed42 | Language Instructions | 2 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_language_38_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1403__seed42 | Language Instructions | 5 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_language_41_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1425__seed42 | Language Instructions | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_language_20_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1449__seed42 | Language Instructions | 4 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_language_8_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1454__seed42 | Language Instructions | 3 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_language_13_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1456__seed42 | Language Instructions | 4 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_language_15_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1457__seed42 | Language Instructions | 4 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_language_16_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1465__seed42 | Language Instructions | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_language_26_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1467__seed42 | Language Instructions | 2 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_language_29_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1479__seed42 | Language Instructions | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_language_44_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1481__seed42 | Language Instructions | 2 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_language_48_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1483__seed42 | Language Instructions | 4 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_language_50_view_0_0_100_0_0_initstate_0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_10__1488__seed42 | Sensor Noise | 4 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_0_noise_5 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_10__1491__seed42 | Sensor Noise | 5 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_0_noise_9 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_10__1499__seed42 | Sensor Noise | 4 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_0_noise_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1504__seed42 | Sensor Noise | 1 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_0_noise_24 | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| libero_10__1508__seed42 | Sensor Noise | 4 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_0_noise_28 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| libero_10__1511__seed42 | Sensor Noise | 1 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_0_noise_31 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1512__seed42 | Sensor Noise | 3 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_0_noise_32 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__1513__seed42 | Sensor Noise | 2 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_0_noise_33 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1517__seed42 | Sensor Noise | 3 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_view_0_0_100_0_0_initstate_0_noise_37 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1532__seed42 | Sensor Noise | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_0_0_100_0_0_initstate_0_noise_5 | 1 | 0 | 0 | 1 | 1 | 1 | 0 |
| libero_10__1538__seed42 | Sensor Noise | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_0_0_100_0_0_initstate_0_noise_13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1545__seed42 | Sensor Noise | 4 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_0_0_100_0_0_initstate_0_noise_20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1546__seed42 | Sensor Noise | 1 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_0_0_100_0_0_initstate_0_noise_26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1564__seed42 | Sensor Noise | 3 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_view_0_0_100_0_0_initstate_0_noise_50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1570__seed42 | Sensor Noise | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_6 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_10__1571__seed42 | Sensor Noise | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_8 | 0 | 0 | 1 | 1 | 0 | 0 | 1 |
| libero_10__1572__seed42 | Sensor Noise | 5 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_10 | 1 | 0 | 1 | 1 | 0 | 0 | 1 |
| libero_10__1579__seed42 | Sensor Noise | 5 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1580__seed42 | Sensor Noise | 5 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1585__seed42 | Sensor Noise | 2 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_23 | 1 | 1 | 1 | 1 | 0 | 1 | 0 |
| libero_10__1589__seed42 | Sensor Noise | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_27 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_10__1598__seed42 | Sensor Noise | 2 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_36 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| libero_10__1599__seed42 | Sensor Noise | 2 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_37 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_10__1605__seed42 | Sensor Noise | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_43 | 0 | 0 | 0 | 1 | 1 | 0 | 1 |
| libero_10__1606__seed42 | Sensor Noise | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_44 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| libero_10__1608__seed42 | Sensor Noise | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_46 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| libero_10__1609__seed42 | Sensor Noise | 5 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1611__seed42 | Sensor Noise | 5 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_view_0_0_100_0_0_initstate_0_noise_49 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1633__seed42 | Sensor Noise | 2 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_0_noise_22 | 0 | 0 | 1 | 0 | 1 | 1 | 0 |
| libero_10__1635__seed42 | Sensor Noise | 3 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_0_noise_24 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| libero_10__1640__seed42 | Sensor Noise | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_0_noise_29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1641__seed42 | Sensor Noise | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_0_noise_30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1642__seed42 | Sensor Noise | 3 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_0_noise_31 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1644__seed42 | Sensor Noise | 3 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_0_noise_33 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| libero_10__1653__seed42 | Sensor Noise | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_view_0_0_100_0_0_initstate_0_noise_43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1661__seed42 | Sensor Noise | 2 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_0_noise_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1662__seed42 | Sensor Noise | 1 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_0_noise_2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_10__1667__seed42 | Sensor Noise | 5 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_0_noise_8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1669__seed42 | Sensor Noise | 5 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_0_noise_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1687__seed42 | Sensor Noise | 5 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_0_noise_29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1688__seed42 | Sensor Noise | 5 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_0_noise_30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1692__seed42 | Sensor Noise | 2 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_0_noise_34 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1693__seed42 | Sensor Noise | 2 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_0_noise_35 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1694__seed42 | Sensor Noise | 3 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_0_noise_36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1696__seed42 | Sensor Noise | 4 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_0_noise_38 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1697__seed42 | Sensor Noise | 4 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_view_0_0_100_0_0_initstate_0_noise_39 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1724__seed42 | Sensor Noise | 5 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_100_0_0_initstate_0_noise_18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1738__seed42 | Sensor Noise | 2 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_view_0_0_100_0_0_initstate_0_noise_33 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| libero_10__1755__seed42 | Sensor Noise | 3 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_0_0_initstate_0_noise_1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_10__1758__seed42 | Sensor Noise | 3 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_0_0_initstate_0_noise_4 | 0 | 1 | 0 | 0 | 1 | 0 | 1 |
| libero_10__1760__seed42 | Sensor Noise | 3 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_0_0_initstate_0_noise_6 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| libero_10__1763__seed42 | Sensor Noise | 4 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_0_0_initstate_0_noise_9 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_10__1765__seed42 | Sensor Noise | 3 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_0_0_initstate_0_noise_12 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| libero_10__1767__seed42 | Sensor Noise | 4 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_0_0_initstate_0_noise_14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1768__seed42 | Sensor Noise | 4 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_0_0_initstate_0_noise_15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1777__seed42 | Sensor Noise | 4 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_0_0_initstate_0_noise_28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1789__seed42 | Sensor Noise | 2 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_0_0_initstate_0_noise_41 | 1 | 1 | 1 | 0 | 1 | 0 | 1 |
| libero_10__1792__seed42 | Sensor Noise | 3 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_view_0_0_100_0_0_initstate_0_noise_44 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1803__seed42 | Sensor Noise | 2 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_0_0_initstate_0_noise_7 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1814__seed42 | Sensor Noise | 4 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_0_0_initstate_0_noise_20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1818__seed42 | Sensor Noise | 2 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_0_0_initstate_0_noise_26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1830__seed42 | Sensor Noise | 2 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_0_0_initstate_0_noise_43 | 1 | 1 | 0 | 1 | 1 | 0 | 1 |
| libero_10__1833__seed42 | Sensor Noise | 2 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_0_0_initstate_0_noise_46 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| libero_10__1836__seed42 | Sensor Noise | 3 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_0_0_initstate_0_noise_49 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1837__seed42 | Sensor Noise | 3 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_view_0_0_100_0_0_initstate_0_noise_50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1838__seed42 | Sensor Noise | 4 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_100_0_0_initstate_0_noise_1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1855__seed42 | Sensor Noise | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_100_0_0_initstate_0_noise_19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1857__seed42 | Sensor Noise | 3 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_100_0_0_initstate_0_noise_22 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_10__1860__seed42 | Sensor Noise | 4 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_100_0_0_initstate_0_noise_26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1869__seed42 | Sensor Noise | 3 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_100_0_0_initstate_0_noise_35 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_10__1871__seed42 | Sensor Noise | 4 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_100_0_0_initstate_0_noise_37 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1878__seed42 | Sensor Noise | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_view_0_0_100_0_0_initstate_0_noise_44 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_10__1887__seed42 | Sensor Noise | 4 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_100_0_0_initstate_0_noise_4 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| libero_10__1888__seed42 | Sensor Noise | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_100_0_0_initstate_0_noise_5 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_10__1900__seed42 | Sensor Noise | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_100_0_0_initstate_0_noise_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1913__seed42 | Sensor Noise | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_100_0_0_initstate_0_noise_30 | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| libero_10__1923__seed42 | Sensor Noise | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_view_0_0_100_0_0_initstate_0_noise_40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1951__seed42 | Objects Layout | 1 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_level3_sample3 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| libero_10__1953__seed42 | Objects Layout | 1 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_level4_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__1962__seed42 | Objects Layout | 1 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_add_16 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1975__seed42 | Objects Layout | 1 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_level3_sample2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1976__seed42 | Objects Layout | 1 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_level4_sample1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__1995__seed42 | Objects Layout | 2 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_add_27 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2000__seed42 | Objects Layout | 1 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_add_7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2006__seed42 | Objects Layout | 1 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_level1_sample4 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| libero_10__2013__seed42 | Objects Layout | 1 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_level3_sample4 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2025__seed42 | Objects Layout | 2 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_add_20 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2028__seed42 | Objects Layout | 1 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_add_24 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__2044__seed42 | Objects Layout | 1 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_level5_sample2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2047__seed42 | Objects Layout | 1 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_add_27 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2052__seed42 | Objects Layout | 1 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_level3_sample3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2055__seed42 | Objects Layout | 2 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_level4_sample4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2082__seed42 | Objects Layout | 3 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_level3_sample1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__2084__seed42 | Objects Layout | 2 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_level3_sample3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2085__seed42 | Objects Layout | 2 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_level3_sample4 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| libero_10__2087__seed42 | Objects Layout | 4 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_level4_sample2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2090__seed42 | Objects Layout | 2 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_level5_sample2 | 0 | 1 | 1 | 0 | 1 | 0 | 1 |
| libero_10__2092__seed42 | Objects Layout | 5 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_level5_sample4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2094__seed42 | Objects Layout | 1 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_add_11 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2095__seed42 | Objects Layout | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_add_13 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| libero_10__2101__seed42 | Objects Layout | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_add_21 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2104__seed42 | Objects Layout | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_add_24 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2117__seed42 | Objects Layout | 1 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_level2_sample3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2138__seed42 | Objects Layout | 1 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_add_23 | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
| libero_10__2141__seed42 | Objects Layout | 2 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_add_26 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__2142__seed42 | Objects Layout | 1 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_add_27 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2151__seed42 | Objects Layout | 2 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_level3_sample2 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2152__seed42 | Objects Layout | 1 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_level3_sample3 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| libero_10__2160__seed42 | Objects Layout | 3 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_level5_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2167__seed42 | Objects Layout | 1 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_add_17 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_10__2171__seed42 | Objects Layout | 2 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_add_22 | 0 | 1 | 1 | 1 | 0 | 1 | 0 |
| libero_10__2176__seed42 | Objects Layout | 1 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_add_4 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| libero_10__2187__seed42 | Objects Layout | 2 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_level3_sample2 | 0 | 1 | 0 | 1 | 1 | 0 | 1 |
| libero_10__2191__seed42 | Objects Layout | 3 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_level4_sample2 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| libero_10__2192__seed42 | Objects Layout | 3 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_level4_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2200__seed42 | Objects Layout | 3 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_add_12 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2207__seed42 | Objects Layout | 2 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_add_19 | 0 | 1 | 1 | 1 | 0 | 1 | 0 |
| libero_10__2213__seed42 | Objects Layout | 3 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_add_24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2216__seed42 | Objects Layout | 4 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_add_27 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| libero_10__2221__seed42 | Objects Layout | 1 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_add_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2234__seed42 | Objects Layout | 3 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_level3_sample2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2235__seed42 | Objects Layout | 2 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_level3_sample3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2236__seed42 | Objects Layout | 3 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_level3_sample4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2255__seed42 | Light Conditions | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_light_19 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2260__seed42 | Light Conditions | 2 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_light_25 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2265__seed42 | Light Conditions | 2 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_light_3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2278__seed42 | Light Conditions | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_light_46 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_10__2279__seed42 | Light Conditions | 5 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_light_47 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__2281__seed42 | Light Conditions | 4 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_light_49 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2286__seed42 | Light Conditions | 3 | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_light_9 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_10__2294__seed42 | Light Conditions | 3 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_light_16 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__2310__seed42 | Light Conditions | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_light_31 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2317__seed42 | Light Conditions | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_light_38 | 1 | 1 | 1 | 0 | 1 | 0 | 1 |
| libero_10__2320__seed42 | Light Conditions | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_light_40 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2321__seed42 | Light Conditions | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_light_41 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_10__2325__seed42 | Light Conditions | 5 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_light_45 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2326__seed42 | Light Conditions | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_light_46 | 0 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_10__2335__seed42 | Light Conditions | 4 | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_light_9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2336__seed42 | Light Conditions | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_light_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2338__seed42 | Light Conditions | 3 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_light_11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2341__seed42 | Light Conditions | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_light_14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2347__seed42 | Light Conditions | 5 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_light_2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2370__seed42 | Light Conditions | 4 | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_light_5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2374__seed42 | Light Conditions | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_light_11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2375__seed42 | Light Conditions | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_light_13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2378__seed42 | Light Conditions | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_light_18 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2383__seed42 | Light Conditions | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_light_27 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2384__seed42 | Light Conditions | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_light_29 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2387__seed42 | Light Conditions | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_light_33 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2388__seed42 | Light Conditions | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_light_34 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2397__seed42 | Light Conditions | 5 | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_light_49 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2399__seed42 | Light Conditions | 2 | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_light_15 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2409__seed42 | Light Conditions | 5 | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_light_40 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2416__seed42 | Light Conditions | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_light_10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2417__seed42 | Light Conditions | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_light_14 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2418__seed42 | Light Conditions | 2 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_light_15 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2425__seed42 | Light Conditions | 1 | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_light_38 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2440__seed42 | Light Conditions | 3 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_light_18 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2449__seed42 | Light Conditions | 3 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_light_27 | 0 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_10__2450__seed42 | Light Conditions | 3 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_light_28 | 0 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_10__2462__seed42 | Light Conditions | 2 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_light_5 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
| libero_10__2468__seed42 | Light Conditions | 4 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_light_13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2479__seed42 | Light Conditions | 5 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_light_30 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_10__2485__seed42 | Light Conditions | 5 | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_light_37 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_10__2497__seed42 | Light Conditions | 3 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_light_1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2504__seed42 | Light Conditions | 2 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_light_20 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_10__2516__seed42 | Light Conditions | 2 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_light_41 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__0005__seed42 | Background Textures | 4 | open_the_middle_drawer_of_the_cabinet_table_16 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_goal__0014__seed42 | Background Textures | 2 | open_the_middle_drawer_of_the_cabinet_table_28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0015__seed42 | Background Textures | 2 | open_the_middle_drawer_of_the_cabinet_table_4 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_goal__0025__seed42 | Background Textures | 3 | open_the_middle_drawer_of_the_cabinet_tb_5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0030__seed42 | Background Textures | 4 | open_the_top_drawer_and_put_the_bowl_inside_table_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0033__seed42 | Background Textures | 1 | open_the_top_drawer_and_put_the_bowl_inside_table_13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0038__seed42 | Background Textures | 2 | open_the_top_drawer_and_put_the_bowl_inside_table_18 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| libero_goal__0044__seed42 | Background Textures | 2 | open_the_top_drawer_and_put_the_bowl_inside_table_23 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0045__seed42 | Background Textures | 2 | open_the_top_drawer_and_put_the_bowl_inside_table_24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0057__seed42 | Background Textures | 2 | open_the_top_drawer_and_put_the_bowl_inside_tb_11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0062__seed42 | Background Textures | 4 | open_the_top_drawer_and_put_the_bowl_inside_tb_18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0065__seed42 | Background Textures | 3 | open_the_top_drawer_and_put_the_bowl_inside_tb_20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0070__seed42 | Background Textures | 3 | open_the_top_drawer_and_put_the_bowl_inside_tb_5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0076__seed42 | Background Textures | 4 | push_the_plate_to_the_front_of_the_stove_table_14 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_goal__0100__seed42 | Background Textures | 2 | push_the_plate_to_the_front_of_the_stove_tb_7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0104__seed42 | Background Textures | 1 | put_the_bowl_on_the_plate_table_12 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_goal__0105__seed42 | Background Textures | 2 | put_the_bowl_on_the_plate_table_13 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0112__seed42 | Background Textures | 2 | put_the_bowl_on_the_plate_table_23 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0114__seed42 | Background Textures | 2 | put_the_bowl_on_the_plate_table_27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0115__seed42 | Background Textures | 1 | put_the_bowl_on_the_plate_table_3 | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_goal__0116__seed42 | Background Textures | 3 | put_the_bowl_on_the_plate_table_4 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_goal__0118__seed42 | Background Textures | 2 | put_the_bowl_on_the_plate_table_8 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_goal__0121__seed42 | Background Textures | 2 | put_the_bowl_on_the_plate_tb_13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0126__seed42 | Background Textures | 3 | put_the_bowl_on_the_plate_tb_2 | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| libero_goal__0127__seed42 | Background Textures | 2 | put_the_bowl_on_the_plate_tb_20 | 0 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_goal__0128__seed42 | Background Textures | 3 | put_the_bowl_on_the_plate_tb_21 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0134__seed42 | Background Textures | 3 | put_the_bowl_on_the_stove_table_3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0135__seed42 | Background Textures | 2 | put_the_bowl_on_the_stove_table_4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0136__seed42 | Background Textures | 3 | put_the_bowl_on_the_stove_table_5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0141__seed42 | Background Textures | 4 | put_the_bowl_on_top_of_the_cabinet_table_13 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__0150__seed42 | Background Textures | 3 | put_the_bowl_on_top_of_the_cabinet_tb_3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__0151__seed42 | Background Textures | 4 | put_the_cream_cheese_in_the_bowl_table_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0157__seed42 | Background Textures | 5 | put_the_cream_cheese_in_the_bowl_table_23 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0158__seed42 | Background Textures | 4 | put_the_cream_cheese_in_the_bowl_table_26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0168__seed42 | Background Textures | 5 | put_the_cream_cheese_in_the_bowl_tb_11 | 0 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_goal__0177__seed42 | Background Textures | 5 | put_the_cream_cheese_in_the_bowl_tb_20 | 1 | 1 | 1 | 0 | 1 | 1 | 0 |
| libero_goal__0179__seed42 | Background Textures | 5 | put_the_cream_cheese_in_the_bowl_tb_22 | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| libero_goal__0187__seed42 | Background Textures | 3 | put_the_wine_bottle_on_the_rack_table_13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0188__seed42 | Background Textures | 4 | put_the_wine_bottle_on_the_rack_table_14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0196__seed42 | Background Textures | 3 | put_the_wine_bottle_on_the_rack_table_22 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__0206__seed42 | Background Textures | 5 | put_the_wine_bottle_on_the_rack_table_7 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_goal__0215__seed42 | Background Textures | 3 | put_the_wine_bottle_on_the_rack_tb_15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0220__seed42 | Background Textures | 2 | put_the_wine_bottle_on_the_rack_tb_20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0224__seed42 | Background Textures | 2 | put_the_wine_bottle_on_the_rack_tb_4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0229__seed42 | Background Textures | 1 | put_the_wine_bottle_on_top_of_the_cabinet_table_1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_goal__0239__seed42 | Background Textures | 2 | put_the_wine_bottle_on_top_of_the_cabinet_table_22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0240__seed42 | Background Textures | 3 | put_the_wine_bottle_on_top_of_the_cabinet_table_23 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0249__seed42 | Background Textures | 3 | put_the_wine_bottle_on_top_of_the_cabinet_tb_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0251__seed42 | Background Textures | 4 | put_the_wine_bottle_on_top_of_the_cabinet_tb_12 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| libero_goal__0254__seed42 | Background Textures | 3 | put_the_wine_bottle_on_top_of_the_cabinet_tb_15 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| libero_goal__0261__seed42 | Background Textures | 3 | put_the_wine_bottle_on_top_of_the_cabinet_tb_22 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__0264__seed42 | Background Textures | 4 | put_the_wine_bottle_on_top_of_the_cabinet_tb_4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0271__seed42 | Background Textures | 2 | turn_on_the_stove_table_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0281__seed42 | Robot Initial States | 3 | open_the_middle_drawer_of_the_cabinet_view_0_0_100_0_0_initstate_1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__0285__seed42 | Robot Initial States | 3 | open_the_middle_drawer_of_the_cabinet_view_0_0_100_0_0_initstate_81 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__0286__seed42 | Robot Initial States | 3 | open_the_middle_drawer_of_the_cabinet_view_0_0_100_0_0_initstate_101 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__0289__seed42 | Robot Initial States | 1 | open_the_middle_drawer_of_the_cabinet_view_0_0_100_0_0_initstate_141 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__0292__seed42 | Robot Initial States | 2 | open_the_middle_drawer_of_the_cabinet_view_0_0_100_0_0_initstate_171 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| libero_goal__0298__seed42 | Robot Initial States | 1 | open_the_middle_drawer_of_the_cabinet_view_0_0_100_0_0_initstate_241 | 0 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_goal__0324__seed42 | Robot Initial States | 1 | put_the_bowl_on_the_stove_view_0_0_100_0_0_initstate_2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0334__seed42 | Robot Initial States | 5 | put_the_bowl_on_the_stove_view_0_0_100_0_0_initstate_222 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0345__seed42 | Robot Initial States | 3 | put_the_bowl_on_the_stove_view_0_0_100_0_0_initstate_332 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0346__seed42 | Robot Initial States | 5 | put_the_bowl_on_the_stove_view_0_0_100_0_0_initstate_342 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0349__seed42 | Robot Initial States | 4 | put_the_bowl_on_the_stove_view_0_0_100_0_0_initstate_372 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0354__seed42 | Robot Initial States | 4 | put_the_bowl_on_the_stove_view_0_0_100_0_0_initstate_422 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0355__seed42 | Robot Initial States | 4 | put_the_bowl_on_the_stove_view_0_0_100_0_0_initstate_432 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0369__seed42 | Robot Initial States | 2 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_103 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| libero_goal__0378__seed42 | Robot Initial States | 3 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_193 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0380__seed42 | Robot Initial States | 2 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_213 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_goal__0391__seed42 | Robot Initial States | 5 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_363 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0393__seed42 | Robot Initial States | 5 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_403 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0399__seed42 | Robot Initial States | 5 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_473 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0402__seed42 | Robot Initial States | 4 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0408__seed42 | Robot Initial States | 3 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_84 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0410__seed42 | Robot Initial States | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_114 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0414__seed42 | Robot Initial States | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_164 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0423__seed42 | Robot Initial States | 4 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_254 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0433__seed42 | Robot Initial States | 4 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_354 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0441__seed42 | Robot Initial States | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_444 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0452__seed42 | Robot Initial States | 3 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_115 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_goal__0456__seed42 | Robot Initial States | 3 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_165 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__0457__seed42 | Robot Initial States | 3 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_175 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |
| libero_goal__0460__seed42 | Robot Initial States | 1 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_205 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0467__seed42 | Robot Initial States | 4 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_275 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_goal__0469__seed42 | Robot Initial States | 2 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_295 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0475__seed42 | Robot Initial States | 4 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_355 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0477__seed42 | Robot Initial States | 1 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_375 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0478__seed42 | Robot Initial States | 5 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_385 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_goal__0479__seed42 | Robot Initial States | 4 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_395 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_goal__0508__seed42 | Robot Initial States | 2 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_276 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0514__seed42 | Robot Initial States | 2 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_336 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0523__seed42 | Robot Initial States | 4 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_436 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0524__seed42 | Robot Initial States | 3 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_446 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0531__seed42 | Robot Initial States | 3 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0554__seed42 | Robot Initial States | 1 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_267 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| libero_goal__0556__seed42 | Robot Initial States | 5 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_287 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0562__seed42 | Robot Initial States | 5 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_357 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0571__seed42 | Robot Initial States | 5 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_447 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_goal__0572__seed42 | Robot Initial States | 5 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_457 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0575__seed42 | Robot Initial States | 4 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_487 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0580__seed42 | Robot Initial States | 2 | turn_on_the_stove_view_0_0_100_0_0_initstate_128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0583__seed42 | Robot Initial States | 3 | turn_on_the_stove_view_0_0_100_0_0_initstate_168 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0591__seed42 | Robot Initial States | 3 | turn_on_the_stove_view_0_0_100_0_0_initstate_258 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0592__seed42 | Robot Initial States | 2 | turn_on_the_stove_view_0_0_100_0_0_initstate_268 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0596__seed42 | Robot Initial States | 1 | turn_on_the_stove_view_0_0_100_0_0_initstate_308 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0599__seed42 | Robot Initial States | 1 | turn_on_the_stove_view_0_0_100_0_0_initstate_338 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0601__seed42 | Robot Initial States | 2 | turn_on_the_stove_view_0_0_100_0_0_initstate_368 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0612__seed42 | Robot Initial States | 2 | turn_on_the_stove_view_0_0_100_0_0_initstate_478 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_goal__0614__seed42 | Robot Initial States | 2 | put_the_bowl_on_the_plate_view_0_0_100_0_0_initstate_19 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_goal__0651__seed42 | Robot Initial States | 2 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_10 | 0 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_goal__0658__seed42 | Robot Initial States | 2 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_90 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_goal__0661__seed42 | Robot Initial States | 3 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_130 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_goal__0664__seed42 | Robot Initial States | 1 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_170 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0671__seed42 | Robot Initial States | 3 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_240 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0678__seed42 | Robot Initial States | 2 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_360 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| libero_goal__0681__seed42 | Robot Initial States | 3 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_390 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0684__seed42 | Robot Initial States | 5 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_440 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0691__seed42 | Camera Viewpoints | 3 | open_the_middle_drawer_of_the_cabinet_view_0_0_100_2_6_initstate_0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| libero_goal__0702__seed42 | Camera Viewpoints | 2 | open_the_middle_drawer_of_the_cabinet_view_7_15_100_0_0_initstate_0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_goal__0708__seed42 | Camera Viewpoints | 3 | open_the_middle_drawer_of_the_cabinet_view_13_15_100_0_0_initstate_0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| libero_goal__0715__seed42 | Camera Viewpoints | 1 | put_the_bowl_on_the_stove_view_0_0_100_4_10_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0721__seed42 | Camera Viewpoints | 1 | put_the_bowl_on_the_stove_view_18_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0722__seed42 | Camera Viewpoints | 2 | put_the_bowl_on_the_stove_view_18_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0730__seed42 | Camera Viewpoints | 1 | put_the_bowl_on_the_stove_view_24_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0734__seed42 | Camera Viewpoints | 2 | put_the_bowl_on_the_stove_view_26_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0737__seed42 | Camera Viewpoints | 2 | put_the_bowl_on_the_stove_view_28_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0738__seed42 | Camera Viewpoints | 2 | put_the_bowl_on_the_stove_view_28_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0744__seed42 | Camera Viewpoints | 2 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_125_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__0749__seed42 | Camera Viewpoints | 1 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_130_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__0752__seed42 | Camera Viewpoints | 2 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_100_6_10_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0755__seed42 | Camera Viewpoints | 3 | put_the_wine_bottle_on_top_of_the_cabinet_view_32_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_goal__0756__seed42 | Camera Viewpoints | 3 | put_the_wine_bottle_on_top_of_the_cabinet_view_32_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0758__seed42 | Camera Viewpoints | 4 | put_the_wine_bottle_on_top_of_the_cabinet_view_33_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0765__seed42 | Camera Viewpoints | 3 | put_the_wine_bottle_on_top_of_the_cabinet_view_37_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__0772__seed42 | Camera Viewpoints | 5 | put_the_wine_bottle_on_top_of_the_cabinet_view_40_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_goal__0774__seed42 | Camera Viewpoints | 5 | put_the_wine_bottle_on_top_of_the_cabinet_view_41_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0792__seed42 | Camera Viewpoints | 4 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_8_4_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0794__seed42 | Camera Viewpoints | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_8_8_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0804__seed42 | Camera Viewpoints | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_47_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0805__seed42 | Camera Viewpoints | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_48_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0816__seed42 | Camera Viewpoints | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_53_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0822__seed42 | Camera Viewpoints | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_56_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0825__seed42 | Camera Viewpoints | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_58_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0829__seed42 | Camera Viewpoints | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_60_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0833__seed42 | Camera Viewpoints | 2 | put_the_bowl_on_top_of_the_cabinet_view_0_0_143_0_0_initstate_0 | 1 | 1 | 1 | 0 | 1 | 0 | 1 |
| libero_goal__0846__seed42 | Camera Viewpoints | 5 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_10_350_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0854__seed42 | Camera Viewpoints | 2 | put_the_bowl_on_top_of_the_cabinet_view_62_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0859__seed42 | Camera Viewpoints | 3 | put_the_bowl_on_top_of_the_cabinet_view_65_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0862__seed42 | Camera Viewpoints | 3 | put_the_bowl_on_top_of_the_cabinet_view_66_15_100_0_0_initstate_0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 |
| libero_goal__0863__seed42 | Camera Viewpoints | 4 | put_the_bowl_on_top_of_the_cabinet_view_67_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0886__seed42 | Camera Viewpoints | 4 | push_the_plate_to_the_front_of_the_stove_view_0_0_156_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0888__seed42 | Camera Viewpoints | 3 | push_the_plate_to_the_front_of_the_stove_view_0_0_158_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0906__seed42 | Camera Viewpoints | 5 | push_the_plate_to_the_front_of_the_stove_view_288_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0925__seed42 | Camera Viewpoints | 4 | push_the_plate_to_the_front_of_the_stove_view_297_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0926__seed42 | Camera Viewpoints | 5 | push_the_plate_to_the_front_of_the_stove_view_298_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0927__seed42 | Camera Viewpoints | 4 | push_the_plate_to_the_front_of_the_stove_view_298_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0930__seed42 | Camera Viewpoints | 2 | put_the_cream_cheese_in_the_bowl_view_0_0_161_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0943__seed42 | Camera Viewpoints | 4 | put_the_cream_cheese_in_the_bowl_view_0_0_100_352_8_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0953__seed42 | Camera Viewpoints | 3 | put_the_cream_cheese_in_the_bowl_view_301_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0981__seed42 | Camera Viewpoints | 2 | turn_on_the_stove_view_0_0_173_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0989__seed42 | Camera Viewpoints | 1 | turn_on_the_stove_view_0_0_100_354_4_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__0991__seed42 | Camera Viewpoints | 2 | turn_on_the_stove_view_0_0_100_354_10_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1011__seed42 | Camera Viewpoints | 2 | turn_on_the_stove_view_323_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1039__seed42 | Camera Viewpoints | 2 | put_the_bowl_on_the_plate_view_330_15_100_0_0_initstate_0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1051__seed42 | Camera Viewpoints | 3 | put_the_bowl_on_the_plate_view_340_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1065__seed42 | Camera Viewpoints | 4 | put_the_wine_bottle_on_the_rack_view_0_0_200_0_0_initstate_0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 |
| libero_goal__1069__seed42 | Camera Viewpoints | 3 | put_the_wine_bottle_on_the_rack_view_0_0_100_358_8_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1082__seed42 | Camera Viewpoints | 3 | put_the_wine_bottle_on_the_rack_view_348_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1086__seed42 | Camera Viewpoints | 2 | put_the_wine_bottle_on_the_rack_view_351_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1092__seed42 | Camera Viewpoints | 3 | put_the_wine_bottle_on_the_rack_view_354_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1102__seed42 | Language Instructions | 4 | open_the_middle_drawer_of_the_cabinet_language_5_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1109__seed42 | Language Instructions | 5 | open_the_middle_drawer_of_the_cabinet_language_12_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1110__seed42 | Language Instructions | 2 | open_the_middle_drawer_of_the_cabinet_language_13_view_0_0_100_0_0_initstate_0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_goal__1111__seed42 | Language Instructions | 4 | open_the_middle_drawer_of_the_cabinet_language_14_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1112__seed42 | Language Instructions | 5 | open_the_middle_drawer_of_the_cabinet_language_15_view_0_0_100_0_0_initstate_0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_goal__1114__seed42 | Language Instructions | 3 | open_the_middle_drawer_of_the_cabinet_language_19_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1116__seed42 | Language Instructions | 5 | open_the_middle_drawer_of_the_cabinet_language_21_view_0_0_100_0_0_initstate_0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1119__seed42 | Language Instructions | 3 | open_the_middle_drawer_of_the_cabinet_language_24_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1121__seed42 | Language Instructions | 4 | open_the_middle_drawer_of_the_cabinet_language_26_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1124__seed42 | Language Instructions | 2 | open_the_middle_drawer_of_the_cabinet_language_30_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_goal__1127__seed42 | Language Instructions | 5 | open_the_middle_drawer_of_the_cabinet_language_33_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1133__seed42 | Language Instructions | 5 | open_the_middle_drawer_of_the_cabinet_language_40_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |
| libero_goal__1139__seed42 | Language Instructions | 5 | open_the_middle_drawer_of_the_cabinet_language_46_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1143__seed42 | Language Instructions | 5 | open_the_middle_drawer_of_the_cabinet_language_50_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1147__seed42 | Language Instructions | 1 | put_the_bowl_on_the_stove_language_6_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1152__seed42 | Language Instructions | 4 | put_the_bowl_on_the_stove_language_26_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1156__seed42 | Language Instructions | 5 | put_the_bowl_on_the_stove_language_40_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1166__seed42 | Language Instructions | 4 | put_the_wine_bottle_on_top_of_the_cabinet_language_5_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1170__seed42 | Language Instructions | 5 | put_the_wine_bottle_on_top_of_the_cabinet_language_9_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1189__seed42 | Language Instructions | 4 | put_the_wine_bottle_on_top_of_the_cabinet_language_35_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1200__seed42 | Language Instructions | 2 | put_the_wine_bottle_on_top_of_the_cabinet_language_49_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1204__seed42 | Language Instructions | 5 | open_the_top_drawer_and_put_the_bowl_inside_language_3_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1214__seed42 | Language Instructions | 5 | open_the_top_drawer_and_put_the_bowl_inside_language_13_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| libero_goal__1218__seed42 | Language Instructions | 3 | open_the_top_drawer_and_put_the_bowl_inside_language_17_view_0_0_100_0_0_initstate_0 | 0 | 1 | 0 | 1 | 0 | 1 | 1 |
| libero_goal__1231__seed42 | Language Instructions | 5 | open_the_top_drawer_and_put_the_bowl_inside_language_30_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1240__seed42 | Language Instructions | 5 | open_the_top_drawer_and_put_the_bowl_inside_language_39_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1258__seed42 | Language Instructions | 2 | put_the_bowl_on_top_of_the_cabinet_language_8_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1264__seed42 | Language Instructions | 4 | put_the_bowl_on_top_of_the_cabinet_language_15_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1277__seed42 | Language Instructions | 2 | put_the_bowl_on_top_of_the_cabinet_language_30_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_goal__1278__seed42 | Language Instructions | 5 | put_the_bowl_on_top_of_the_cabinet_language_31_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1281__seed42 | Language Instructions | 5 | put_the_bowl_on_top_of_the_cabinet_language_35_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1285__seed42 | Language Instructions | 1 | put_the_bowl_on_top_of_the_cabinet_language_39_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1291__seed42 | Language Instructions | 4 | put_the_bowl_on_top_of_the_cabinet_language_45_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |
| libero_goal__1303__seed42 | Language Instructions | 2 | push_the_plate_to_the_front_of_the_stove_language_11_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1306__seed42 | Language Instructions | 5 | push_the_plate_to_the_front_of_the_stove_language_14_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1311__seed42 | Language Instructions | 5 | push_the_plate_to_the_front_of_the_stove_language_20_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1312__seed42 | Language Instructions | 5 | push_the_plate_to_the_front_of_the_stove_language_21_view_0_0_100_0_0_initstate_0 | 1 | 0 | 0 | 1 | 1 | 0 | 1 |
| libero_goal__1315__seed42 | Language Instructions | 1 | push_the_plate_to_the_front_of_the_stove_language_25_view_0_0_100_0_0_initstate_0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1317__seed42 | Language Instructions | 5 | push_the_plate_to_the_front_of_the_stove_language_27_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1335__seed42 | Language Instructions | 4 | put_the_cream_cheese_in_the_bowl_language_1_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| libero_goal__1338__seed42 | Language Instructions | 5 | put_the_cream_cheese_in_the_bowl_language_5_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1363__seed42 | Language Instructions | 3 | put_the_cream_cheese_in_the_bowl_language_30_view_0_0_100_0_0_initstate_0 | 0 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_goal__1365__seed42 | Language Instructions | 3 | put_the_cream_cheese_in_the_bowl_language_32_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1375__seed42 | Language Instructions | 5 | put_the_cream_cheese_in_the_bowl_language_42_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1377__seed42 | Language Instructions | 2 | put_the_cream_cheese_in_the_bowl_language_44_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1380__seed42 | Language Instructions | 3 | put_the_cream_cheese_in_the_bowl_language_47_view_0_0_100_0_0_initstate_0 | 0 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_goal__1382__seed42 | Language Instructions | 5 | put_the_cream_cheese_in_the_bowl_language_49_view_0_0_100_0_0_initstate_0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| libero_goal__1386__seed42 | Language Instructions | 4 | turn_on_the_stove_language_4_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1405__seed42 | Language Instructions | 3 | turn_on_the_stove_language_29_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1409__seed42 | Language Instructions | 1 | turn_on_the_stove_language_35_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1412__seed42 | Language Instructions | 2 | turn_on_the_stove_language_38_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1414__seed42 | Language Instructions | 4 | turn_on_the_stove_language_42_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1415__seed42 | Language Instructions | 4 | turn_on_the_stove_language_43_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1429__seed42 | Language Instructions | 5 | put_the_bowl_on_the_plate_language_9_view_0_0_100_0_0_initstate_0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_goal__1430__seed42 | Language Instructions | 1 | put_the_bowl_on_the_plate_language_11_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1438__seed42 | Language Instructions | 5 | put_the_bowl_on_the_plate_language_23_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1439__seed42 | Language Instructions | 2 | put_the_bowl_on_the_plate_language_26_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1442__seed42 | Language Instructions | 2 | put_the_bowl_on_the_plate_language_30_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_goal__1446__seed42 | Language Instructions | 1 | put_the_bowl_on_the_plate_language_34_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1453__seed42 | Language Instructions | 5 | put_the_bowl_on_the_plate_language_43_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1454__seed42 | Language Instructions | 4 | put_the_bowl_on_the_plate_language_46_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1462__seed42 | Language Instructions | 1 | put_the_wine_bottle_on_the_rack_language_5_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1465__seed42 | Language Instructions | 2 | put_the_wine_bottle_on_the_rack_language_8_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1474__seed42 | Language Instructions | 4 | put_the_wine_bottle_on_the_rack_language_17_view_0_0_100_0_0_initstate_0 | 0 | 0 | 1 | 0 | 0 | 1 | 1 |
| libero_goal__1491__seed42 | Language Instructions | 5 | put_the_wine_bottle_on_the_rack_language_34_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1502__seed42 | Language Instructions | 2 | put_the_wine_bottle_on_the_rack_language_45_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1506__seed42 | Language Instructions | 2 | put_the_wine_bottle_on_the_rack_language_49_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_goal__1507__seed42 | Language Instructions | 5 | put_the_wine_bottle_on_the_rack_language_50_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1510__seed42 | Sensor Noise | 2 | open_the_middle_drawer_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1521__seed42 | Sensor Noise | 4 | open_the_middle_drawer_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_15 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_goal__1531__seed42 | Sensor Noise | 1 | open_the_middle_drawer_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_26 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| libero_goal__1539__seed42 | Sensor Noise | 5 | open_the_middle_drawer_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1554__seed42 | Sensor Noise | 4 | put_the_bowl_on_the_stove_view_0_0_100_0_0_initstate_0_noise_8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1555__seed42 | Sensor Noise | 4 | put_the_bowl_on_the_stove_view_0_0_100_0_0_initstate_0_noise_9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1570__seed42 | Sensor Noise | 1 | put_the_bowl_on_the_stove_view_0_0_100_0_0_initstate_0_noise_37 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1573__seed42 | Sensor Noise | 2 | put_the_bowl_on_the_stove_view_0_0_100_0_0_initstate_0_noise_44 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1591__seed42 | Sensor Noise | 2 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1592__seed42 | Sensor Noise | 3 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1594__seed42 | Sensor Noise | 4 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_18 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| libero_goal__1602__seed42 | Sensor Noise | 1 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_31 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_goal__1613__seed42 | Sensor Noise | 2 | put_the_wine_bottle_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_47 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_goal__1621__seed42 | Sensor Noise | 4 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_0_noise_7 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1630__seed42 | Sensor Noise | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_0_noise_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1631__seed42 | Sensor Noise | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_0_noise_19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1633__seed42 | Sensor Noise | 2 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_0_noise_21 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1637__seed42 | Sensor Noise | 3 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_0_noise_27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1639__seed42 | Sensor Noise | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_0_noise_30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1654__seed42 | Sensor Noise | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_0_noise_47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1657__seed42 | Sensor Noise | 5 | open_the_top_drawer_and_put_the_bowl_inside_view_0_0_100_0_0_initstate_0_noise_50 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1668__seed42 | Sensor Noise | 3 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_12 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_goal__1673__seed42 | Sensor Noise | 5 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_17 | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| libero_goal__1678__seed42 | Sensor Noise | 4 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_22 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| libero_goal__1690__seed42 | Sensor Noise | 3 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_36 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| libero_goal__1698__seed42 | Sensor Noise | 4 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_44 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| libero_goal__1703__seed42 | Sensor Noise | 5 | put_the_bowl_on_top_of_the_cabinet_view_0_0_100_0_0_initstate_0_noise_50 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| libero_goal__1704__seed42 | Sensor Noise | 1 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_0_noise_3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1708__seed42 | Sensor Noise | 4 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_0_noise_7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1710__seed42 | Sensor Noise | 5 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_0_noise_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1715__seed42 | Sensor Noise | 4 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_0_noise_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1717__seed42 | Sensor Noise | 4 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_0_noise_20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1721__seed42 | Sensor Noise | 1 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_0_noise_27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1731__seed42 | Sensor Noise | 2 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_0_noise_45 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1734__seed42 | Sensor Noise | 3 | push_the_plate_to_the_front_of_the_stove_view_0_0_100_0_0_initstate_0_noise_48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1759__seed42 | Sensor Noise | 1 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_0_noise_26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1764__seed42 | Sensor Noise | 2 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_0_noise_31 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1766__seed42 | Sensor Noise | 2 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_0_noise_33 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1772__seed42 | Sensor Noise | 3 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_0_noise_39 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1774__seed42 | Sensor Noise | 2 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_0_noise_41 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_goal__1777__seed42 | Sensor Noise | 2 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_0_noise_44 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1781__seed42 | Sensor Noise | 4 | put_the_cream_cheese_in_the_bowl_view_0_0_100_0_0_initstate_0_noise_48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1788__seed42 | Sensor Noise | 3 | turn_on_the_stove_view_0_0_100_0_0_initstate_0_noise_9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1798__seed42 | Sensor Noise | 1 | turn_on_the_stove_view_0_0_100_0_0_initstate_0_noise_28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1801__seed42 | Sensor Noise | 1 | turn_on_the_stove_view_0_0_100_0_0_initstate_0_noise_37 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1811__seed42 | Sensor Noise | 2 | turn_on_the_stove_view_0_0_100_0_0_initstate_0_noise_49 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1822__seed42 | Sensor Noise | 2 | put_the_bowl_on_the_plate_view_0_0_100_0_0_initstate_0_noise_16 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1825__seed42 | Sensor Noise | 3 | put_the_bowl_on_the_plate_view_0_0_100_0_0_initstate_0_noise_19 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_goal__1840__seed42 | Sensor Noise | 2 | put_the_bowl_on_the_plate_view_0_0_100_0_0_initstate_0_noise_46 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1847__seed42 | Sensor Noise | 1 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_0_noise_3 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1848__seed42 | Sensor Noise | 2 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_0_noise_4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1849__seed42 | Sensor Noise | 3 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_0_noise_5 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_goal__1850__seed42 | Sensor Noise | 4 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_0_noise_6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1854__seed42 | Sensor Noise | 5 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_0_noise_10 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_goal__1866__seed42 | Sensor Noise | 1 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_0_noise_25 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_goal__1876__seed42 | Sensor Noise | 1 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_0_noise_36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1878__seed42 | Sensor Noise | 2 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_0_noise_41 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1881__seed42 | Sensor Noise | 4 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_0_noise_45 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_goal__1884__seed42 | Sensor Noise | 4 | put_the_wine_bottle_on_the_rack_view_0_0_100_0_0_initstate_0_noise_48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1890__seed42 | Objects Layout | 2 | open_the_middle_drawer_of_the_cabinet_add_13 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1900__seed42 | Objects Layout | 1 | open_the_middle_drawer_of_the_cabinet_add_23 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| libero_goal__1907__seed42 | Objects Layout | 2 | open_the_middle_drawer_of_the_cabinet_add_30 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1908__seed42 | Objects Layout | 1 | open_the_middle_drawer_of_the_cabinet_add_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1910__seed42 | Objects Layout | 1 | open_the_middle_drawer_of_the_cabinet_add_8 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| libero_goal__1916__seed42 | Objects Layout | 5 | open_the_middle_drawer_of_the_cabinet_moved_level2_sample1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_goal__1931__seed42 | Objects Layout | 2 | open_the_top_drawer_and_put_the_bowl_inside_add_10 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| libero_goal__1946__seed42 | Objects Layout | 2 | open_the_top_drawer_and_put_the_bowl_inside_add_26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1949__seed42 | Objects Layout | 1 | open_the_top_drawer_and_put_the_bowl_inside_add_29 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| libero_goal__1953__seed42 | Objects Layout | 3 | open_the_top_drawer_and_put_the_bowl_inside_add_5 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1954__seed42 | Objects Layout | 3 | open_the_top_drawer_and_put_the_bowl_inside_add_6 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_goal__1962__seed42 | Objects Layout | 4 | open_the_top_drawer_and_put_the_bowl_inside_moved_level2_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1965__seed42 | Objects Layout | 5 | open_the_top_drawer_and_put_the_bowl_inside_moved_level3_sample2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1970__seed42 | Objects Layout | 5 | open_the_top_drawer_and_put_the_bowl_inside_moved_level4_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1972__seed42 | Objects Layout | 5 | open_the_top_drawer_and_put_the_bowl_inside_moved_level5_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1977__seed42 | Objects Layout | 1 | push_the_plate_to_the_front_of_the_stove_add_10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1979__seed42 | Objects Layout | 1 | push_the_plate_to_the_front_of_the_stove_add_12 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| libero_goal__1986__seed42 | Objects Layout | 1 | push_the_plate_to_the_front_of_the_stove_add_2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__1989__seed42 | Objects Layout | 2 | push_the_plate_to_the_front_of_the_stove_add_23 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__1992__seed42 | Objects Layout | 1 | push_the_plate_to_the_front_of_the_stove_add_26 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_goal__1999__seed42 | Objects Layout | 1 | push_the_plate_to_the_front_of_the_stove_add_6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2006__seed42 | Objects Layout | 5 | push_the_plate_to_the_front_of_the_stove_moved_level2_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2010__seed42 | Objects Layout | 4 | push_the_plate_to_the_front_of_the_stove_moved_level3_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2022__seed42 | Objects Layout | 1 | put_the_bowl_on_the_plate_add_1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2023__seed42 | Objects Layout | 3 | put_the_bowl_on_the_plate_add_10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2025__seed42 | Objects Layout | 3 | put_the_bowl_on_the_plate_add_13 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| libero_goal__2031__seed42 | Objects Layout | 2 | put_the_bowl_on_the_plate_add_19 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| libero_goal__2041__seed42 | Objects Layout | 3 | put_the_bowl_on_the_plate_add_29 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2049__seed42 | Objects Layout | 2 | put_the_bowl_on_the_plate_moved_level1_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2066__seed42 | Objects Layout | 1 | put_the_bowl_on_the_stove_add_14 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| libero_goal__2074__seed42 | Objects Layout | 1 | put_the_bowl_on_the_stove_add_29 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| libero_goal__2077__seed42 | Objects Layout | 2 | put_the_bowl_on_the_stove_moved_level1_sample4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2082__seed42 | Objects Layout | 3 | put_the_bowl_on_the_stove_moved_level3_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2087__seed42 | Objects Layout | 4 | put_the_bowl_on_the_stove_moved_level4_sample2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2089__seed42 | Objects Layout | 4 | put_the_bowl_on_the_stove_moved_level4_sample4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2090__seed42 | Objects Layout | 5 | put_the_bowl_on_the_stove_moved_level5_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2104__seed42 | Objects Layout | 2 | put_the_bowl_on_top_of_the_cabinet_add_25 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2114__seed42 | Objects Layout | 4 | put_the_bowl_on_top_of_the_cabinet_moved_level2_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2116__seed42 | Objects Layout | 3 | put_the_bowl_on_top_of_the_cabinet_moved_level2_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2117__seed42 | Objects Layout | 4 | put_the_bowl_on_top_of_the_cabinet_moved_level2_sample4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2120__seed42 | Objects Layout | 3 | put_the_bowl_on_top_of_the_cabinet_moved_level3_sample4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2123__seed42 | Objects Layout | 5 | put_the_bowl_on_top_of_the_cabinet_moved_level4_sample4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2124__seed42 | Objects Layout | 5 | put_the_bowl_on_top_of_the_cabinet_moved_level5_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2142__seed42 | Objects Layout | 2 | put_the_cream_cheese_in_the_bowl_add_25 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2144__seed42 | Objects Layout | 3 | put_the_cream_cheese_in_the_bowl_add_27 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2153__seed42 | Objects Layout | 1 | put_the_cream_cheese_in_the_bowl_moved_level1_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2186__seed42 | Objects Layout | 2 | put_the_wine_bottle_on_the_rack_add_23 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2192__seed42 | Objects Layout | 1 | put_the_wine_bottle_on_the_rack_add_29 | 0 | 0 | 1 | 0 | 1 | 1 | 0 |
| libero_goal__2194__seed42 | Objects Layout | 1 | put_the_wine_bottle_on_the_rack_add_30 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2195__seed42 | Objects Layout | 1 | put_the_wine_bottle_on_the_rack_add_4 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| libero_goal__2197__seed42 | Objects Layout | 1 | put_the_wine_bottle_on_the_rack_add_6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2237__seed42 | Objects Layout | 1 | put_the_wine_bottle_on_top_of_the_cabinet_add_29 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2239__seed42 | Objects Layout | 1 | put_the_wine_bottle_on_top_of_the_cabinet_add_4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2254__seed42 | Objects Layout | 4 | put_the_wine_bottle_on_top_of_the_cabinet_moved_level3_sample2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2260__seed42 | Objects Layout | 5 | put_the_wine_bottle_on_top_of_the_cabinet_moved_level4_sample4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2264__seed42 | Objects Layout | 1 | turn_on_the_stove_add_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2280__seed42 | Objects Layout | 2 | turn_on_the_stove_add_24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2282__seed42 | Objects Layout | 1 | turn_on_the_stove_add_26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2293__seed42 | Objects Layout | 1 | turn_on_the_stove_add_9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2298__seed42 | Objects Layout | 5 | turn_on_the_stove_moved_level2_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2309__seed42 | Objects Layout | 5 | turn_on_the_stove_moved_level5_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2310__seed42 | Objects Layout | 4 | turn_on_the_stove_moved_level5_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2315__seed42 | Light Conditions | unspecified | open_the_middle_drawer_of_the_cabinet_light_12 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2319__seed42 | Light Conditions | 1 | open_the_middle_drawer_of_the_cabinet_light_16 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2321__seed42 | Light Conditions | unspecified | open_the_middle_drawer_of_the_cabinet_light_18 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2327__seed42 | Light Conditions | unspecified | open_the_middle_drawer_of_the_cabinet_light_25 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2337__seed42 | Light Conditions | 1 | open_the_middle_drawer_of_the_cabinet_light_35 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2343__seed42 | Light Conditions | 2 | open_the_middle_drawer_of_the_cabinet_light_43 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2344__seed42 | Light Conditions | 2 | open_the_middle_drawer_of_the_cabinet_light_45 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2345__seed42 | Light Conditions | 2 | open_the_middle_drawer_of_the_cabinet_light_46 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2357__seed42 | Light Conditions | unspecified | open_the_top_drawer_and_put_the_bowl_inside_light_12 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_goal__2375__seed42 | Light Conditions | 3 | open_the_top_drawer_and_put_the_bowl_inside_light_29 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_goal__2379__seed42 | Light Conditions | 5 | open_the_top_drawer_and_put_the_bowl_inside_light_32 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_goal__2395__seed42 | Light Conditions | 3 | open_the_top_drawer_and_put_the_bowl_inside_light_47 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2400__seed42 | Light Conditions | unspecified | open_the_top_drawer_and_put_the_bowl_inside_light_6 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_goal__2451__seed42 | Light Conditions | 1 | put_the_bowl_on_the_plate_light_29 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2466__seed42 | Light Conditions | 4 | put_the_bowl_on_the_stove_light_11 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| libero_goal__2473__seed42 | Light Conditions | 1 | put_the_bowl_on_top_of_the_cabinet_light_21 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2480__seed42 | Light Conditions | unspecified | put_the_bowl_on_top_of_the_cabinet_light_50 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_goal__2481__seed42 | Light Conditions | 1 | put_the_bowl_on_top_of_the_cabinet_light_6 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_goal__2489__seed42 | Light Conditions | 1 | put_the_cream_cheese_in_the_bowl_light_22 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2491__seed42 | Light Conditions | unspecified | put_the_cream_cheese_in_the_bowl_light_26 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_goal__2499__seed42 | Light Conditions | 1 | put_the_cream_cheese_in_the_bowl_light_42 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2503__seed42 | Light Conditions | 4 | put_the_cream_cheese_in_the_bowl_light_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2540__seed42 | Light Conditions | 1 | put_the_wine_bottle_on_the_rack_light_7 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2541__seed42 | Light Conditions | unspecified | put_the_wine_bottle_on_the_rack_light_8 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| libero_goal__2542__seed42 | Light Conditions | 3 | put_the_wine_bottle_on_top_of_the_cabinet_light_10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2543__seed42 | Light Conditions | 3 | put_the_wine_bottle_on_top_of_the_cabinet_light_11 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2553__seed42 | Light Conditions | 5 | put_the_wine_bottle_on_top_of_the_cabinet_light_3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2561__seed42 | Light Conditions | 1 | put_the_wine_bottle_on_top_of_the_cabinet_light_39 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2563__seed42 | Light Conditions | unspecified | put_the_wine_bottle_on_top_of_the_cabinet_light_40 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_goal__2589__seed42 | Light Conditions | 2 | turn_on_the_stove_light_5 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| libero_spatial__0003__seed42 | Background Textures | 2 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_13 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0008__seed42 | Background Textures | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_18 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_spatial__0009__seed42 | Background Textures | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_19 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0015__seed42 | Background Textures | 2 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_28 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0026__seed42 | Background Textures | 2 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_tb_6 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| libero_spatial__0033__seed42 | Background Textures | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_table_15 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0041__seed42 | Background Textures | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_table_23 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0054__seed42 | Background Textures | 4 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_tb_11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0055__seed42 | Background Textures | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_tb_12 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0060__seed42 | Background Textures | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_tb_17 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0062__seed42 | Background Textures | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_tb_19 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0074__seed42 | Background Textures | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_tb_9 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_spatial__0095__seed42 | Background Textures | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_tb_17 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0101__seed42 | Background Textures | 1 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_table_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0117__seed42 | Background Textures | 1 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_tb_20 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0123__seed42 | Background Textures | 2 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_table_16 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| libero_spatial__0124__seed42 | Background Textures | 3 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_table_24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0129__seed42 | Background Textures | 1 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_table_12 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| libero_spatial__0132__seed42 | Background Textures | 3 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_table_9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0134__seed42 | Background Textures | 2 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_tb_20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0139__seed42 | Background Textures | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_table_13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0147__seed42 | Background Textures | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_table_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0150__seed42 | Background Textures | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_tb_11 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |
| libero_spatial__0151__seed42 | Background Textures | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_tb_12 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0152__seed42 | Background Textures | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_tb_13 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_spatial__0156__seed42 | Background Textures | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_tb_17 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0157__seed42 | Background Textures | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_tb_18 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_spatial__0161__seed42 | Background Textures | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_tb_22 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| libero_spatial__0168__seed42 | Background Textures | 1 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_table_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0174__seed42 | Background Textures | 1 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_table_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0177__seed42 | Background Textures | 1 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_table_2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0182__seed42 | Background Textures | 1 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_table_26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0185__seed42 | Background Textures | 3 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_table_6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0194__seed42 | Background Textures | 3 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_tb_18 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| libero_spatial__0195__seed42 | Background Textures | 2 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_tb_19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0201__seed42 | Background Textures | 1 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_tb_8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0204__seed42 | Background Textures | 4 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_table_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0222__seed42 | Background Textures | 3 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_table_7 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| libero_spatial__0265__seed42 | Robot Initial States | 2 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_161 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_spatial__0267__seed42 | Robot Initial States | 4 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_211 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0287__seed42 | Robot Initial States | 4 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_471 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_spatial__0290__seed42 | Robot Initial States | 1 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_42 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_spatial__0296__seed42 | Robot Initial States | 4 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_152 | 1 | 1 | 0 | 0 | 1 | 0 | 1 |
| libero_spatial__0309__seed42 | Robot Initial States | 3 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_302 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0310__seed42 | Robot Initial States | 5 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_312 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0317__seed42 | Robot Initial States | 3 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_382 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0325__seed42 | Robot Initial States | 5 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_472 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0331__seed42 | Robot Initial States | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_83 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0335__seed42 | Robot Initial States | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_153 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0345__seed42 | Robot Initial States | 1 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_293 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_spatial__0352__seed42 | Robot Initial States | 5 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_363 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0353__seed42 | Robot Initial States | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_373 | 1 | 0 | 1 | 1 | 1 | 0 | 1 |
| libero_spatial__0363__seed42 | Robot Initial States | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_114 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0369__seed42 | Robot Initial States | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_224 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0370__seed42 | Robot Initial States | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_284 | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| libero_spatial__0377__seed42 | Robot Initial States | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_394 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0379__seed42 | Robot Initial States | 3 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_424 | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| libero_spatial__0392__seed42 | Robot Initial States | 5 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_125 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0402__seed42 | Robot Initial States | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_235 | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| libero_spatial__0427__seed42 | Robot Initial States | 1 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_16 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0436__seed42 | Robot Initial States | 3 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_236 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0438__seed42 | Robot Initial States | 1 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_266 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| libero_spatial__0446__seed42 | Robot Initial States | 3 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_356 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0449__seed42 | Robot Initial States | 3 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_396 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0455__seed42 | Robot Initial States | 5 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_466 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0470__seed42 | Robot Initial States | 1 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_317 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0472__seed42 | Robot Initial States | 3 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_337 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| libero_spatial__0481__seed42 | Robot Initial States | 5 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_457 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0498__seed42 | Robot Initial States | 3 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_198 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0513__seed42 | Robot Initial States | 1 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_348 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_spatial__0517__seed42 | Robot Initial States | 5 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_388 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0523__seed42 | Robot Initial States | 5 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_448 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0524__seed42 | Robot Initial States | 5 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_458 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0529__seed42 | Robot Initial States | 1 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_19 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| libero_spatial__0534__seed42 | Robot Initial States | 1 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_129 | 1 | 0 | 1 | 0 | 1 | 1 | 1 |
| libero_spatial__0536__seed42 | Robot Initial States | 3 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_149 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| libero_spatial__0559__seed42 | Robot Initial States | 2 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_419 | 1 | 0 | 0 | 1 | 1 | 1 | 1 |
| libero_spatial__0564__seed42 | Robot Initial States | 2 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_489 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0584__seed42 | Robot Initial States | 5 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_270 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0585__seed42 | Robot Initial States | 2 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_280 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0591__seed42 | Robot Initial States | 4 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_340 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0596__seed42 | Robot Initial States | 4 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_390 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0618__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_4_8_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0619__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_4_10_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0620__seed42 | Camera Viewpoints | 4 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_4_350_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0623__seed42 | Camera Viewpoints | 5 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_16_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0634__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_22_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0635__seed42 | Camera Viewpoints | 5 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_22_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0638__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_24_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0643__seed42 | Camera Viewpoints | 5 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_26_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0648__seed42 | Camera Viewpoints | 5 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_29_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0651__seed42 | Camera Viewpoints | 1 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_122_0_0_initstate_0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| libero_spatial__0653__seed42 | Camera Viewpoints | 1 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_126_0_0_initstate_0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| libero_spatial__0663__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_32_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0666__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_34_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0687__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_0_0_132_0_0_initstate_0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| libero_spatial__0695__seed42 | Camera Viewpoints | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_0_0_140_0_0_initstate_0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 |
| libero_spatial__0699__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_8_10_initstate_0 | 1 | 0 | 1 | 1 | 0 | 0 | 1 |
| libero_spatial__0703__seed42 | Camera Viewpoints | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_8_356_initstate_0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_spatial__0711__seed42 | Camera Viewpoints | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_49_0_100_0_0_initstate_0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0715__seed42 | Camera Viewpoints | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_51_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0720__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_53_15_100_0_0_initstate_0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 |
| libero_spatial__0727__seed42 | Camera Viewpoints | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_57_0_100_0_0_initstate_0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| libero_spatial__0731__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_59_0_100_0_0_initstate_0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_spatial__0737__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_143_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0748__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_10_8_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_spatial__0757__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_62_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_spatial__0758__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_63_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0759__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_63_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0763__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_65_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0782__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_75_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0784__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_152_0_0_initstate_0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_spatial__0786__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_155_0_0_initstate_0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| libero_spatial__0794__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_350_6_initstate_0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0804__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_287_0_100_0_0_initstate_0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| libero_spatial__0806__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_288_0_100_0_0_initstate_0 | 0 | 0 | 1 | 1 | 1 | 0 | 1 |
| libero_spatial__0809__seed42 | Camera Viewpoints | 5 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_289_15_100_0_0_initstate_0 | 0 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_spatial__0810__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_290_0_100_0_0_initstate_0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_spatial__0823__seed42 | Camera Viewpoints | 4 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_297_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0838__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_0_0_170_0_0_initstate_0 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_spatial__0849__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_303_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0854__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_306_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_spatial__0857__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_308_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0858__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_308_15_100_0_0_initstate_0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| libero_spatial__0865__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_312_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0866__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_312_15_100_0_0_initstate_0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0871__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_0_0_171_0_0_initstate_0 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_spatial__0884__seed42 | Camera Viewpoints | 1 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_0_0_100_354_356_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0886__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_315_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0890__seed42 | Camera Viewpoints | 4 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_317_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0897__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_321_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0900__seed42 | Camera Viewpoints | 4 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_322_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0907__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_326_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0914__seed42 | Camera Viewpoints | 4 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_181_0_0_initstate_0 | 0 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_spatial__0916__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_183_0_0_initstate_0 | 1 | 0 | 1 | 1 | 0 | 0 | 1 |
| libero_spatial__0918__seed42 | Camera Viewpoints | 4 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_185_0_0_initstate_0 | 0 | 1 | 1 | 0 | 0 | 1 | 1 |
| libero_spatial__0921__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_188_0_0_initstate_0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
| libero_spatial__0938__seed42 | Camera Viewpoints | 2 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_335_0_100_0_0_initstate_0 | 0 | 0 | 0 | 1 | 1 | 0 | 1 |
| libero_spatial__0941__seed42 | Camera Viewpoints | 1 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_338_0_100_0_0_initstate_0 | 0 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_spatial__0945__seed42 | Camera Viewpoints | 1 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_343_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0966__seed42 | Camera Viewpoints | 5 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_345_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0982__seed42 | Camera Viewpoints | 3 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_358_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__0989__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_6_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0991__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_8_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0995__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_12_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0998__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_15_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__0999__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_18_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1000__seed42 | Language Instructions | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_19_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1002__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_21_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1004__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_23_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1005__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_24_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1006__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_25_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1009__seed42 | Language Instructions | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_28_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1024__seed42 | Language Instructions | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_language_45_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1031__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_language_2_view_0_0_100_0_0_initstate_0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 |
| libero_spatial__1036__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_language_8_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1075__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_language_23_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1076__seed42 | Language Instructions | 5 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_language_25_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1080__seed42 | Language Instructions | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_language_33_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1082__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_language_37_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1089__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_language_49_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1091__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_language_2_view_0_0_100_0_0_initstate_0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_spatial__1112__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_language_29_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1118__seed42 | Language Instructions | 3 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_language_36_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1126__seed42 | Language Instructions | 3 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_language_44_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1137__seed42 | Language Instructions | 5 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_language_9_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_spatial__1139__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_language_11_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1144__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_language_18_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_spatial__1145__seed42 | Language Instructions | 5 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_language_19_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1150__seed42 | Language Instructions | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_language_27_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1162__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_language_41_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1166__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_language_46_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1181__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_language_21_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1182__seed42 | Language Instructions | 3 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_language_23_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1185__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_language_27_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |
| libero_spatial__1196__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_language_44_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 |
| libero_spatial__1202__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_language_2_view_0_0_100_0_0_initstate_0 | 1 | 0 | 1 | 1 | 1 | 1 | 0 |
| libero_spatial__1205__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_language_6_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1217__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_language_19_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| libero_spatial__1221__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_language_23_view_0_0_100_0_0_initstate_0 | 1 | 1 | 0 | 0 | 1 | 1 | 0 |
| libero_spatial__1225__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_language_27_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1227__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_language_29_view_0_0_100_0_0_initstate_0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 |
| libero_spatial__1232__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_language_34_view_0_0_100_0_0_initstate_0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 |
| libero_spatial__1240__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_language_42_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1242__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_language_44_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| libero_spatial__1247__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_1_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1248__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_2_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1249__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_3_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1250__seed42 | Language Instructions | 3 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_5_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1252__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_7_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1253__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_10_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1258__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_15_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1262__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_19_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1263__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_21_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1273__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_41_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1275__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_43_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1276__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_44_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1279__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_language_48_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1301__seed42 | Language Instructions | 3 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_language_21_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1304__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_language_24_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1305__seed42 | Language Instructions | 3 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_language_25_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1327__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_language_1_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1331__seed42 | Language Instructions | 2 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_language_5_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1332__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_language_6_view_0_0_100_0_0_initstate_0 | 0 | 0 | 1 | 1 | 0 | 1 | 1 |
| libero_spatial__1339__seed42 | Language Instructions | 1 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_language_14_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1351__seed42 | Language Instructions | 5 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_language_27_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1352__seed42 | Language Instructions | 5 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_language_28_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1364__seed42 | Language Instructions | 4 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_language_41_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1376__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1382__seed42 | Sensor Noise | 1 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_12 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1385__seed42 | Sensor Noise | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_15 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1395__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_38 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1401__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_46 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_spatial__1406__seed42 | Sensor Noise | 1 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1421__seed42 | Sensor Noise | 1 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1422__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1424__seed42 | Sensor Noise | 4 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1432__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_43 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1442__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1451__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_15 | 1 | 0 | 0 | 1 | 1 | 1 | 1 |
| libero_spatial__1453__seed42 | Sensor Noise | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_17 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |
| libero_spatial__1459__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_28 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_spatial__1466__seed42 | Sensor Noise | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_40 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1469__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_45 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_spatial__1472__seed42 | Sensor Noise | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_48 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1477__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_7 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1481__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_13 | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| libero_spatial__1506__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1509__seed42 | Sensor Noise | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1514__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1516__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1521__seed42 | Sensor Noise | 5 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1523__seed42 | Sensor Noise | 1 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1528__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1534__seed42 | Sensor Noise | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_38 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1540__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_45 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1542__seed42 | Sensor Noise | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_47 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1560__seed42 | Sensor Noise | 4 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_18 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |
| libero_spatial__1562__seed42 | Sensor Noise | 4 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_20 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_spatial__1565__seed42 | Sensor Noise | 3 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_28 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_spatial__1572__seed42 | Sensor Noise | 4 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1599__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_28 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_spatial__1609__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_43 | 0 | 0 | 1 | 1 | 1 | 1 | 0 |
| libero_spatial__1610__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_44 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| libero_spatial__1617__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1622__seed42 | Sensor Noise | 3 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1625__seed42 | Sensor Noise | 4 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1629__seed42 | Sensor Noise | 3 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1652__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_45 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1659__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_5 | 1 | 0 | 0 | 1 | 1 | 0 | 0 |
| libero_spatial__1668__seed42 | Sensor Noise | 4 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_17 | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| libero_spatial__1670__seed42 | Sensor Noise | 5 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_20 | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| libero_spatial__1673__seed42 | Sensor Noise | 3 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_29 | 0 | 0 | 1 | 1 | 1 | 1 | 0 |
| libero_spatial__1675__seed42 | Sensor Noise | 1 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_35 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1686__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1687__seed42 | Sensor Noise | 2 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1694__seed42 | Sensor Noise | 3 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1695__seed42 | Sensor Noise | 4 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1698__seed42 | Sensor Noise | 4 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1700__seed42 | Sensor Noise | 3 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1701__seed42 | Sensor Noise | 4 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1703__seed42 | Sensor Noise | 1 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1704__seed42 | Sensor Noise | 1 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_view_0_0_100_0_0_initstate_0_noise_26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1726__seed42 | Objects Layout | 5 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_add_11 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1729__seed42 | Objects Layout | 5 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_add_14 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1740__seed42 | Objects Layout | 5 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_add_26 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1743__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_add_29 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1744__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_add_30 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1749__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_level1_sample1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1762__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_level4_sample2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_spatial__1765__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_level5_sample1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1769__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_add_10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1783__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_add_29 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1792__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_level1_sample3 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1795__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_level2_sample2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1796__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_level2_sample3 | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| libero_spatial__1808__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_add_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1809__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_add_11 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_spatial__1812__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_add_15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1814__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_add_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1815__seed42 | Objects Layout | 5 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_add_18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1823__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_add_26 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_spatial__1826__seed42 | Objects Layout | 5 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_add_29 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| libero_spatial__1836__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_level1_sample4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1837__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_level2_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__1850__seed42 | Objects Layout | 5 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_level5_sample3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1854__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_add_14 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1869__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_add_7 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1876__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_level3_sample3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1877__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_level3_sample4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1888__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_add_25 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1896__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_level1_sample4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1907__seed42 | Objects Layout | 5 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_level4_sample4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1917__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_add_17 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| libero_spatial__1945__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_level5_sample2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_spatial__1950__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_add_12 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1954__seed42 | Objects Layout | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_add_17 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1957__seed42 | Objects Layout | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_add_20 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1962__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_add_25 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_spatial__1974__seed42 | Objects Layout | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_level1_sample2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1977__seed42 | Objects Layout | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_level2_sample1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1980__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_level2_sample4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1981__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_level3_sample1 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| libero_spatial__1986__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_level4_sample2 | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| libero_spatial__1989__seed42 | Objects Layout | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_level5_sample1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__1998__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_add_14 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_spatial__2001__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_add_18 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2010__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_add_30 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2018__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_level2_sample1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 |
| libero_spatial__2029__seed42 | Objects Layout | 5 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_level5_sample1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2033__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_add_12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__2034__seed42 | Objects Layout | 2 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_add_13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__2035__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_add_14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__2042__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_add_22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__2058__seed42 | Objects Layout | 2 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_level2_sample4 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_spatial__2069__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_add_12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__2073__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_add_18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__2078__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_add_22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__2086__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_add_5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__2089__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_add_8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__2102__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_level4_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__2104__seed42 | Objects Layout | 4 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_level4_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__2108__seed42 | Objects Layout | 3 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_level5_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_spatial__2112__seed42 | Light Conditions | 4 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_light_11 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2122__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_light_20 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| libero_spatial__2126__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_light_26 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2137__seed42 | Light Conditions | 4 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_light_38 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2141__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_light_43 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2142__seed42 | Light Conditions | 2 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_light_45 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2143__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_light_46 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2144__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_light_47 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2157__seed42 | Light Conditions | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_light_14 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2167__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_light_23 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2172__seed42 | Light Conditions | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_light_28 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| libero_spatial__2179__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_light_34 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2188__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_light_42 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2196__seed42 | Light Conditions | 2 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_light_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2200__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_light_8 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2207__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_light_14 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2209__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_light_16 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2212__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_light_19 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2217__seed42 | Light Conditions | 2 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_light_23 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2235__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_light_41 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_spatial__2258__seed42 | Light Conditions | 2 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_light_30 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2263__seed42 | Light Conditions | 1 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_light_36 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2278__seed42 | Light Conditions | 1 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_light_15 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2284__seed42 | Light Conditions | 1 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_light_38 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_spatial__2293__seed42 | Light Conditions | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_light_14 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2298__seed42 | Light Conditions | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_light_25 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2299__seed42 | Light Conditions | 1 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_light_26 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2305__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_light_41 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2306__seed42 | Light Conditions | 2 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_light_42 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2318__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_light_13 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2344__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_light_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2369__seed42 | Light Conditions | 2 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_light_38 | 1 | 0 | 0 | 1 | 0 | 1 | 1 |
| libero_spatial__2371__seed42 | Light Conditions | 2 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_light_4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2378__seed42 | Light Conditions | 2 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_light_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2383__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_light_12 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2391__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_light_24 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2393__seed42 | Light Conditions | 2 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_light_28 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_spatial__2396__seed42 | Light Conditions | 3 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_light_31 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0001__seed42 | Background Textures | 3 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_10 | 0 | 0 | 0 | 1 | 1 | 0 | 1 |
| libero_object__0004__seed42 | Background Textures | 3 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_14 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0006__seed42 | Background Textures | 2 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0007__seed42 | Background Textures | 3 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_17 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_object__0011__seed42 | Background Textures | 3 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_20 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| libero_object__0016__seed42 | Background Textures | 3 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_28 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_object__0019__seed42 | Background Textures | 3 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_table_9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0028__seed42 | Background Textures | 3 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_tb_6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0031__seed42 | Background Textures | 2 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_tb_9 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_object__0032__seed42 | Background Textures | 1 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_table_1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| libero_object__0033__seed42 | Background Textures | 1 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_table_11 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0039__seed42 | Background Textures | 2 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_table_17 | 0 | 0 | 1 | 0 | 1 | 1 | 0 |
| libero_object__0044__seed42 | Background Textures | 2 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_table_21 | 1 | 1 | 1 | 0 | 0 | 1 | 0 |
| libero_object__0047__seed42 | Background Textures | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_table_24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0049__seed42 | Background Textures | 2 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_table_26 | 1 | 0 | 0 | 1 | 0 | 1 | 0 |
| libero_object__0051__seed42 | Background Textures | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_table_28 | 1 | 1 | 0 | 0 | 1 | 1 | 0 |
| libero_object__0061__seed42 | Background Textures | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_tb_12 | 0 | 1 | 0 | 1 | 1 | 0 | 0 |
| libero_object__0070__seed42 | Background Textures | 2 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_tb_21 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0071__seed42 | Background Textures | 1 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_tb_22 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0079__seed42 | Background Textures | 4 | pick_up_the_butter_and_place_it_in_the_basket_table_12 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0080__seed42 | Background Textures | 4 | pick_up_the_butter_and_place_it_in_the_basket_table_14 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0089__seed42 | Background Textures | 2 | pick_up_the_butter_and_place_it_in_the_basket_table_27 | 0 | 0 | 1 | 1 | 0 | 0 | 1 |
| libero_object__0104__seed42 | Background Textures | 3 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_table_16 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_object__0111__seed42 | Background Textures | 1 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_table_27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0115__seed42 | Background Textures | 2 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_table_8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0116__seed42 | Background Textures | 2 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_tb_1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0120__seed42 | Background Textures | 1 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_tb_20 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0123__seed42 | Background Textures | 3 | pick_up_the_cream_cheese_and_place_it_in_the_basket_table_24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0124__seed42 | Background Textures | 3 | pick_up_the_cream_cheese_and_place_it_in_the_basket_table_3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0127__seed42 | Background Textures | 3 | pick_up_the_ketchup_and_place_it_in_the_basket_table_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0130__seed42 | Background Textures | 2 | pick_up_the_ketchup_and_place_it_in_the_basket_table_13 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0140__seed42 | Background Textures | 1 | pick_up_the_milk_and_place_it_in_the_basket_table_12 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_object__0146__seed42 | Background Textures | 4 | pick_up_the_milk_and_place_it_in_the_basket_table_26 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_object__0170__seed42 | Background Textures | 1 | pick_up_the_orange_juice_and_place_it_in_the_basket_table_1 | 0 | 1 | 1 | 0 | 1 | 0 | 1 |
| libero_object__0171__seed42 | Background Textures | 3 | pick_up_the_orange_juice_and_place_it_in_the_basket_table_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0174__seed42 | Background Textures | 3 | pick_up_the_orange_juice_and_place_it_in_the_basket_table_14 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| libero_object__0204__seed42 | Background Textures | 1 | pick_up_the_orange_juice_and_place_it_in_the_basket_tb_22 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0210__seed42 | Background Textures | 1 | pick_up_the_salad_dressing_and_place_it_in_the_basket_table_1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0230__seed42 | Background Textures | 1 | pick_up_the_salad_dressing_and_place_it_in_the_basket_tb_4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0232__seed42 | Background Textures | 1 | pick_up_the_salad_dressing_and_place_it_in_the_basket_tb_6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0252__seed42 | Robot Initial States | 1 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_101 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0257__seed42 | Robot Initial States | 2 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_171 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0265__seed42 | Robot Initial States | 3 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_251 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| libero_object__0268__seed42 | Robot Initial States | 3 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_281 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0275__seed42 | Robot Initial States | 3 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_351 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0278__seed42 | Robot Initial States | 4 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_381 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| libero_object__0282__seed42 | Robot Initial States | 4 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_421 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0291__seed42 | Robot Initial States | 2 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_42 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0292__seed42 | Robot Initial States | 3 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_62 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0301__seed42 | Robot Initial States | 3 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_202 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0303__seed42 | Robot Initial States | 2 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_232 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0304__seed42 | Robot Initial States | 2 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_242 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0305__seed42 | Robot Initial States | 3 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_252 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0309__seed42 | Robot Initial States | 4 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_302 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0329__seed42 | Robot Initial States | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0338__seed42 | Robot Initial States | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_153 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0351__seed42 | Robot Initial States | 4 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_303 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0360__seed42 | Robot Initial States | 5 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_403 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0370__seed42 | Robot Initial States | 1 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_74 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0378__seed42 | Robot Initial States | 2 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_204 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| libero_object__0387__seed42 | Robot Initial States | 5 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_304 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0390__seed42 | Robot Initial States | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_334 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| libero_object__0393__seed42 | Robot Initial States | 1 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_364 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0395__seed42 | Robot Initial States | 4 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_384 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0402__seed42 | Robot Initial States | 5 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_454 | 1 | 1 | 0 | 1 | 1 | 0 | 0 |
| libero_object__0412__seed42 | Robot Initial States | 2 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_145 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0413__seed42 | Robot Initial States | 2 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_155 | 0 | 1 | 1 | 1 | 0 | 1 | 0 |
| libero_object__0421__seed42 | Robot Initial States | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_265 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0425__seed42 | Robot Initial States | 4 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_305 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| libero_object__0426__seed42 | Robot Initial States | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_315 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0432__seed42 | Robot Initial States | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_375 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0434__seed42 | Robot Initial States | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_395 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0437__seed42 | Robot Initial States | 2 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_425 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_object__0444__seed42 | Robot Initial States | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_495 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0447__seed42 | Robot Initial States | 2 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_106 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0449__seed42 | Robot Initial States | 1 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_136 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0453__seed42 | Robot Initial States | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_186 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0456__seed42 | Robot Initial States | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_216 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_object__0464__seed42 | Robot Initial States | 3 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_296 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0493__seed42 | Robot Initial States | 4 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_167 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0500__seed42 | Robot Initial States | 5 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_257 | 1 | 0 | 1 | 0 | 0 | 0 | 1 |
| libero_object__0513__seed42 | Robot Initial States | 2 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_387 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0517__seed42 | Robot Initial States | 5 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_427 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_object__0518__seed42 | Robot Initial States | 4 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_437 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0521__seed42 | Robot Initial States | 1 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_467 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_object__0527__seed42 | Robot Initial States | 2 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_108 | 1 | 1 | 1 | 0 | 0 | 1 | 0 |
| libero_object__0533__seed42 | Robot Initial States | 2 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_178 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| libero_object__0550__seed42 | Robot Initial States | 5 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_358 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0566__seed42 | Robot Initial States | 2 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_129 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0567__seed42 | Robot Initial States | 4 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_139 | 1 | 1 | 0 | 0 | 1 | 1 | 1 |
| libero_object__0571__seed42 | Robot Initial States | 4 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_219 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0577__seed42 | Robot Initial States | 2 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_289 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0591__seed42 | Robot Initial States | 5 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_429 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0595__seed42 | Robot Initial States | 3 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_469 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0601__seed42 | Robot Initial States | 4 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_50 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_object__0611__seed42 | Robot Initial States | 2 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_150 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0617__seed42 | Robot Initial States | 3 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_210 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_object__0620__seed42 | Robot Initial States | 2 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_240 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0638__seed42 | Robot Initial States | 4 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_420 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0651__seed42 | Camera Viewpoints | 1 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_2_352_initstate_0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0654__seed42 | Camera Viewpoints | 1 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_2_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0677__seed42 | Camera Viewpoints | 2 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_19_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0687__seed42 | Camera Viewpoints | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_121_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0696__seed42 | Camera Viewpoints | 3 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_130_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0697__seed42 | Camera Viewpoints | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_100_6_2_initstate_0 | 0 | 1 | 0 | 1 | 0 | 0 | 1 |
| libero_object__0699__seed42 | Camera Viewpoints | 5 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_100_6_6_initstate_0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0718__seed42 | Camera Viewpoints | 1 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_38_0_100_0_0_initstate_0 | 0 | 0 | 1 | 0 | 1 | 1 | 1 |
| libero_object__0719__seed42 | Camera Viewpoints | 1 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_38_15_100_0_0_initstate_0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| libero_object__0723__seed42 | Camera Viewpoints | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_40_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0724__seed42 | Camera Viewpoints | 1 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_41_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0728__seed42 | Camera Viewpoints | 3 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_43_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_object__0733__seed42 | Camera Viewpoints | 4 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_45_15_100_0_0_initstate_0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 |
| libero_object__0734__seed42 | Camera Viewpoints | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_132_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0760__seed42 | Camera Viewpoints | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_50_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| libero_object__0763__seed42 | Camera Viewpoints | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_52_0_100_0_0_initstate_0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 |
| libero_object__0770__seed42 | Camera Viewpoints | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_55_15_100_0_0_initstate_0 | 0 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_object__0778__seed42 | Camera Viewpoints | 2 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_59_15_100_0_0_initstate_0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 |
| libero_object__0782__seed42 | Camera Viewpoints | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_142_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0783__seed42 | Camera Viewpoints | 4 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_143_0_0_initstate_0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_object__0792__seed42 | Camera Viewpoints | 3 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_10_4_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0798__seed42 | Camera Viewpoints | 2 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_10_354_initstate_0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_object__0806__seed42 | Camera Viewpoints | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_view_64_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0831__seed42 | Camera Viewpoints | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_154_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0835__seed42 | Camera Viewpoints | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_158_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0840__seed42 | Camera Viewpoints | 2 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_350_6_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0847__seed42 | Camera Viewpoints | 2 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_350_358_initstate_0 | 1 | 0 | 0 | 1 | 1 | 0 | 1 |
| libero_object__0861__seed42 | Camera Viewpoints | 4 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_291_15_100_0_0_initstate_0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 |
| libero_object__0865__seed42 | Camera Viewpoints | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_293_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0874__seed42 | Camera Viewpoints | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_298_0_100_0_0_initstate_0 | 0 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_object__0884__seed42 | Camera Viewpoints | 4 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_167_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0887__seed42 | Camera Viewpoints | 4 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_170_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0891__seed42 | Camera Viewpoints | 2 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_352_10_initstate_0 | 0 | 1 | 0 | 1 | 0 | 1 | 1 |
| libero_object__0892__seed42 | Camera Viewpoints | 4 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_352_350_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0895__seed42 | Camera Viewpoints | 2 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_352_356_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0897__seed42 | Camera Viewpoints | 5 | pick_up_the_butter_and_place_it_in_the_basket_view_300_0_100_0_0_initstate_0 | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| libero_object__0898__seed42 | Camera Viewpoints | 5 | pick_up_the_butter_and_place_it_in_the_basket_view_300_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0902__seed42 | Camera Viewpoints | 5 | pick_up_the_butter_and_place_it_in_the_basket_view_302_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0903__seed42 | Camera Viewpoints | 5 | pick_up_the_butter_and_place_it_in_the_basket_view_303_0_100_0_0_initstate_0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| libero_object__0929__seed42 | Camera Viewpoints | 5 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_173_0_0_initstate_0 | 0 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_object__0932__seed42 | Camera Viewpoints | 5 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_176_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| libero_object__0933__seed42 | Camera Viewpoints | 5 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_177_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0935__seed42 | Camera Viewpoints | 5 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_179_0_0_initstate_0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| libero_object__0939__seed42 | Camera Viewpoints | 4 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_100_354_6_initstate_0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0940__seed42 | Camera Viewpoints | 5 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_100_354_10_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0951__seed42 | Camera Viewpoints | 5 | pick_up_the_milk_and_place_it_in_the_basket_view_317_15_100_0_0_initstate_0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_object__0953__seed42 | Camera Viewpoints | 5 | pick_up_the_milk_and_place_it_in_the_basket_view_318_15_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_object__0958__seed42 | Camera Viewpoints | 5 | pick_up_the_milk_and_place_it_in_the_basket_view_321_0_100_0_0_initstate_0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
| libero_object__0963__seed42 | Camera Viewpoints | 5 | pick_up_the_milk_and_place_it_in_the_basket_view_323_15_100_0_0_initstate_0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| libero_object__0975__seed42 | Camera Viewpoints | 4 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_181_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0981__seed42 | Camera Viewpoints | 5 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_187_0_0_initstate_0 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_object__0988__seed42 | Camera Viewpoints | 1 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_356_354_initstate_0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__0995__seed42 | Camera Viewpoints | 1 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_333_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1006__seed42 | Camera Viewpoints | 4 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_338_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1008__seed42 | Camera Viewpoints | 3 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_339_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1026__seed42 | Camera Viewpoints | 1 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_358_2_initstate_0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_object__1027__seed42 | Camera Viewpoints | 4 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_358_6_initstate_0 | 1 | 0 | 1 | 0 | 0 | 1 | 1 |
| libero_object__1029__seed42 | Camera Viewpoints | 4 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_358_10_initstate_0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| libero_object__1034__seed42 | Camera Viewpoints | 1 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_345_15_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1052__seed42 | Language Instructions | 1 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_language_24_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1073__seed42 | Language Instructions | 3 | pick_up_the_cream_cheese_and_place_it_in_the_basket_language_14_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1076__seed42 | Language Instructions | 4 | pick_up_the_cream_cheese_and_place_it_in_the_basket_language_18_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1078__seed42 | Language Instructions | 5 | pick_up_the_cream_cheese_and_place_it_in_the_basket_language_21_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1079__seed42 | Language Instructions | 1 | pick_up_the_cream_cheese_and_place_it_in_the_basket_language_24_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1081__seed42 | Language Instructions | 4 | pick_up_the_cream_cheese_and_place_it_in_the_basket_language_26_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1101__seed42 | Language Instructions | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_language_5_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1116__seed42 | Language Instructions | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_language_28_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1123__seed42 | Language Instructions | 1 | pick_up_the_salad_dressing_and_place_it_in_the_basket_language_37_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1136__seed42 | Language Instructions | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_language_4_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1138__seed42 | Language Instructions | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_language_6_view_0_0_100_0_0_initstate_0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1140__seed42 | Language Instructions | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_language_15_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1141__seed42 | Language Instructions | 2 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_language_16_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1147__seed42 | Language Instructions | 1 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_language_28_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1148__seed42 | Language Instructions | 4 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_language_30_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1149__seed42 | Language Instructions | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_language_31_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1150__seed42 | Language Instructions | 1 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_language_32_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1153__seed42 | Language Instructions | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_language_42_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1156__seed42 | Language Instructions | 1 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_language_45_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1157__seed42 | Language Instructions | 1 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_language_47_view_0_0_100_0_0_initstate_0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 |
| libero_object__1165__seed42 | Language Instructions | 1 | pick_up_the_ketchup_and_place_it_in_the_basket_language_7_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1173__seed42 | Language Instructions | 4 | pick_up_the_ketchup_and_place_it_in_the_basket_language_18_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1188__seed42 | Language Instructions | 4 | pick_up_the_ketchup_and_place_it_in_the_basket_language_38_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1202__seed42 | Language Instructions | 2 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_language_6_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1212__seed42 | Language Instructions | 1 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_language_18_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1225__seed42 | Language Instructions | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_language_34_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1230__seed42 | Language Instructions | 3 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_language_43_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1235__seed42 | Language Instructions | 5 | pick_up_the_butter_and_place_it_in_the_basket_language_4_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1237__seed42 | Language Instructions | 2 | pick_up_the_butter_and_place_it_in_the_basket_language_6_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1238__seed42 | Language Instructions | 3 | pick_up_the_butter_and_place_it_in_the_basket_language_7_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1249__seed42 | Language Instructions | 1 | pick_up_the_butter_and_place_it_in_the_basket_language_27_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1256__seed42 | Language Instructions | 1 | pick_up_the_butter_and_place_it_in_the_basket_language_42_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1268__seed42 | Language Instructions | 1 | pick_up_the_milk_and_place_it_in_the_basket_language_7_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1269__seed42 | Language Instructions | 5 | pick_up_the_milk_and_place_it_in_the_basket_language_9_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1277__seed42 | Language Instructions | 5 | pick_up_the_milk_and_place_it_in_the_basket_language_17_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1285__seed42 | Language Instructions | 2 | pick_up_the_milk_and_place_it_in_the_basket_language_27_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1291__seed42 | Language Instructions | 5 | pick_up_the_milk_and_place_it_in_the_basket_language_36_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1295__seed42 | Language Instructions | 1 | pick_up_the_milk_and_place_it_in_the_basket_language_42_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1300__seed42 | Language Instructions | 2 | pick_up_the_milk_and_place_it_in_the_basket_language_47_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1302__seed42 | Language Instructions | 2 | pick_up_the_milk_and_place_it_in_the_basket_language_49_view_0_0_100_0_0_initstate_0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 |
| libero_object__1305__seed42 | Language Instructions | 1 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_language_2_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1309__seed42 | Language Instructions | 3 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_language_6_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1320__seed42 | Language Instructions | 2 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_language_18_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1322__seed42 | Language Instructions | 4 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_language_20_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1324__seed42 | Language Instructions | 4 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_language_22_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1327__seed42 | Language Instructions | 2 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_language_25_view_0_0_100_0_0_initstate_0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| libero_object__1338__seed42 | Language Instructions | 4 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_language_36_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1355__seed42 | Language Instructions | 2 | pick_up_the_orange_juice_and_place_it_in_the_basket_language_4_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1361__seed42 | Language Instructions | 3 | pick_up_the_orange_juice_and_place_it_in_the_basket_language_12_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1365__seed42 | Language Instructions | 5 | pick_up_the_orange_juice_and_place_it_in_the_basket_language_16_view_0_0_100_0_0_initstate_0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1372__seed42 | Language Instructions | 5 | pick_up_the_orange_juice_and_place_it_in_the_basket_language_23_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1377__seed42 | Language Instructions | 5 | pick_up_the_orange_juice_and_place_it_in_the_basket_language_28_view_0_0_100_0_0_initstate_0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1412__seed42 | Sensor Noise | 5 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_19 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1419__seed42 | Sensor Noise | 3 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_28 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_object__1427__seed42 | Sensor Noise | 4 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_40 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| libero_object__1436__seed42 | Sensor Noise | 4 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_49 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1437__seed42 | Sensor Noise | 4 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_50 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| libero_object__1447__seed42 | Sensor Noise | 5 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_10 | 1 | 1 | 1 | 0 | 0 | 1 | 0 |
| libero_object__1450__seed42 | Sensor Noise | 2 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1452__seed42 | Sensor Noise | 4 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1454__seed42 | Sensor Noise | 5 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1455__seed42 | Sensor Noise | 5 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1461__seed42 | Sensor Noise | 2 | pick_up_the_cream_cheese_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_26 | 1 | 0 | 0 | 0 | 1 | 1 | 0 |
| libero_object__1490__seed42 | Sensor Noise | 3 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1494__seed42 | Sensor Noise | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1497__seed42 | Sensor Noise | 5 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1498__seed42 | Sensor Noise | 4 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1516__seed42 | Sensor Noise | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_43 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| libero_object__1517__seed42 | Sensor Noise | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_44 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1525__seed42 | Sensor Noise | 2 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1533__seed42 | Sensor Noise | 2 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_13 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| libero_object__1534__seed42 | Sensor Noise | 2 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1535__seed42 | Sensor Noise | 2 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_15 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_object__1546__seed42 | Sensor Noise | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_29 | 0 | 0 | 1 | 1 | 1 | 1 | 0 |
| libero_object__1552__seed42 | Sensor Noise | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_38 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_object__1570__seed42 | Sensor Noise | 3 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1574__seed42 | Sensor Noise | 4 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1580__seed42 | Sensor Noise | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1590__seed42 | Sensor Noise | 4 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_29 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| libero_object__1602__seed42 | Sensor Noise | 3 | pick_up_the_ketchup_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_45 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1610__seed42 | Sensor Noise | 2 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_3 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_object__1611__seed42 | Sensor Noise | 2 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_4 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| libero_object__1612__seed42 | Sensor Noise | 2 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_5 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| libero_object__1621__seed42 | Sensor Noise | 3 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1623__seed42 | Sensor Noise | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1625__seed42 | Sensor Noise | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_20 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| libero_object__1626__seed42 | Sensor Noise | 2 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_23 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| libero_object__1663__seed42 | Sensor Noise | 2 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1670__seed42 | Sensor Noise | 1 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_22 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1677__seed42 | Sensor Noise | 4 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1678__seed42 | Sensor Noise | 4 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1679__seed42 | Sensor Noise | 1 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_33 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1682__seed42 | Sensor Noise | 2 | pick_up_the_butter_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_36 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1700__seed42 | Sensor Noise | 2 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_6 | 1 | 0 | 1 | 0 | 1 | 1 | 1 |
| libero_object__1701__seed42 | Sensor Noise | 2 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_7 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| libero_object__1707__seed42 | Sensor Noise | 4 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1710__seed42 | Sensor Noise | 5 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_17 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1711__seed42 | Sensor Noise | 5 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_19 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1715__seed42 | Sensor Noise | 2 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1721__seed42 | Sensor Noise | 2 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_35 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| libero_object__1729__seed42 | Sensor Noise | 2 | pick_up_the_milk_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_45 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| libero_object__1739__seed42 | Sensor Noise | 4 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_6 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_object__1749__seed42 | Sensor Noise | 5 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1751__seed42 | Sensor Noise | 5 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_19 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| libero_object__1754__seed42 | Sensor Noise | 1 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_26 | 1 | 1 | 1 | 1 | 0 | 0 | 1 |
| libero_object__1762__seed42 | Sensor Noise | 2 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_36 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1774__seed42 | Sensor Noise | 4 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1776__seed42 | Sensor Noise | 3 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_50 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1780__seed42 | Sensor Noise | 4 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_5 | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| libero_object__1789__seed42 | Sensor Noise | 4 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1793__seed42 | Sensor Noise | 5 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_18 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1794__seed42 | Sensor Noise | 5 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_19 | 0 | 1 | 1 | 0 | 1 | 0 | 0 |
| libero_object__1800__seed42 | Sensor Noise | 4 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1803__seed42 | Sensor Noise | 2 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_35 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| libero_object__1808__seed42 | Sensor Noise | 5 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_40 | 0 | 1 | 0 | 1 | 0 | 1 | 0 |
| libero_object__1810__seed42 | Sensor Noise | 3 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_43 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
| libero_object__1812__seed42 | Sensor Noise | 3 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_45 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1813__seed42 | Sensor Noise | 4 | pick_up_the_orange_juice_and_place_it_in_the_basket_view_0_0_100_0_0_initstate_0_noise_46 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| libero_object__1818__seed42 | Objects Layout | 1 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_add_10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1828__seed42 | Objects Layout | 1 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_add_21 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1831__seed42 | Objects Layout | 1 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_add_25 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1836__seed42 | Objects Layout | 1 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_add_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1839__seed42 | Objects Layout | 4 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_level1_sample1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1840__seed42 | Objects Layout | 4 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_level1_sample2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1841__seed42 | Objects Layout | 4 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_level1_sample3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1842__seed42 | Objects Layout | 4 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_level1_sample4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1844__seed42 | Objects Layout | 4 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_level2_sample2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1858__seed42 | Objects Layout | 5 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_level5_sample4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1863__seed42 | Objects Layout | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_add_15 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1888__seed42 | Objects Layout | 5 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_level2_sample2 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| libero_object__1893__seed42 | Objects Layout | 5 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_level3_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1900__seed42 | Objects Layout | 5 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_level5_sample2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__1907__seed42 | Objects Layout | 1 | pick_up_the_butter_and_place_it_in_the_basket_add_15 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1919__seed42 | Objects Layout | 4 | pick_up_the_butter_and_place_it_in_the_basket_add_28 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1925__seed42 | Objects Layout | 1 | pick_up_the_butter_and_place_it_in_the_basket_add_8 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1926__seed42 | Objects Layout | 1 | pick_up_the_butter_and_place_it_in_the_basket_add_9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1931__seed42 | Objects Layout | 4 | pick_up_the_butter_and_place_it_in_the_basket_level2_sample1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1934__seed42 | Objects Layout | 3 | pick_up_the_butter_and_place_it_in_the_basket_level2_sample4 | 1 | 1 | 1 | 0 | 1 | 1 | 1 |
| libero_object__1954__seed42 | Objects Layout | 1 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_add_22 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1974__seed42 | Objects Layout | 5 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_level3_sample2 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| libero_object__1985__seed42 | Objects Layout | 1 | pick_up_the_cream_cheese_and_place_it_in_the_basket_add_14 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__1996__seed42 | Objects Layout | 5 | pick_up_the_cream_cheese_and_place_it_in_the_basket_level2_sample1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2000__seed42 | Objects Layout | 5 | pick_up_the_cream_cheese_and_place_it_in_the_basket_level3_sample1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2001__seed42 | Objects Layout | 5 | pick_up_the_cream_cheese_and_place_it_in_the_basket_level3_sample2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2018__seed42 | Objects Layout | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_level1_sample2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2024__seed42 | Objects Layout | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_level2_sample4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2031__seed42 | Objects Layout | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_level4_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__2048__seed42 | Objects Layout | 3 | pick_up_the_milk_and_place_it_in_the_basket_add_22 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2055__seed42 | Objects Layout | 2 | pick_up_the_milk_and_place_it_in_the_basket_add_29 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2058__seed42 | Objects Layout | 3 | pick_up_the_milk_and_place_it_in_the_basket_add_4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2059__seed42 | Objects Layout | 3 | pick_up_the_milk_and_place_it_in_the_basket_add_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2072__seed42 | Objects Layout | 5 | pick_up_the_milk_and_place_it_in_the_basket_level3_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__2094__seed42 | Objects Layout | 2 | pick_up_the_orange_juice_and_place_it_in_the_basket_add_20 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2095__seed42 | Objects Layout | 3 | pick_up_the_orange_juice_and_place_it_in_the_basket_add_21 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2106__seed42 | Objects Layout | 3 | pick_up_the_orange_juice_and_place_it_in_the_basket_add_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2108__seed42 | Objects Layout | 3 | pick_up_the_orange_juice_and_place_it_in_the_basket_add_7 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2111__seed42 | Objects Layout | 4 | pick_up_the_orange_juice_and_place_it_in_the_basket_level1_sample1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2112__seed42 | Objects Layout | 5 | pick_up_the_orange_juice_and_place_it_in_the_basket_level1_sample2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2121__seed42 | Objects Layout | 5 | pick_up_the_orange_juice_and_place_it_in_the_basket_level3_sample3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__2128__seed42 | Objects Layout | 5 | pick_up_the_orange_juice_and_place_it_in_the_basket_level5_sample2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__2131__seed42 | Objects Layout | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_add_10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2132__seed42 | Objects Layout | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_add_12 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2148__seed42 | Objects Layout | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_add_29 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2160__seed42 | Objects Layout | 4 | pick_up_the_salad_dressing_and_place_it_in_the_basket_level2_sample1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2180__seed42 | Objects Layout | 2 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_add_16 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2187__seed42 | Objects Layout | 2 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_add_22 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2202__seed42 | Objects Layout | 2 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_add_9 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2206__seed42 | Objects Layout | 3 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_level1_sample4 | 0 | 0 | 1 | 0 | 1 | 1 | 1 |
| libero_object__2207__seed42 | Objects Layout | 4 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_level2_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__2211__seed42 | Objects Layout | 4 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_level3_sample1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__2221__seed42 | Light Conditions | 5 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_light_1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| libero_object__2226__seed42 | Light Conditions | 5 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_light_14 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2230__seed42 | Light Conditions | 5 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_light_18 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2234__seed42 | Light Conditions | 5 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_light_21 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2243__seed42 | Light Conditions | 5 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_light_31 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2245__seed42 | Light Conditions | 4 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_light_35 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2249__seed42 | Light Conditions | 4 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_light_41 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2256__seed42 | Light Conditions | 5 | pick_up_the_alphabet_soup_and_place_it_in_the_basket_light_49 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2265__seed42 | Light Conditions | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_light_12 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2269__seed42 | Light Conditions | 4 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_light_16 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2284__seed42 | Light Conditions | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_light_3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2291__seed42 | Light Conditions | 4 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_light_36 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2300__seed42 | Light Conditions | 3 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_light_44 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2308__seed42 | Light Conditions | 4 | pick_up_the_bbq_sauce_and_place_it_in_the_basket_light_6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2311__seed42 | Light Conditions | 4 | pick_up_the_butter_and_place_it_in_the_basket_light_10 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2314__seed42 | Light Conditions | 5 | pick_up_the_butter_and_place_it_in_the_basket_light_13 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2321__seed42 | Light Conditions | 5 | pick_up_the_butter_and_place_it_in_the_basket_light_2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2326__seed42 | Light Conditions | 4 | pick_up_the_butter_and_place_it_in_the_basket_light_24 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2335__seed42 | Light Conditions | 4 | pick_up_the_butter_and_place_it_in_the_basket_light_33 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2342__seed42 | Light Conditions | 5 | pick_up_the_butter_and_place_it_in_the_basket_light_4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2343__seed42 | Light Conditions | 4 | pick_up_the_butter_and_place_it_in_the_basket_light_40 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2345__seed42 | Light Conditions | 5 | pick_up_the_butter_and_place_it_in_the_basket_light_42 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2353__seed42 | Light Conditions | 1 | pick_up_the_butter_and_place_it_in_the_basket_light_6 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2366__seed42 | Light Conditions | 4 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_light_27 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2368__seed42 | Light Conditions | 3 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_light_30 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2379__seed42 | Light Conditions | 4 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket_light_41 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2393__seed42 | Light Conditions | 5 | pick_up_the_ketchup_and_place_it_in_the_basket_light_2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2397__seed42 | Light Conditions | 3 | pick_up_the_ketchup_and_place_it_in_the_basket_light_39 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2401__seed42 | Light Conditions | 4 | pick_up_the_ketchup_and_place_it_in_the_basket_light_50 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2404__seed42 | Light Conditions | 4 | pick_up_the_milk_and_place_it_in_the_basket_light_1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2417__seed42 | Light Conditions | 4 | pick_up_the_milk_and_place_it_in_the_basket_light_38 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2419__seed42 | Light Conditions | 4 | pick_up_the_milk_and_place_it_in_the_basket_light_41 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2420__seed42 | Light Conditions | 3 | pick_up_the_milk_and_place_it_in_the_basket_light_42 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2425__seed42 | Light Conditions | 4 | pick_up_the_milk_and_place_it_in_the_basket_light_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2426__seed42 | Light Conditions | 4 | pick_up_the_milk_and_place_it_in_the_basket_light_50 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2429__seed42 | Light Conditions | 5 | pick_up_the_milk_and_place_it_in_the_basket_light_8 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2440__seed42 | Light Conditions | 3 | pick_up_the_orange_juice_and_place_it_in_the_basket_light_20 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2443__seed42 | Light Conditions | 4 | pick_up_the_orange_juice_and_place_it_in_the_basket_light_24 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2447__seed42 | Light Conditions | 3 | pick_up_the_orange_juice_and_place_it_in_the_basket_light_28 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2452__seed42 | Light Conditions | 4 | pick_up_the_orange_juice_and_place_it_in_the_basket_light_36 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2454__seed42 | Light Conditions | 4 | pick_up_the_orange_juice_and_place_it_in_the_basket_light_41 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2463__seed42 | Light Conditions | 4 | pick_up_the_orange_juice_and_place_it_in_the_basket_light_8 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2481__seed42 | Light Conditions | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_light_35 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2482__seed42 | Light Conditions | 2 | pick_up_the_salad_dressing_and_place_it_in_the_basket_light_36 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2487__seed42 | Light Conditions | 3 | pick_up_the_salad_dressing_and_place_it_in_the_basket_light_40 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2488__seed42 | Light Conditions | 4 | pick_up_the_salad_dressing_and_place_it_in_the_basket_light_43 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2496__seed42 | Light Conditions | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_light_1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2497__seed42 | Light Conditions | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_light_11 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2503__seed42 | Light Conditions | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_light_20 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| libero_object__2516__seed42 | Light Conditions | 5 | pick_up_the_tomato_sauce_and_place_it_in_the_basket_light_5 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

</details>

The same task-level outcomes are available as [task_results.csv](task_results.csv).
