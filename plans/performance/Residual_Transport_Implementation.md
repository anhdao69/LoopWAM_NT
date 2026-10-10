# Residual Transport implementation report

Implementation is complete for the 20 executable recipes in the experiment matrix. Training completion, checkpoint publication, and simulator evaluation are separate milestones. R-FM remains undefined in the specification and is deferred.

This is the current report index. Earlier launch reports preserve the chronology and should be read as dated snapshots.

- [Editable results tables](Residual_Transport_Results.md)
- [Detailed results CSV](Residual_Transport_Results.csv)
- [Original specification](../residual_transport.md)
- [Initial implementation and phase-1 launch report](Residual_Transport_Launch_20261010.md)
- [Single-GPU support and phase-2 launch report](Residual_Transport_Phase2_Launch_20261010.md)
- [Verification evidence](../residual_transport_evidence/)
- [Completed implementation checklist](../../docs/superpowers/plans/2026-10-10-residual-transport.md)

## Repository and provenance

Branch: **residual_trans** in **anhdao69/LoopWAM_NT**. Implementation and training were performed remotely in /mnt/data/vmo-ai-task/anhdh35/LoopWAM_NT.

The first five jobs use immutable source f669184242b5. RT-A2 and RT-B2a use a7403b5aab91, which adds explicit one-GPU support. Later documentation commits do not change the source used by existing jobs.

The parent is the full four-suite v0 4/4 checkpoint from FastWAM/runs/loopwam_nt/v0_full_libero_job4659_20261006/train/latest.pt.

| Asset | SHA256 |
|---|---|
| Frozen parent | 4ffe1b14f35484ea582189ba0f06c5a41ac199e1fa733b491719727105e5e5cc |
| Canonical normalization | d5ad351eb3272f416d34082e8af5483ba1118ceca355a37d9f268fdb269fb3e4 |
| Wan VAE | 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981 |

## Architecture implemented

The frozen parent supplies video features and an independent frozen action teacher. A student action expert reuses the previous denoising step's core residual at the action pre/core boundary. Four-loop steps always start cold; shorter steps apply the configured transport. Video features use the correct late-loop alignment. Text attention keys and values are cached.

| Variant | Implementation |
|---|---|
| T0 | Cold start, zero transport |
| T1 | Identity reuse of the previous residual |
| T2 | First-order residual forecast; identity when only one history state exists |
| T3 | Learned per-channel time gate, initialized near 0.95 |
| T4 | T3 plus rank-32 time-conditioned residual correction; zero output initialization |
| T5 | T4 with the frozen embedding of the actual Euler delta |
| T6 | Cross-attention from current pre-core features to the previous residual; zero output projection |

P-a trains transport alone with no anchor. P-b adds rank-16, alpha-16 LoRA to the six action core blocks, covering self-attention, cross-attention and feed-forward projections. P-c trains the full action expert. Optional zero-initialized step-size conditioning supports variable NFE.

FU and TF training regimes are implemented. Default objective weights are local 1.0, endpoint 0.5, command-space gripper 0.1 and original flow-matching anchor 0.5. Endpoint supervision uses the first ten actions and backpropagates through the last three denoising steps. Teacher targets are detached; truncated history includes every stored field. Native teacher time uses 0–1000 units; transport time uses normalized tau.

RT+B2 validates a final RT-B2a epoch-10 checkpoint, inherits its action weights and step conditioning, installs fresh LoRA, and retains the original frozen teacher.

## Data and optimization

All runs use 277,713 cached windows from 1,712 demonstrations across all four LIBERO suites, without a subset. Cached first-frame latents, text context and canonical normalization are reused.

The production budget is **10 epochs, global batch 128, 21,700 optimizer updates**, totaling 2,777,130 real window presentations. Each epoch's final batch contains 81 real windows; its gradient is weighted by the real count. This user-approved budget supersedes the specification's older 6,000-update example.

Training uses FP32 parameters and optimizer state, BF16 compute, fused AdamW, a 3% warmup followed by cosine decay, gradient clipping at 1, EMA 0.999, deterministic algorithms, and per-window noise seeds. Two-GPU runs use DDP. RT-A2 uses one GPU, microbatch 64 and two accumulation passes; two-GPU submitted jobs use microbatch 64 and one pass per rank. Global sample ordering is preserved across layouts.

## Recovery, inference, and publication

- Atomic recovery at every 100 updates, epoch boundaries and termination signals includes weights, optimizer, EMA, schedule/cursor state, RNG states and the data/configuration contract.
- Per-output locks prevent simultaneous writers. Wrappers retry failures and resume from latest.pt; batch jobs support preemption recovery.
- Portable checkpoints are retained separately at update 2,000 and epochs 8, 9 and 10.
- Portable exports contain both a merged backbone and the unmerged trained student. RT inference loads the unmerged student to preserve BF16 operation order: merging adapters caused measurable rounding drift in a trained smoke.
- The native evaluation adapter supports explicit transport schedules and zero-shot T0/T1/T2 baselines. Eager and per-schedule CUDA graph paths are implemented.
- Uploaders publish portable checkpoints and provenance/configuration to public anhdao69/ResidualTrans, retry failures, verify remote size/LFS SHA256 and write receipts. Optimizer/RNG recovery files and synthetic smoke weights are excluded. Publication is complete only when the corresponding receipt exists.

## Code map

| Responsibility | Source |
|---|---|
| T0–T6, LoRA, schedules, step conditioning, native action core | [residual_transport.py](../../src/fastwam/models/wan22/residual_transport.py) |
| Teacher/student rollout, losses, scopes, portable export | [transport_policy.py](../../src/fastwam/models/wan22/transport_policy.py) |
| Native RT evaluation and CUDA graphs | [transport_inference.py](../../src/fastwam/models/wan22/transport_inference.py) |
| Full cached dataset | [transport_cached.py](../../src/fastwam/datasets/transport_cached.py) |
| Exact batching, optimization, recovery and checkpoints | [train_transport.py](../../scripts/train_transport.py) |
| 20 recipes and scientific gates | [transport_experiments.py](../../scripts/transport_experiments.py) |
| Submission, GPU sizing, overrides and deduplication | [submit_transport.py](../../scripts/submit_transport.py) |
| Slurm resources and runtime recovery | [submit_transport.sbatch](../../scripts/submit_transport.sbatch), [run_transport_job.sh](../../scripts/run_transport_job.sh) |
| Verified public upload | [upload_transport.py](../../scripts/upload_transport.py) |
| Evaluation entry point | [evaluate_loopwam_libero.py](../../scripts/evaluate_loopwam_libero.py) |
| Focused regression coverage | [test_residual_transport.py](../../tests/test_residual_transport.py), [test_transport_training.py](../../tests/test_transport_training.py) |

## Recorded verification and limits

These are results from the committed implementation evidence, not new test executions for this documentation update.

| Check | Recorded evidence/result |
|---|---|
| Supported project suite | project_pytest.log: 328 passed, 7 skipped |
| Focused suite after single-GPU support | single_gpu_unit.log: 30 passed |
| Dataset provenance and decoding | data_verification.json: all-window counts, 16 native decoded-window matches, 1,000 gripper signs |
| Native recipes | native_variants.json: finite forward/backward for all 20 recipes and configured schedules; cold-parent parity |
| Two-GPU recovery | resume_verification.json: ten resumed updates bit-exact for weights, Adam, EMA, loss and gradient norm |
| Single-GPU accumulation/recovery | single_resume_verification.json: ten resumed updates bit-exact |
| CUDA graph equivalence | graph_verification.json: 26 cases, 50 changing queries each, zero observed eager/graph error |
| Export and B2a inheritance | export_verification.json: exact trained-student reload and correct inheritance with original teacher preserved |
| Batch layout parity | layout_verification.json: 2 versus 4 logical ranks on 2 physical GPUs; no four-physical-GPU NCCL claim |
| RT-A2 capacity | single_gpu_A2_verification.json: 20 global-128 updates over six schedules; peak 83.83 decimal GB, within 80 GiB H100 memory |
| RT-B2a capacity | capacity_RT-B2a_64.json: native forward/backward/optimizer checks at microbatch 64; peak 44.23 decimal GB before DDP overhead |
| Production and handoff | production_first20.json, single_gpu_production_first20.json, single_gpu_batch_handoff.json |

The full project suite was recorded before the single-GPU additions; only the focused suite was rerun afterward. Unscoped pytest additionally collects optional RoboTwin scripts with unavailable dependencies; the supported suite is tests/. RT-B2a's capacity probe is not evidence that its queued full production DDP run has completed. No full simulator evaluation was performed on the H100 training cluster. Training loss and smoke parity do not establish policy success or a scientific gate.

## Submission snapshot

Captured **2026-10-10T20:30:56.678497+00:00**. See [machine-readable snapshot](../residual_transport_evidence/report_submission_snapshot.json). Queue status changes after capture.

| Run | Job | GPUs | State at capture | Source |
|---|---:|---:|---|---|
| RT-A | 4847 | 2 | RUNNING | f669184242b5 |
| RT-B4 | 4846 | 2 | RUNNING | f669184242b5 |
| RT-T1ft | 4849 | 2 | PENDING | f669184242b5 |
| RT-Pa | 4850 | 2 | PENDING | f669184242b5 |
| RT-TF | 4851 | 2 | PENDING | f669184242b5 |
| RT-A2 | 4861 | 1 | RUNNING | a7403b5aab91 |
| RT-B2a | 4859 | 2 | PENDING | a7403b5aab91 |

All seven jobs are standalone with no Slurm dependencies. The user explicitly overrode evaluation waits for these seven submissions. This does not mark Gate 0 or Gate 1 as passed. The remaining 13 executable recipes retain their gates; RT+B2 also needs the final B2a checkpoint.

RT-A resumed from update 220 after its earlier interactive step. RT-A2 checkpointed at update 58, moved to batch 4861 with a 48-hour limit, and interactive allocation 4857 was canceled at the user's request. The reports' older interactive instructions are historical.

## Remaining experimental work

1. Let the submitted jobs finish and check epoch-8/9/10 upload receipts.
2. Evaluate the update-2,000 checkpoints on Long for an early read; do not use early scores for gate decisions.
3. Evaluate retained final checkpoints on the separate evaluation server using the matched protocol in the results template.
4. Fill the results tables, paired confidence intervals and scientific decisions.
5. Submit gated recipes only after the required evidence or a further explicit user override. R-FM requires a defined target and endpoint before implementation.
