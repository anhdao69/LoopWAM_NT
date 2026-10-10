# Residual Transport launch — 2026-10-10

## Scope and decisions

All implementation, tests and training ran remotely in /mnt/data/vmo-ai-task/anhdh35/LoopWAM_NT, branch residual_trans. The user explicitly retained evaluation gates: **only RT-A and RT-B4 are eligible now**. No simulator evaluation was run on H100. Other recipes are implemented and tested but await external evidence. R-FM remains deferred: the specification provides no target/endpoint definition, so it is not an executable recipe.

Ten epochs replace the old6,000-update budget. Every production run uses all277,713 windows from1,712 locally available demonstrations, global batch 128, exactly two H100s, and21,700 updates (2,777,130 window presentations). Each epoch's final batch contains 81 real windows and is weighted by its real count. Separate epoch8/9/10 checkpoints and an update-2,000 early-read checkpoint are retained.

## Research and parent

Relevant branches inspected: LoopWAM_NT (baseline 3eac53a) and KV_concat (latest inspected 87040f4). KV concatenation/mixing changes remain separate experiments.

The specification and architecture, training, checkpoint, launch, evaluation and consolidated reports under plans/performance informed this implementation. The consolidated2026-10-08 report distinguishes Long-only v0 (96/100 for one evaluation seed,91.67% mean across evaluation seeds) from the separate full four-suite v0 checkpoint (388/400). The latter is our frozen parent. Dense-S12/S30, v1 multi-exit, v2 cross-step attention, asymmetric loops and repeated4/4 results do not establish a matched RT gate. No prior result is presented as an RT result.

Parent assets:
- ../FastWAM/runs/loopwam_nt/v0_full_libero_job4659_20261006/train/latest.pt
- Teacher SHA256: 4ffe1b14f35484ea582189ba0f06c5a41ac199e1fa733b491719727105e5e5cc
- Normalization SHA256: d5ad351eb3272f416d34082e8af5483ba1118ceca355a37d9f268fdb269fb3e4
- VAE SHA256: 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981
- Complete cached latents: ../FastWAM/runs/loopwam_nt/v0_full_libero_job4659_20261006/latents/
- Python: ../FastWAM/.venv/bin/python; PyTorch 2.7.1+cu128.

## Implementation and optimization

The residual_transport, transport_policy and transport_inference modules implement T0–T6, all eight schedules plus parent schedules, late video-loop alignment, always-cold four-loop steps, complete history truncation, cached text K/V, rank 16/alpha 16 core attention+FFN LoRA, zero-init step conditioning, and eager/CUDA graph inference.

The original frozen parent supplies native-grid local targets and same-noise endpoint trajectories. FU/TF regimes, scopes P-a/P-b/P-c, local/endpoint/command-gripper/original-FM anchor losses, first-ten-action endpoint loss, last-three-step endpoint backpropagation and paired initialization are implemented. Parent timesteps use native 0–1000 units; transport features use normalized tau.

Cached data reuses original normalizer, first-frame latents and text context. Production uses FP32 parameters/optimizer with BF16 compute, DDP, fused AdamW, pinned persistent loaders, four workers/rank and deterministic algorithms. Microbatch64/rank without block checkpointing is selected for P-b; gated P-c defaults to microbatch 8 with checkpointing. Larger P-c sizing remains an optimization task before its gated launch.

Atomic recovery every 100 updates, epoch boundary and signal saves weights, optimizer, EMA, scheduler, sampler cursor, all-rank RNG and exact asset/data/config contract. The wrapper retries three times and resumes latest.pt; batch jobs requeue on preemption. Locks prevent duplicate writers. --retry-failed resubmits only terminal failed batch jobs.

Portable exports contain a merged backbone and the original unmerged student. The RT adapter uses the latter: merging LoRA changed BF16 rounding by up to 0.02333 on a trained smoke. Preserving operation order made export/reload bit-exact. The ordinary parent loader refuses RT weights to prevent silent parameter loss.

RT+B2 validates a final epoch 10 B2a recipe, inherits action weights and step conditioning, installs fresh LoRA, and retains the original frozen teacher. Evaluation supports RT via --transport-schedule; parent zero-shot baselines use --transport-zero-shot-kind T0|T1|T2 and a schedule.

## Matrix

| Runs | Configuration | Gate |
|---|---|---|
| RT-A | T4, P-b, FU, S1/S2 | Eligible |
| RT-B4 | T0, otherwise matched RT-A | Eligible |
| RT-T1ft | T1 identity, P-b | Gate 0 + round 1 complete |
| RT-Pa | Transport only, no anchor | Gate 0 + round 1 complete |
| RT-TF | Teacher-forced | Round 2 complete with earlier gates |
| RT-A2 | Step-conditioned T4, S1/S2/S5/S6/S7/S8 | Gate 1 |
| RT-B2a | Step-conditioned full-action T0, parent2 | Gate 1 |
| RT+B2 | Final B2a initialization, T4, S8/S7 | Gate 1 + B2a complete |
| RT-B4-2 | Step-conditioned T0, matched RT-A2 schedules | Phase 2 stack round |
| RT-T3/T5/T6 | Alternative transport | Both phases read |
| RT-Pc | Full-action tuning | Both phases read |
| RT-NFE1 | Step-conditioned full-action T0, one step | Both phases read |
| RT-no-local/end/grip/anchor | Four separate loss ablations | Both phases read |
| RT-A-seed43/44 | Additional training seeds | Both phases read |

Twenty executable recipes in scripts/transport_experiments.py. Parent NFE 20/10/5/4/2/1 and zero-shot T1/T2 are eval-only. No gate results are invented.

## Verification

Evidence under plans/residual_transport_evidence:
- data_verification.json: complete provenance/count,16 decoded-window matches across four suites,1,000 dataset gripper signs.
- native_variants.json: cold-parent bit-exact parity, finite forward/backward for every recipe/schedule, frozen parent gradients.
- resume_verification.json: ten resumed updates match uninterrupted weights, Adam, EMA, loss and gradient norms bit for bit.
- graph_verification.json:26 cases,50 changing queries each, zero eager/graph error.
- export_verification.json: trained export/reload plus graph parity, exact B2a initialization, unchanged teacher. Synthetic test weights stay remote and are never published.
- verified_bench_A64.log: corrected deterministic global128 smoke,20 updates,73.07 GB/GPU peak. Updates18–20 took2.70/2.33/2.32 s. This is not a full-run duration prediction.
- project_pytest.log: supported project suite. Unscoped pytest also collected optional third-party RoboTwin scripts and failed on missing dependencies; these scripts are outside tests/.
- layout_verification.json:2 versus4 logical ranks on exactly2 physical H100s, Gloo gradient aggregation. This tests partition/scaling, not four-device NCCL.
- Residual reconstruction and FP32 LoRA merging use floating-point tolerances; exact BF16 inference retains adapters.

Pre-production fixes covered timestep units, deterministic resume, paired LoRA RNG, B2a identity, RT evaluation metadata, BF16 merge drift, canonical normalization bytes and failed-job recovery. Invalid early smokes never initialize production. New tests ignored by repository patterns are force-added to Git.

## Paths, submission and publication

Output root: runs/residual_transport/campaign_20261010. Each run has config.json, train.log, metrics.jsonl, latest.pt, data/, separate portable checkpoints and upload.log. A committed source snapshot has REVISION and VERIFIED.json.

Commands (run from repository after sourcing scripts/transport_env.sh):

    python scripts/submit_transport.py --source "$RT_SOURCE" --output-root "$RT_OUTPUT_ROOT"
    python scripts/submit_transport.py --source "$RT_SOURCE" --output-root "$RT_OUTPUT_ROOT" --submit
    python scripts/submit_transport.py --source "$RT_SOURCE" --output-root "$RT_OUTPUT_ROOT" --gates gates.json --submit
    python scripts/submit_transport.py --source "$RT_SOURCE" --output-root "$RT_OUTPUT_ROOT" --submit --retry-failed

Use gates.json only after genuine external evaluations per specification5.2–5.3. The Slurm script requests two H10080 GB,30 CPUs,256 GB RAM,120 h and preemption notification. Allocation 4843 is used for smokes and RT-A, preserving its shell step; unrelated 4825 and 4844 are untouched. RT-B4 is independent.

Public destination: https://huggingface.co/anhdao69/ResidualTrans . Uploaders run inside training wrappers, retry exponentially, publish each run's portable early/epoch checkpoints plus config/data metadata, verify size/LFS SHA256 and save atomic receipts. Recovery optimizer/RNG files and synthetic test weights remain remote. Final uploads are checked before wrapper success.

Job IDs, source revision and status are recorded in plans/residual_transport_evidence/launch_status.json and the output root's submissions.json. Pending/running means training is unfinished. Epoch8/9/10 uploads are not claimed before their receipts exist.
