# Fresh v0 training on all four LIBERO suites

## Requested run

Fresh canonical Wan initialization, v0 with four loops, global batch 128,
ten complete epochs on all available Spatial/Object/Goal/Long demonstrations.
Use idle four H100s in existing allocation 4659. Preserve other user workloads.
All available demonstrations are training data; simulator rollouts provide evaluation.
The local distribution contains 1,712 episodes and 277,713 frame anchors, not 2,000
demonstrations. Record per-task coverage and exact source hashes.

## Implementation and verification

1. Add a four-suite data factory preserving the existing Long 90/10 interface.
   Concatenate suites in fixed order with unique global cache indices and one
   action/state normalizer fitted over all training frames. Audit text caches.
2. Add explicit full-LIBERO trainer selection and bind dataset scope, suite list,
   epochs and normalization to saved checkpoint contracts. No resume checkpoint.
   Exact tail weighting must cover every real window exactly once each epoch.
3. Generalize simulator suite selection and final-budget validation. Preserve
   existing Long evaluation compatibility and reject incomplete checkpoints.
4. Benchmark v0 DDP/ZeRO1/ZeRO2 and microbatches on four H100s with FP32 policy
   and optimizer states, BF16 compute, fused optimizer and structured attention.
   Verify the selected setting in the native trainer on the complete dataset,
   cold and warm VAE caches, then simulator smoke on every suite.
5. Launch a fresh production run from a pinned source revision. Ten epochs imply
   2,170 updates/epoch, 21,700 total updates and 2,777,130 real windows.
6. Automatically evaluate the completed checkpoint on all four suites, ten
   initial states for every task (400 episodes), with videos and per-suite and
   overall success summaries. Retain existing 700 policy-step limit, 30 settling
   steps, ten diffusion steps and replan interval ten; record the protocol.
7. Record benchmark evidence, measured runtime estimate and Slurm step. Check
   first production updates, fresh optimizer, exact data budget and GPU activity.
   Keep allocation alive after completion; no new cancellation was requested.

## Gates

CPU regression tests cover cross-suite indexing, pooled normalization, complete
coverage, dynamic checkpoint budgets, suite validation and fail-closed pipeline.
GPU preflight checks finite losses/gradients, VAE anchor, memory fit and checkpoint
reload. Production starts only after all gates succeed. Source changes after
preflight invalidate preparation. Evaluation starts only after all ten epochs.
