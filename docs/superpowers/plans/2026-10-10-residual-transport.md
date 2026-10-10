# Residual Transport Implementation Plan

> Execute inline using superpowers:executing-plans; verify with superpowers:test-driven-development.

Goal: Implement the complete specified RT architecture and training infrastructure; submit every currently eligible run.
Architecture: Preserve the v0 parent and share its frozen video prefill with an independent frozen action teacher. Add transport at the action pre/core boundary, cache action text K/V, and train with the specified rollout losses.
Tech stack: PyTorch DDP, BF16 autocast/FP32 trainable weights, existing full-LIBERO dataset, Slurm, Hugging Face.
Spec: plans/residual_transport.md plus the user's attachment and gate clarification.

## Global constraints
All implementation/testing/training stays in /mnt/data/vmo-ai-task/anhdh35/LoopWAM_NT on residual_trans.
Two H100s per training job, effective global batch 128, all 277713 windows, 10 epochs, retain epoch 8/9/10 separately.
No full simulator evaluation on H100. Preserve jobs 4825 and 4843; use idle 4843 for smoke and a feasible eligible run.
All published weights go to public anhdao69/ResidualTrans, independently named, retryable uploads.
User explicitly retains evaluation gates: only RT-A and RT-B4 currently eligible. Implement/configure gated variants without launching prematurely.

## Review focus
- Cold full-depth equivalence, cached attention parity and late video-loop alignment.
- Teacher target detachment and complete transport history truncation.
- Variable-NFE grid targets, phase-2 step conditioning, command-space gripper conversion.
- Exact tail weighting, per-rank RNG/cursor recovery, graph/eager and layout parity.
- Independent output paths, immutable job code, upload retries and job deduplication.

## Tasks
- [x] 1. Research all relevant branches/reports/configs/checkpoints and verify assets/environment.
- [x] 2. Add tested T0–T6 transport, core LoRA, step-length embedding, cached text attention, action rollout/inference.
- [x] 3. Add tested full-data FU/TF trainer, teacher targets/losses, all parameter scopes, exact resume/EMA, epoch checkpoints and measurement.
- [x] 4. Add full experiment matrix, gated submission, retryable public uploads and durable independent Slurm jobs.
- [x] 5. Run CPU/native 2-H100 smokes, optimization measurements, resume/graph/parity checks; fix failures.
- [ ] 6. Review, commit/push residual_trans, submit eligible runs and verify queue/training; document evidence and blocked gates.
