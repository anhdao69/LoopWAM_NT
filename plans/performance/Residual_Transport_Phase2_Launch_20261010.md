# Residual Transport: next two runs, mixed GPU counts

The user's latest request is two more training runs, one on two GPUs and one on the newly available single-GPU interactive allocation. This overrides the earlier uniform two-GPU requirement for RT-A2. The selected next two independent runs are RT-A2 and RT-B2a; their phase-2 evaluation wait is overridden for submission, without asserting Gate 1 has passed. RT+B2 still requires the final B2a checkpoint and is not submitted.

## Configuration

| Run | GPUs | Microbatch/GPU | Accumulation | Global batch | Scope |
|---|---:|---:|---:|---:|---|
| RT-A2 | 1 | 64 | 2 | 128 | T4, P-b, step conditioning, six schedules |
| RT-B2a | 2 | 64 | 1 | 128 | T0, full action expert, step conditioning, parent2 |

Both runs use the original frozen full-LIBERO parent, all 277,713 windows, 10 epochs / 21,700 updates, FP32 weights and BF16 compute. Epoch 8/9/10 and early update-2,000 exports, checkpoint recovery and retrying public uploads are unchanged. Neither starts from smoke weights.

Allocation 4857 is an idle, unlimited-time single-H100 job on worker-1 with 16 allocated CPUs and 128GB RAM. RT-A2 uses a detached Slurm step inside it. Keep allocation 4857 alive for this run. Its wrapper retries failures and saves resumable state; terminating the allocation requires restarting the run with world size 1 and the same microbatch. This run has no extra queued backup or Slurm dependency. RT-B2a is a separate two-H100 batch job with no dependencies. Existing five RT jobs and other workloads are left intact.

## Changes

- Trainer accepts an explicit --expected-gpus 1 or 2, while retaining the global128/ten-epoch production requirements.
- Job wrapper uses RT_WORLD_SIZE (default2) to configure torchrun and the trainer's explicit GPU contract.
- Evaluation validation accepts either GPU count and checks world*microbatch divides128.
- Submission supports the explicit --independent-phase2-pair override, per-job GPU sizing, microbatch/checkpoint controls and single-GPU interactive registration.
- Existing jobs remain pinned to their original snapshot. New runs receive a new committed snapshot.

## Validation and sizing

- Thirty focused tests passed, including single-GPU retained-checkpoint evaluation and exact one-/two-GPU global-batch sample ordering across the entire dataset.
- A single-GPU RT-A2 smoke processed20 global128 updates across all six schedules, with finite gradients. Peak allocation83.83GB; no OOM. Microbatch32 used45.45GB on S1 but needs four accumulation passes. Microbatch64 was selected for throughput; the allocation must remain exclusive to this training process.
- RT-B2a microbatch64 native forward/backward/optimizer checks used44.23GB per GPU before DDP overhead. Every trainable parameter received a gradient. This supports using64 without block checkpointing instead of the conservative earlier8+checkpoint default.
- Single-GPU interrupted/uninterrupted recovery reproduced ten subsequent updates bit for bit, including weights, Adam state, EMA, losses and gradient norms; see single_resume_verification.json.

Launch IDs and current state are recorded in phase2_submission_status.json. This report describes training setup, not evaluation results.
