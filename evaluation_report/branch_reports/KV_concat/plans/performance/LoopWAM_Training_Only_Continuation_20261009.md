# Training-only continuation — October 9, 2026

At the user's request, all remaining simulator evaluation, inference latency and
cross-queue evaluation barriers are removed. Completed Long training/evaluation
and uploaded weights are retained. Training/model code and dataset contracts are
unchanged.

## Submitted Slurm jobs

| New job | Dependency | Training stages in new allocation | Resources |
|---|---|---|---|
| **4795** | `afterany:4770` | Full-suite aligned 3/3 → Dense-S12 | 2 H100, 32 CPU, 256 GiB, 120 h |
| **4796** | `afterany:4771` | Full-suite aligned 2/2 → Dense-S30 | 2 H100, 32 CPU, 256 GiB, 120 h |

The currently running **4770 / full 4/1** and **4771 / full 1/4** continue their
live training to completion. Resubmitting from their last periodic checkpoints
would repeat unsaved updates, so those active training processes are preserved.
Only the old campaign controller is paused, preventing it from launching its
next evaluation. Its torchrun child continues normally.

A CPU-only monitor in each old allocation waits for full training completion,
then validates the final 21,700-update checkpoint, all 2,777,130 real training
windows, optimizer state, both rank RNG states, finite metrics, source hashes
and the original fairness contract. It records the checkpoint hash and cancels
**only its own old allocation**. The dependent new job requires that successful
handoff before starting. A monitor failure records the error and leaves live
training intact; the stopped controller still cannot start evaluation. Such a
failure requires operator recovery and never authorizes an incomplete handoff.

No remaining inference or simulator evaluation is scheduled, including after the
Dense models. Each new allocation exits after its two training stages finish.
Previously produced evaluation records are unchanged.

## Unchanged training contract

All six full-suite models use all four local LIBERO suites, global batch 128,
seed 42, ten epochs and the same frozen latent cache. Each new model starts from
the canonical donor with a fresh optimizer. The already running 4/1 and 1/4 keep
their current weights and optimizer throughout; no rollback or restart occurred.

| Model | DDP microbatch × accumulation × GPUs |
|---|---|
| 4/1 (currently running) | 8 × 8 × 2 |
| 1/4 (currently running) | 16 × 4 × 2 |
| 3/3 | 8 × 8 × 2 |
| 2/2 | 16 × 4 × 2 |
| Dense-S12 | 16 × 4 × 2 |
| Dense-S30 | 8 × 8 × 2 |

Training source remains immutable commit
`3a89bfac5cfccad66bfebbe00014f7f739a78adc`. The external continuation runner is
SHA-256 bound to `a4316882fa7675706b4f5afe6b9bc80e2bc3a72ea89ee35d7183c4d066f3b339`.
It imports the original training command builder and completion gates; it does
not edit the running trainer or model. Remaining configurations must explicitly
be DDP, matching the previously measured fastest valid configurations.

## Verification and status

- Five focused unit tests pass locally and on the server: stage ordering,
  controller ownership/job/command identity, final checkpoint completeness,
  handoff hash binding, and rejection of unsupported backend configurations.
- A separate subprocess check confirmed that stopping only the controller PID
  leaves its training-like child executing.
- Independent review found two issues; both were fixed before submission:
  preserve live training on monitor failure, and explicitly restrict this
  continuation to the verified DDP configurations.
- Both remote plans passed immutable source checks and contain only training
  commands with global batch 128 and validation samples zero.
- Both guards are armed; Slurm lists 4795/4796 pending on their dependencies.
- After arming, current training advanced to **8,529 / 21,700 (4/1)** and
  **16,189 / 21,700 (1/4)**. No guard failure was recorded.

## Runtime estimate

At approximately **2026-10-09 21:08 UTC (5:08 p.m. EDT)**:

| Chain | Remaining current training | Subsequent training estimate | Total remaining training |
|---|---:|---:|---:|
| 4/1 → 3/3 → Dense-S12 | 12.64 h | 27.86 h | **40.50 h** |
| 1/4 → 2/2 → Dense-S30 | 2.78 h | 35.67 h | **38.45 h** |

Current-stage estimates use measured update times. Subsequent stages use the
verified native preflight measurements. Allow approximately **40–44 hours** for
the longer chain with setup/checkpoint overhead, placing completion around
**October 11, 9 a.m.–1 p.m. EDT**, plus any Slurm scheduling wait or contention.
This is a projection, not completed runtime evidence.

## Files

- [Continuation runner](../../scripts/run_training_only_continuation.py)
- [Slurm template](../../scripts/submit_training_only_continuation.sbatch)
- [Operational evidence](../evidence/kv_concat_20261008/training_only_20261009/)

Server operations:
`/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/training_only_handoff_20261009/`.
New training outputs:
`/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/training_only_job4795/` and
`/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/training_only_job4796/`.
