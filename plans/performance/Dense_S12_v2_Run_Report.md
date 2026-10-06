# Dense-S12 → v2: four-H100 run report

## Architecture and training contract

Dense-S12 follows the technical report: 3 pre + 6 middle + 3 post donor pairs, one traversal, final-only flow-matching supervision, no XSA. It has **584,536,135 trainable policy parameters**. v2 retains four shared-core traversals, all-exit supervision and core-only XSA.

Both runs start fresh from canonical Wan initialization with empty optimizers; neither resumes smoke or previous training checkpoints. Each uses LIBERO-Long, global batch 128, ten epochs, 92,678 training windows/epoch, 725 updates/epoch and 7,250 total updates. FP32 policy/master weights and AdamW moments, BF16 compute, loss weighting, seed 42 and the evaluation protocol are preserved.

## Fastest verified configurations

| Model | Backend | Per-GPU batch | Accumulation | Workers/rank | Warm benchmark | Native warm smoke |
|---|---|---:|---:|---:|---:|---:|
| Dense-S12 | ZeRO-1 | 32 | 1 | 8 | 0.893s/update | 1.112s/update |
| v2 | DDP | 8 | 4 | 4 | 2.561s/update | 2.581s/update |

Tested DDP, ZeRO-1 and ZeRO-2, larger microbatches, and worker counts. Dense DDP/microbatch 32 and v2/microbatch 16 exceeded memory and were rejected. Selected configurations passed actual cold-cache native-VAE training, checkpoint saves/reloads and simulator inference. Fused AdamW and structured attention are enabled. See [benchmark and smoke evidence](../evidence/dense_v2/job4659).

## Runtime estimate

- **Dense training: about 2.45 h**, using a first cold-cache epoch at 2.160 s/update and nine warm epochs at 1.112 s/update.
- **v2 training: about 5.20 h**, using 2.581 s/update and the private preprocessing cache populated by dense.
- These estimates exclude checkpointing, epoch validation, startup and simulator rollouts. They are projections from short measured runs, not completed training times.
- Each final evaluation runs 100 episodes (10/task) with videos, four GPU workers, up to 700 policy steps, 30 settling steps, 10 denoising steps and replanning every 10 steps. Smoke episodes of 100 policy steps took 17–27s. A simple sevenfold extrapolation on the busiest worker's 30 episodes gives roughly 1.5–1.6 h/evaluation; early successes and task variation can change that substantially.
- **Budget approximately 11 h plus checkpoint/validation overhead for the entire sequence.** Full-run measurements will supersede these estimates.

## Verification

- Project CPU suite: 149 passed, 7 CUDA skips; CUDA checks separately: 7 passed.
- Dense forward and gradient equality with v0 at K=1; strict one-pass enforcement and checkpoint reconstruction.
- Four-rank production DDP/ZeRO-1/ZeRO-2 comparisons with exact global and partial-tail groups: maximum parameter difference 1.49e-8; optimizer moments and native/portable checkpoint recovery checked.
- Dense and v2 each passed 10 cold and 10 warm native training updates with finite losses/gradients, then four 100-step simulator episodes with saved videos. Smoke models had no successful episodes; these tests establish execution correctness, not learned task performance.
- Preflight completed in Slurm step 4659.16. Input source hashes are locked for production. Independent reviews covered architecture, trainer/backends, evaluator and pipeline gates.

## Execution

The production sequence is **fresh Dense-S12 training → 100-rollout dense evaluation → fresh v2 training → 100-rollout v2 evaluation**. Each next stage requires a successful previous stage and validated complete outputs. Source drift, pre-existing production outputs, missing training steps or incomplete evaluations stop the pipeline.

Allocation: 4659 on worker-3, four H100 80GB GPUs. Existing v0/v1 jobs remain unchanged.

Remote source: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/source_dense_v2_work`.

Remote outputs: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/dense_v2_job4659_20261006`.

Progress: `pipeline_status.json`, `production.log`, `dense_s12_train/metrics.jsonl`, `dense_s12_train/timing.json`, then equivalent `v2_train` files. Final evaluations live in `dense_s12_evaluation` and `v2_evaluation`.

## Production startup verified

Started **2026-10-06 03:27:33 UTC** (October 5, 23:27:33 America/New_York), Slurm step **4659.17**, step name `test_training_dense_v2`, pinned source commit `012cc8ff305ee067874009e0be705837301f9d4d`.

Verified the first 15 actual production updates: empty initial optimizer, `resume: null`, global 128 / world 4 / microbatch 32 / accumulation 1, one dense traversal, 7,250 planned updates. All observed losses and gradient norms are finite. Updates 2–15 averaged **2.114s/update** while filling the latent cache. Video loss moved from 0.819 to 0.386 and action loss from 1.350 to 1.051; this is an encouraging early change, not a convergence claim.

The snapshot confirms v0 allocation 4689 and v1 job 4691 remain running. See [production startup evidence](../evidence/dense_v2/job4659/production_startup_verified.json). The full training and final evaluations are still pending; the detached pipeline proceeds automatically after each successful stage.
