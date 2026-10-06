# LoopWAM v1: fresh training and final LIBERO inference in one Slurm job

## Requested contract

Fresh v1 from the same canonical Wan donor artifact as v0, global batch128, ten LIBERO-Long epochs:7250updates and926780real windows. Preserve v0 allocation4689 and SimpleMemVLN4659. Separate batch allocation requests two H100s,16CPUs,128GBhostRAM and24hours. Source is pinned in an isolated checkout with explicit source PYTHONPATH.

After training, run100LIBERO-Long simulator episodes (10/task), split across both GPUs, and save videos, individual episode records, per-task success rates and a combined summary. User explicitly confirmed this evaluation scope.

## Pipeline

1. Measure actual v1 warm-cache DDP throughput at microbatch4,8,16 using five global128updates each, excluding the first two from steady timing. Only CUDA OOM is an allowed unsuccessful candidate; other errors stop the job.
2. Rank valid measured candidates, then verify cold-cache training memory fit with three real v1 updates, all intended gradients, exact VAE anchoring, and a saved checkpoint. OOM-only fallback tries the next candidate. These temporary weights are never used to initialize production.
3. Reconstruct that smoke checkpoint and run two simulator episodes, one/GPU, with20policy steps each. A model, normalization, simulator or video failure prevents full training.
4. Start the full v1 run in a new directory from canonical Wan weights, fresh optimizer/scheduler/data cursor. Use fused FP32 AdamW, BF16 compute, DDP gradient bucket views, equivalent structured attention and a private lazy BF16 latent cache. Benchmark, smoke and production cache writers never share the running v0 cache.
5. Verify complete training status, final7250update checkpoint,926780real windows, v1/global128/10epochs/fresh-start manifest before starting final inference. Any full-training failure stops the job; partial checkpoints cannot masquerade as final evaluation.
6. Evaluate the final checkpoint on100rollouts. Store checkpoint/VAE/statistics hashes and runtime versions, enforce distinct initial states, check exact task/episode coverage, and record terminal outcomes. No training benchmark result is called a trained policy success rate.

The batch-size search is DDP-specific. The preceding same-hardware v0 comparisons found DDP faster than actual ZeRO1/2; no unmeasured claim that every possible backend/compilation strategy has been exhausted is made. This job chooses the fastest candidate that passes both measured v1 throughput and cold-memory validation.

## Evaluation protocol

Match the supplied plan§17.1:700policy steps/task episode cap for libero_10,30initial waiting steps, action chunk32, execute10before replanning, ten denoising steps, CFG1, no action ensemble, four recurrent loops and binary gripper control. Cameras use the same order, rotation,224px antialiased resize and horizontal concatenation as training. Proprioception uses current position/axis-angle/gripper state and the exact training-only normalizer; cached instruction padding masks are retained.

Dataset gripper0means closed/1means open; simulator+1means closed/-1means open. The adapter converts and thresholds accordingly. Actual parquet audit:104280rows, g0mean absolute finger positions[.01754,.01827], g1mean[.03752,.03756], consistent with this convention. Full-image inputs and action normalization have regression coverage.

CPU preflight has constructed the simulator, loaded trusted LIBERO initial states explicitly with weights_only=False for PyTorch2.7 compatibility, stepped/rendered both cameras and written anMP4. All ten task language strings resolve to cached128x4096T5contexts with valid padding masks. Checkpoint completeness, normalization hashes, gripper conversion, task sharding, terminal handling and replanning are tested.

## Job and evidence

Submission/job ID, selected microbatch and startup evidence will be appended after Slurm execution. Runtime paths are under `runs/loopwam_nt/v1_bs128_jobJOBID/`:

- `job_status.json`: current pipeline stage or failure.
- `selection.json`: benchmark results, successful candidates and OOM exclusions.
- `train/`: production manifest, losses, timing, checkpoints and private latent cache.
- `inference/summary.json`:100episode total/per-task success rates.
- `inference/videos/`: recorded simulator videos.

The job is autonomous after submission. Actual runtime depends on the v1 benchmarks and scheduler start; ten-epoch completion and robot success are not claimed in advance.

Pre-submission full CPU acceptance:98passed,7CUDA-only tests skipped,22.06seconds. The new rollout suite contributes9passing tests; GPU memory/training/inference are gated inside the batch job before production.

## Submitted and verified

Slurm **4691**, renamed at the user's request to **test_training**, is running on worker-1 with two H100s. v0step4689.61 and SimpleMemVLN4659 are separate and were not stopped. Source is pinned at ccf8a67dc5c479d06a7e0acd64da9da100603e15 in runs/loopwam_nt/source_v1_ccf8a67; explicit PYTHONPATH was verified to import that checkout. The submission template's default job name is now test_training.

Measured cached v1 throughput:microbatch4=6.333s/update, microbatch8=4.752s/update; microbatch16 failed CUDA OOM and was excluded. Microbatch8 also passed cold-cache training, exact native VAE anchor checks and all intended gradient checks. Selected accumulation8 gives global128. Three-update smoke steady cold time=7.427s/update. Two independent GPU simulator smoke episodes completed and saved videos; these20-step/three-update checks are plumbing tests, not trained success-rate results.

Full fresh production started successfully. At 2026-10-06T01:11:14.880550+00:00, update13 completed with finite video/action losses and gradient norms. Manifest says v1, resume=null, canonical Wan artifact with fresh optimizer, empty optimizer state before update1, epochs10, planned7250updates. Production output: runs/loopwam_nt/v1_bs128_job4691/train. Final100episode evaluation will use that directory's completed latest.pt and save under runs/loopwam_nt/v1_bs128_job4691/inference.

Training projection **10.11hours** uses one cold epoch plus nine cached epochs; checkpoint/validation overhead and the subsequent100rollouts are additional. Final inference has not run yet because training is ongoing. Monitor Slurm job4691 and job_status.json stage; OOM exclusions are expected candidate failures, not production failures.
