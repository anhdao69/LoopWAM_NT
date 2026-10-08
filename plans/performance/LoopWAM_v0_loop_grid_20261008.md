# LoopWAM v0 video/action loop grid — 2026-10-08

## Experiment contract

Three independent fresh LIBERO-Long runs: video/action core repetitions **4/2, 2/2, 1/4**. Each uses two H100 80 GB GPUs; allocation 4728 on worker-1 runs 4/2, and allocation 4659 on worker-3 runs 2/2 and 1/4 on disjoint GPU pairs. Production source is pinned to `8a29ffd`.

All models retain 584,536,135 policy parameters and the 3-prelude / 6-shared-core / 3-coda structure per expert. Effective video/action block applications are respectively 30/18, 18/18, and 12/30. Original v0 uses 4/4 repetitions, or 30/30 block applications.

Action-to-video core alignment is `max(1, video_loops - action_loops + action_iteration)`, with action iterations numbered from one. Thus 4/2 reads video loops 3 and 4; 2/2 reads loops 1 and 2; 1/4 reads video loop 1 four times. The last variant computes the video stream once, retains differentiable observation-only KV tensors, and accumulates action gradients through every reuse. Future video frames cannot condition actions. Prelude and coda use their corresponding video blocks. This is an explicit architecture ablation; neither deeper action nor fewer video loops is assumed to improve success rate.

Matched controls:

- Same 344-training / 44-validation demonstration split, 92,678 training windows per epoch.
- Ten epochs, 7,250 optimizer updates, 926,780 real windows, global batch 128, seed 42.
- Fresh canonical Wan donor initialization and fresh optimizer, never a trained checkpoint.
- FP32 parameters and optimizer moments; BF16 autocast; AdamW lr 1e-4, betas (0.9, 0.95), eps 1e-8, weight decay 0.01; 5% warmup and cosine schedule; clip norm 1.
- Native frozen VAE latent cache reused only for speed; data, normalization, asset hashes and sample contracts checked against the original v0 manifest.
- DDP / ZeRO-1 / ZeRO-2 and per-GPU microbatches measured independently; global batch and optimization schedule remain fixed.

## Verification and launch gates

Remote tests: **150 passed, 2 skipped** for model, gradient, mask, checkpoint, rollout and prior architecture regressions; **6 passed** for new pipeline depth and command gates.

For every pair, tests compare joint training with observation-only cached inference, including parameter gradients, check future-video isolation, and verify saved checkpoint depth restoration. Native GPU gates additionally require benchmark finite gradients, two fresh ten-update training runs (cold and warm VAE cache), and two simulator smoke episodes with saved videos. Only after these pass does production initialize from scratch. Training completeness and final checkpoint identity are checked before the automatic final evaluation.

## Automatic evaluation

Each final checkpoint receives **100 LIBERO-Long rollouts**: ten episodes for each of ten tasks, seed 42 with the original task/episode/replan offsets; 700-step policy horizon, 30 settling steps, 32-action chunk, replan every ten actions, ten denoising steps and guidance scale 1. Videos remain on the server. The pipeline rejects missing/duplicate episodes or missing videos and records the final checkpoint SHA-256.

## Server outputs

Root: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/`.

- `loop_grid_job4728_20261008/v4a2/`
- `loop_grid_job4659_20261008/v2a2/`
- `loop_grid_job4659_20261008/v1a4/`

Each directory records benchmark results, resource mapping, selected configuration, native smoke output, source hashes, production training, and automatic evaluation. Final runtime/status measurements will be appended after launch checks.

## Prior completed 4/1 control

The preceding matched 4/1 run completed 7,250 updates in 24,663.51 seconds (6 h 51 m), followed by 100 rollouts in 3,314.38 seconds (55 m). Success rate: **81/100 (81%)**, compared with **96/100** for the original matched Long 4/4 v0 run. These are single-training-seed observations and do not establish statistical superiority between architectures.

## Verified production launch — 03:49:41 UTC, October 8

All three native cold/warm ten-update training checks and all six simulator smoke episodes passed. All three full training processes are running with fresh initialization and finite logged loss/gradient values. No production resume was used. Production source files are unchanged from the verified preflight.

| Video/action loops | Allocation / step | GPU indices | Backend | Microbatch / accumulation | Completed updates at audit | Mean production update time |
|---|---|---|---|---|---|---|
| 4/2 | 4728.17, worker-1 | 0,1 | DDP | 8 / 8 | 23 / 7,250 | 3.459 s |
| 2/2 | 4659.48, worker-3 | 0,1 | DDP | 16 / 4 | 24 / 7,250 | 2.212 s |
| 1/4 | 4659.48, worker-3 | 2,3 | DDP | 16 / 4 | 24 / 7,250 | 2.207 s |

Each experiment has 24 allocated CPU threads, eight loader workers per rank and its own GPU visibility mask. Worker-3's two experiments have disjoint CPU affinity sets as well as disjoint GPUs. The original interactive allocations remain alive after the experiment steps; this launch does not cancel either allocation.

### Measured speed comparison

Times below are seconds per global-batch-128 optimizer update. Initial trials use eight updates with warmup excluded; the selected DDP setting was confirmed for twenty updates. All comparisons keep FP32 parameter/optimizer state and BF16 compute.

| Pair | DDP mb8 | DDP mb16 | ZeRO-1 at selected batch | ZeRO-2 at selected batch | DDP confirmation |
|---|---|---|---|---|---|
| 4/2 | 3.466 | OOM | 3.539 (mb8) | 3.641 (mb8) | 3.449 |
| 2/2 | 2.454 | 2.036 | 2.145 (mb16) | 2.177 (mb16) | 2.061 |
| 1/4 | 2.489 | 1.921 | 2.046 (mb16) | 2.187 (mb16) | 2.091 |

Microbatch 32 exceeded GPU memory for both smaller-video variants. These isolated benchmark failures were recorded and rejected; they did not affect production checkpoints. The short timing measurements have ordinary run-to-run variation. DDP won each initial comparison, and the chosen configuration passed the longer benchmark and native training/rollout checks. This search does not establish a universal optimum over every possible runtime setting.

### Estimated completion

- **4/2:** approximately **7 hours training**, then **45–90 minutes evaluation**; roughly **07:30–08:30 EDT October 8** for the complete pipeline.
- **2/2:** approximately **4.2–4.6 hours training**, then **45–90 minutes evaluation**; roughly **04:45–06:00 EDT October 8**.
- **1/4:** approximately **4.2–4.6 hours training**, then **45–90 minutes evaluation**; roughly **04:45–06:00 EDT October 8**.

These are projections from short native timing measurements and the prior completed rollout duration, not completed runtimes. Simulator duration depends on episode lengths and the new action budget. Early production update times include normal timing variation; evaluate these estimates again after longer training. There is no final success rate yet for any of the three new variants.

Machine-readable launch evidence, source/asset hashes, benchmark measurements, resource assignments, smoke evaluation records, and finite production updates are in [LoopWAM_v0_loop_grid_20261008_evidence.json](LoopWAM_v0_loop_grid_20261008_evidence.json).
