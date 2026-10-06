# Full LIBERO v0 launch — 2026-10-06

## Run contract

- Allocation/step: **4659.30**, worker-3, four H100 80 GB GPUs.
- Pinned code: `3627760ae1e56972692a86c7d8a1e5caee6f24f8`.
- Output root: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_full_libero_job4659_20261006`.
- v0: 584,536,135 trainable policy parameters, four loops, final-exit supervision.
- Fresh canonical Wan initialization and fresh AdamW; seed 42.
- Ten complete epochs, **global batch 128 = 4 GPUs × microbatch 8 × accumulation 4**.
- DDP, fused AdamW, structured attention, FP32 policy/optimizer states, BF16 compute.
- Frozen VAE clip cache; all 277,713 global indices have distinct cache entries.
- One global action/state normalizer fitted on the complete training mixture.

## Dataset coverage

| Suite | Demonstrations | Frame anchors per epoch |
|---|---:|---:|
| Spatial | 434 | 53,229 |
| Object | 457 | 67,309 |
| Goal | 433 | 52,895 |
| Long | 388 | 104,280 |
| **Total** | **1,712** | **277,713** |

Every available demonstration is used for training. The local distribution has
fewer than 50 demonstrations per task; its exact coverage and content hashes are
recorded in the data manifest. All 40 task descriptions matched the existing text
embeddings. Beginning/middle/end video samples decoded in every suite. Gripper
labels were checked against all state/action rows in each suite.

One epoch requires 2,170 updates. Training covers **21,700 updates and 2,777,130
real windows**. The final global batch of each epoch has 81 real windows; dummy
padding contributes zero loss and the gradient is normalized by the real count.

## Speed selection

Seven-update trials measured updates 3–7; all successful trials preserved FP32
policy and Adam moments. The architectural comparison used the previous Long
data at the same tensor shapes, followed by native full-dataset verification.

| Backend, microbatch per GPU | Warm seconds/update | Peak allocated GB |
|---|---:|---:|
| DDP, 8 | 1.882; repeat 1.881 | 59.48 |
| ZeRO-1, 8 | 1.915; repeat 1.920 | 54.57 |
| ZeRO-2, 8 | 1.980 | 52.58 |
| DDP / ZeRO-1 / ZeRO-2, 16 | All ran out of memory | — |

Selected **DDP microbatch 8**. Native full-dataset ten-update trials measured
**3.2203 s/update cold** and **1.8877 s/update warm**. The frozen VAE cache makes
epochs 2–10 faster after all windows have been encoded once.

Projection: `2170 × (3.2203 + 9 × 1.8877) / 3600 = 12.18 hours` of training updates.
Allow roughly **12–13 hours for training** including saves and setup. This is a
short-run projection; shared storage/compute contention can change throughput.
Final simulator evaluation will add time, provisionally **1–3 hours**, depending
on episode lengths and success rates.

The generic projection in `train/timing.json` extrapolates the current update
mean, including cold-cache work. During epoch 1 it overestimates the later warm
epochs; the 12.18-hour estimate above explicitly accounts for cache warming.

## Verification and automatic evaluation

- Server regression suite: 214 passed, 7 skipped (CUDA-specific tests unavailable
  in the CPU test process). After the source-freeze review fix, all 12 pipeline
  tests passed on the pinned server checkout; focused local evaluator/pipeline
  tests passed 44/44.
- Native cold and warm training: ten updates each, finite gradients/losses,
  VAE anchor checks, checkpoint writes and checkpoint reload during inference.
- Simulator smoke: four GPU workers in each suite, **16 episodes total**, with
  videos. Smoke runs use ten-update checkpoints and only 30 policy steps.
- Production initialization is independent of smoke model/optimizer weights.
- Source hashes bind trainer, model, data, evaluator and simulator preprocessing.
  They are checked after training and before/after every final evaluation suite.
- Final evaluation runs only after the full ten-epoch checkpoint passes update,
  data exposure, normalization, suite and architecture checks.
- Four suites × ten tasks × ten initial states = **400 final episodes**, with
  per-task/per-suite success rates and videos. The retained protocol uses 700
  policy steps, 30 settling steps, 10 diffusion steps and replanning every 10
  actions for every suite. This common cap differs from suite-specific caps used
  in some other evaluation protocols and should accompany reported results.
- Each suite writes `evaluation_<suite>/summary.json`; the aggregate writes
  `summary.json`. Training status is `train/timing.json`; orchestration status is
  `pipeline_status.json`. Videos/checkpoints remain on the server.

The pipeline entered fresh production training at approximately **14:18 UTC on
October 6**. This document is a launch report; final success rates will be
available after training and all simulator rollouts finish.

At **14:20:20 UTC**, 35 production updates (4,480 real windows) were verified,
with finite losses and gradients. All four GPUs reported **100% utilization**
and approximately 61,900 MiB used each. The average combined video/action loss
over updates 1–5 was 2.088; over updates 31–35 it was 1.107. This initial decrease
is a startup check; final task performance requires the simulator evaluation.

Evidence: [`../evidence/full_libero_v0_20261006`](../evidence/full_libero_v0_20261006).
