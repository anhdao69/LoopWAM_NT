# Dense-S30 LIBERO-Long launch — 2026-10-06

## Architecture and initialization

The technical plan's native-layer Dense-S30 uses all 30 canonical compact Wan
donor layers in their original order. Each video/action block pair has independent
weights. It retains LoopWAM's video width 1536/FFN 6144, action width 512/FFN 2048,
12×128 attention, native Wan2.1 VAE, text features and action/proprio heads.
Training uses final-only flow loss without XSA or intermediate-exit supervision.

| Model | Stored block pairs | Effective block-pair applications | Policy parameters |
|---|---:|---:|---:|
| Dense-S12 | 12 | 12 | 584,536,135 |
| LoopWAM v0, K=4 | 12 | 30 | 584,536,135 |
| **Dense-S30** | **30** | **30** | **1,416,114,247** |

Dense-S30 is the matched inference-depth/width control for LoopWAM. Its larger
parameter count makes it a useful test of weight sharing. All 30 native donors
are loaded; it is initialized independently from the same canonical artifact,
with a fresh optimizer. Checkpoint metadata and strict reload validate the 30
layer configurations and donor identities.

## Data and training contract

The user confirmed the same Long split as the earlier controls: **344 training
and 44 validation demonstrations**, giving 92,678 training anchors and 11,602
validation anchors. A comparison with the previous v0 run confirmed equality
of train/validation episode IDs, window counts, normalization hash, all data
file hashes and text-cache hashes.

Ten epochs cover **7,250 updates and 926,780 real windows**, seed 42, global batch 128.
The final global batch of each epoch has 6 real windows, with exact weighting
and zero-loss dummy padding. Evaluation uses the final 10-epoch checkpoint.

## Speed selection on two H100 80GB GPUs

All successful trials used FP32 policy and optimizer states, BF16 compute, fused
AdamW and structured attention. Seven-update trials measured updates 3–7. Every
successful result's model/benchmark source hashes match the committed code.

| Backend | Microbatch/GPU | Block checkpointing | Warm seconds/update | Peak allocated GB |
|---|---:|---|---:|---:|
| DDP | 4 | Off | 5.166 | 55.11 |
| **DDP** | **8** | **Off** | **3.760** | **77.96** |
| ZeRO-1 | 8 | Off | 3.953 | 70.65 |
| ZeRO-2 | 8 | Off | 4.273 | 67.97 |
| ZeRO-1 / ZeRO-2 | 16 | Off | Out of memory | — |
| ZeRO-1 | 16 | On | 4.648 | 29.53 |
| ZeRO-1 | 32 | On | 4.298 | 35.83 |

Selected **DDP, microbatch 8/GPU, accumulation 8**, global 128, four data workers/rank.
Native ten-update cold/warm preflight confirmed 6.4568/3.8167 seconds per update.
The actual runner saved and reloaded checkpoints; mean save time was 19.00 seconds.
Cold native peak allocated/reserved memory was 77.71/81.95 GB.

## Verification and execution

- CPU regression suite: **258 passed, 7 skipped** (GPU tests unavailable in
  the CPU test process). The full pytest output is saved with the evidence.
- Native cold and warm smoke each completed 10 updates with finite losses/gradients,
  all intended parameter gradients, VAE anchor checks and checkpoint writes.
- Two-rank simulator smoke completed two 100-step episodes and saved both videos.
  These short smoke episodes check execution; final performance comes from the
  completed policy's 100-episode evaluation.
- Source pinned at `b12085164218211c6b404a7ce2dd92a9dd7d36bf`.
- Existing interactive allocation/step **4719.5**, worker-0, two H100s, 24 CPUs.
- Pipeline source: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/source_dense_s30_20261006`.
- Output: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/dense_s30_job4719_20261006`.
- Production: `dense_s30_train/`; final evaluation: `dense_s30_evaluation/`.

The pipeline verifies all ten epochs before automatic evaluation: ten tasks ×
ten initial states = **100 LIBERO-Long episodes with videos**. It retains the
existing 700-step limit, 30 settling steps, 10 diffusion steps and 10-action replan
interval. Source hashes are checked around training and evaluation; failures
stop the pipeline and record an error. Checkpoints and videos stay on the server.

## Finish estimate

### Production startup verification

The archived startup snapshot contains 39 consecutive updates with finite losses
and gradients. Mean combined loss changed from 1.7211 over the first five updates
to 0.5943 over the last five; this is an early execution check, not a convergence
result. At 15:00:53 UTC the live run had completed 43 updates (5,504 real windows),
and both H100s reported 100% utilization, using 79,469 and 79,715 MiB. The run
starts with a fresh optimizer and no resumed training checkpoint.

### Projection

Measured training estimate:

`725 × (6.4568 + 9 × 3.8167) + 10 × 19.00 = 29,775 seconds ≈ 8.27 hours`.

This accounts for one cold VAE-cache epoch, nine warm epochs and ten checkpoint
saves. Setup, held-out validation and final rollouts add time. Production began
at approximately **14:56 UTC on October 6**. Training is projected to finish near
**23:15 UTC**, with overall training/evaluation completion provisionally around
**00:00–01:30 UTC on October 7 (8–9:30 PM New York time on October 6)**.

The estimate comes from short native measurements; episode lengths and shared
storage load affect completion. The generic `timing.json` projection extrapolates
the current update mean and overestimates later warm epochs during epoch 1.

Evidence: [`../evidence/dense_s30_20261006`](../evidence/dense_s30_20261006).
