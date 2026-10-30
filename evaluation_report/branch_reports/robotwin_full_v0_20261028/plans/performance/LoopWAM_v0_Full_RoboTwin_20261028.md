# Original LoopWAM v0: full RoboTwin 2.0, five epochs

## Final user-approved recipe

Original LoopWAM v0, 3 prelude / 6 shared core / 3 coda, four video and four action loops, aligned attention. Each expert applies 30 blocks. **No ChronoLoop model, memory module, recurrent state, or history frames.** `RobotLoopWAM` extends the original `LoopWAM` only to accept validated cached robot latents and save a strictly identified14D robot checkpoint; its MoT is the original `LoopMoT`, and forward/loss/inference come from original v0. The no-memory DDP wrapper converts original per-window mean loss into a sum before global averaging.

Fresh canonical Wan2.1 donor initialization, the same donor mapping as original LIBERO v0: `[0,1,2,9,10,11,12,13,14,27,28,29]`. This is **not** the centered-donor ablation and **not** a trained LIBERO or ChronoLoop checkpoint. CPU-seeded 42 default-linear initialization for the new 14-D action encoder,14D action head and14D proprioception encoder. All 673 other expert tensors match donor-bank tensors exactly. Policy count 584,567,886; no memory parameters. Artifact SHA256 `a3bf2097bafddac278148c1cb0321163f68cfe2b954e206436da0e3f0bd3c535`.

Full FastWAM RoboTwin release:50 tasks×(50 clean+500 randomized)=27,500 demonstrations,6,075,103 frame-anchor windows. Existing cache reused:all 6,075,103 video latents present;921,032 instruction embeddings from prior complete preprocessing. Three cameras packed 384×320,9 video frames at 0,4,…32,32 absolute 14-D joint actions, per-anchor instruction, published global z-score normalization. Source stats file SHA256 `7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095`; canonical parsed-stats JSON SHA256 `38ed9c8d466356fb8f4688da470b14a036dbc51e0fc5484cc5959b507adbcd48`. These hashes differ because one hashes file bytes and the other canonical JSON; both were checked. No recomputed normalization or held-out subset.

**Exact real batch 128**, per user's clarification. Every window appears once per epoch, shuffled with `torch.randperm(seed42+zero_based_epoch)`. Eight GPUs receive 16 real windows each per full update. The final batch of each epoch contains 95 real windows (12 on seven ranks,11 on the eighth).47,462 updates/epoch;**237,310 total updates**,**30,375,515 real windows**. The earlier CL-0 stream-slot recipe is superseded: no stream padding or overlapping epochs is used. Dataset/cache helper names retain their existing prefix; this does not instantiate a ChronoLoop model.

AdamW, LR 1e-4 for backbone and new interfaces, betas (0.9, 0.95),epsilon 1e-8,decay 0.01,clip 1.0;11,865 warmup updates, cosine to 1e-6 at the final update. FP32 trainable weights and Adam moments, BF16 computation. Accumulated gradients normalize by actual global window count, including the 95-window tail. Epoch 3/4/5 weight snapshots at 142,386/189,848/237,310; exact recovery checkpoint every 1,000 updates and epoch ends. Only epoch 5 is the final reported model.

## Resource transition and data reuse

All six preceding LIBERO models completed and HF uploads were hash-verified. Only then were user-owned allocations 5248 and 4925 canceled, allowing the user's pending eight-GPU job 5489 to run on worker-2. Its original interactive shell remains active. No other user's allocation was changed. Parent allocation 5489 is unlimited but must remain active for training to continue.

Server source `/mnt/data/vmo-ai-task/anhdh35/LoopWAM_robotwin_full`; run root `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin_full_v0_20261028`; source branch `robotwin_full_v0_20261028`, based on277c635 to reuse audited full-data interfaces. Runtime code pinned separately from documentation updates. All outputs under `/mnt/data`. The 1.12 TB latent cache stays on shared disk, not in the 960 GB allocation RAM. Source data/cache directories are read-only during training apart from existing reusable dataset index loading.

## Validation and measured speed

CPU suite:321 passed,24 skipped. GPU suite:328 passed,17 skipped;17 skipped cases require an optional ChronoLoop parent and are not part of this original-v0 run. New tests cover latent bypass (never encode dummy video), CPU seed42 interfaces, mean-to-sum loss gradients, robot checkpoint depth/loss-scale rejection, exact per-epoch coverage, tail95 and resume order. Independent review additionally executed sampler and prefetch bookkeeping across small/dummy-rank cases and microbatches 1/2/4/8/16, confirming DDP global normalization and LR endpoint.

Native audit:full 50dataset/episode counts, complete cache provenance and validity,1,024 finite latent samples, text sample loading, published normalization,673 exact donor tensor comparisons, original MoT type and 584,567,886 parameters. Previously cached full text extraction was audited over all 921,032 prompts; this launch reuses that audit and checks sampled prompts.

| Eight-GPU configuration | Seconds/update | Real windows/s | Peak allocated GB/rank |
|---|---:|---:|---:|
|micro4,accum4,eager,no activation checkpointing|1.3278|96.4|41.0|
|micro8,accum2,eager,no activation checkpointing|1.1089|115.4|68.8|
|micro16,accum1,eager,activation checkpointing|1.3369|95.7|17.6|
|**micro8,accum2,compiled,no activation checkpointing**|**0.6988**|**183.2**|**52.9**|

These are 30-update eager and 45-update compiled trials with initial 15 updates excluded; elapsed update time includes waiting for data. The exact same 128-window sequence and keyed per-window noise were used. Fused AdamW, structured attention with 480 observation tokens, gradient bucket views,4 text loader threads per rank, background prefetch. This is the fastest tested configuration; all production runs use fresh donor weights/optimizer rather than any benchmark or smoke checkpoint.

Native smoke gates require: ten fresh updates; strict checkpoint restore; loss wrapper equal to original loss; finite repeated-seed 32×14 inference; optimizer resume for two more updates with exact model/optimizer-state restoration and cursor validation; independent fresh-run consistency uses rtol1e-3/atol1e-4; and a real 95-window DDP tail using selected compiled configuration. Precision gate checks every trainable gradient exists/isfinite and Adam moments/weights are FP32. Production checks source hashes match accepted gates, test pass counts, exact audit, selected configuration and a new output directory.

Diagnosed preflight issues are preserved in evidence: initial feature tests expectedly failed before implementation; the base branch Dense-S30 test fixture lacked required native-donor metadata; headless simulator imports required the existing OSMesa environment; stats-file bytes and canonical JSON have different SHA256; the preprocessed data manifest used different JSON formatting, so semantic identity is checked without rewriting it. No production contract was relaxed. The superseded stream-slot benchmark stopped before training on the manifest-format guard; final production uses the corrected exact-batch trainer.

No simulator evaluation or RoboTwin success rate is claimed by this training launch.

An initial independent-run comparison failed an overly strict absolute 2e-4 loss threshold at update 2, before any save/resume. First-update losses were bit-identical, but gradients differed slightly; maximum observed early action-loss drift was 5.57e-4 (~0.024%). Parallel BF16/attention backward nondeterminism is a hypothesis, not isolated to a specific kernel. The corrected gate verifies every saved optimizer tensor/scalar and parameter-group setting exactly after restore, all model weights exactly on reload, and the resume cursor/next-window IDs/seeds/LR. Independent runs are compared only for numerical consistency. The failed gate is retained; production starts only after new accepted gates. No model, precision, data, LR or batch contract changed.

## Production release and exact-batch verification

Production was released on **2026-10-28 at 06:10 UTC**, on eight-H100 allocation **5489**, using runtime commit **de52d2fdfb3b2b7b4057f88ed4f5e0a4e3943712**. The selected layout is **8 GPUs × microbatch 8 × accumulation 2 = 128 real windows**. It uses a fresh canonical donor initialization and fresh optimizer. Production output: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin_full_v0_20261028/release_v2/production`.

Accepted native gates record exact optimizer restoration, verified resumed cursor, numerical consistency of independent runs, and a finite real 95-window epoch-tail update. All 109 source-file hashes bound to these gates match the committed runtime. The archived CPU/GPU logs record the pass counts above.

Budget arithmetic was independently rechecked: each epoch consists of 47,461 full updates plus one 95-window update. Thus 5 × 47,462 = **237,310 optimizer updates**, and 5 × 6,075,103 = **30,375,515 window presentations**. These are five passes over 6,075,103 unique windows. Epoch shuffle seeds are 42, 43, 44, 45 and 46. Gradients average over 128 windows on full updates and over the actual 95 windows on each tail update. The schedule reaches LR 1e-6 on update 237,310.

The benchmark projects **46.1 hours of update time**; the launch forecast allowing for initialization and checkpoint I/O is **48–52 hours**, approximately **October 30, 06:10–10:10 UTC**, provided allocation 5489 remains active. This is a launch forecast, not a completion measurement. SSH timed out on repeated checks while finalizing this report, so the newest production heartbeat could not be downloaded. Training completion and RoboTwin success rates remain unreported.

Evidence: [machine-readable report](LoopWAM_v0_Full_RoboTwin_20261028.json), [release record](../evidence/robotwin_full_v0_20261028/production_release.json), [accepted gates](../evidence/robotwin_full_v0_20261028/release_v2/gates_passed.json), [benchmark](../evidence/robotwin_full_v0_20261028/exact_batch/bench_results.json). Checkpoints remain on the server.
