# Centered Wan donor partition ablation — 2026-10-26

This campaign supersedes the earlier fixed-donor grid at the user's request. Six fresh models train on the full four-suite LIBERO dataset. No 3/6/3 production control is included in this queue. No completed training or new success-rate results are claimed in this launch report.

## Initialization and experiment contract

Both video and action experts select the corresponding converted native Wan donor tensors. Prelude takes the first P blocks; the shared core takes C contiguous blocks beginning at floor((30−C)/2); coda takes the final S blocks. Indices are zero-based and inclusive. Odd-width cores use the lower-index tie.

| Prelude/core/coda | Loops in both experts | Prelude donors | Core donors | Coda donors | Lane/order |
|---|---:|---|---|---|---|
|0/6/6|4|none|12–17|24–29|A/1|
|1/6/5|4|0|12–17|25–29|B/1|
|5/6/1|4|0–4|12–17|29|A/2|
|6/6/0|4|0–5|12–17|none|B/2|
|1/9/2|3|0|10–18|28–29|A/3|
|4/3/5|7|0–3|13–15|25–29|B/3|

Every model has 12 stored blocks and 30 block applications per expert, aligned action KV, and 584,536,135 policy parameters. The same artifact supplies unchanged non-block/head/proprioception tensors. Artifact SHA-256: `a3bf2097bafddac278148c1cb0321163f68cfe2b954e206436da0e3f0bd3c535`.

This experiment varies partition **and** selected donor layers together, as approved. It is distinct from the earlier experiment that held the twelve donor layers fixed. No trained checkpoint is resumed; each model gets a fresh optimizer and donor initialization.

Training: all 1,712 locally available demonstrations across four LIBERO suites; 277,713 real windows per epoch; ten epochs; 21,700 updates; 2,777,130 real windows; global batch 128; seed 42. Same dataset, normalization and complete frozen VAE latent cache as the original full-suite baseline. AdamW lr 1e-4, betas (0.9, 0.95), epsilon 1e-8, decay 0.01, clip 1, 5% warmup and cosine decay. FP32 policy weights and Adam moments, BF16 computation. This all-training protocol has no held-out demonstration split. Seed variance is not measured.

## Implementation and verification

Branch `partition_grid_20261026`, implementation commits `2211a53` and `96dc3f9`. The previous code base was `fb81dee`; the previous launch report was `1a57ac3`. Default canonical initialization is preserved. New behavior requires explicit `--donor-strategy stage_centered`; selected indices and strategy are stored and validated in architecture metadata and training contracts. Factory restoration rejects requested/saved strategy mismatches. Centered checkpoints use the partition-aware format even for a 3/6/3 reference.

Final project CPU suite: **473 passed, 16 skipped**. Final project GPU suite: **489 passed**. Native donor audit: all **679 tensors per model** compared exactly against the selected full donor bank for all six models, including unchanged non-block tensors; exact parameter counts verified. Tests cover exact selection, checkpoint guards, legacy behavior, partition forward/gradient/inference behavior, batch/precision gates and export retry integrity.

Independent review found two queue issues: insufficient benchmark acceptance checks and inability to safely recover an interrupted export before provenance was written. Both were fixed, with expected failing tests followed by passing tests. Benchmark validation now checks FP32 state, global batch, GPU count, dataset windows and finite loss/gradient/timing. Export recovery compares checkpoint metadata and every tensor against the original final checkpoint before reuse.

Native ten-update smoke runs and strict checkpoint reload/inference checks are required for **each** layout. The latter use real dataset input, current proprioception, 32×7 actions and repeated-seed bit-identical finite outputs. They are inference execution checks, not simulator evaluations. Production checks exact baseline data/asset/normalization/seed/update-budget parity and selected donor mapping. Source hashes and batch configuration bind both accepted smoke lanes to the production release.

## Resources, speed and output handling

The user explicitly authorized stopping SimpleMemVLN. Its identified step `4925.194` was canceled; its guardian relaunched it as `4925.198`, which was also canceled. The campaign's documented `guardian/STOP` control was created and the guardian confirmed exit. The superseded fixed-donor LoopWAM step `5248.63` was stopped. Existing outputs and both parent allocations/original interactive shells were preserved.

Current allocations are **5248 and 4925, four H100s each**, both on worker-2. Each lane uses all four GPUs, with three sequential runs. Lane A uses allocation 5248; lane B uses 4925. Eight GPUs total. Data, checkpoints, logs and exports remain under `/mnt/data`, not home.

The four-GPU sweep compares DDP, ZeRO-1 and ZeRO-2 at global batch 128. It tests microbatch 8 without activation checkpointing, larger uncheckpointed batches, and checkpointed microbatches up to 32. The common centered 3/6/3 model is a throughput reference only. Some initial ZeRO trials overlapped SimpleMemVLN's unexpected restart; the entire ZeRO sweep was repeated after stopping its guardian. Only the clean repeat is used for the final choice. OOM trials are recorded as capacity failures, not successful timings.

Production uses fused AdamW, gradient bucket views, structured attention, four dataloader workers per rank, and the existing latent cache. Checkpoints replace `latest.pt` every 500 updates and at epoch ends; this queue does not retain a separate file for every epoch. The final local checkpoint retains optimizer state. Errors stop the affected lane. No simulator evaluation is scheduled between runs, following the training-only queue instruction.

After each complete ten-epoch run passes verification, the runner exports FP32 `policy.pt` without optimizer plus normalization, data/training manifests, timing, fairness evidence and provenance. It checks source/export hashes, rejects different existing released weights, verifies the HF weight LFS hash and downloaded metadata hashes. Three failed upload attempts create an explicit pending-upload marker while preserving training and allowing the next model to train.

Destination: `anhdao69/LoopWAM_NT/libero-all-suites/partition-grid-centered-20261026/<label>/`. Uploads are configured, not yet completed at this launch snapshot. The interactive allocations must remain active; the pipeline does not cancel them or automatically relaunch a failed production step.

Server root: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_centered_20261026`.
Source: `/mnt/data/vmo-ai-task/anhdh35/LoopWAM_partition_grid`.
Evidence: `plans/evidence/partition_centered_20261026/`.


## Accepted release and measured timing

All six native ten-update training, fairness and checkpoint-inference gates passed. Production released at **2026-10-26 18:50:56 UTC**, source `96dc3f97312e546c3b11664da17fbb0d1ee5d84e`. Both first production runs were verified at update 29 after fresh starts: 1.8787 and 1.8746 seconds/update, finite losses/gradients. Production steps are 5248.70 and 4925.201. A GPU snapshot showed 95–100% utilization and 58,641–58,653 MiB used per device (81,559 MiB total). GPU UUIDs confirm eight distinct devices; CUDA indices are remapped to 0–3 separately in each allocation. See [release](../evidence/partition_centered_20261026/release.json) and [production snapshot](../evidence/partition_centered_20261026/production_start_verified.json).

| Backend | Microbatch/GPU | Activation checkpointing | Seconds/update | Outcome |
|---|---:|---|---:|---|
|ddp|8|no|1.8740|passed|
|ddp|16|no|—|oom|
|ddp|16|yes|2.2431|passed|
|ddp|32|yes|2.0656|passed|
|zero1|8|no|1.9144|passed|
|zero1|16|no|—|oom|
|zero1|32|yes|2.2259|passed|
|zero2|8|no|1.9670|passed|
|zero2|16|no|—|oom|
|zero2|32|yes|2.1739|passed|

OOM rows have no accepted timing. Eight-update reference trials exclude initial warm-up. **Selected DDP, microbatch 8 × four GPUs × accumulation 4 = global 128, no activation checkpointing.** The clean ZeRO repeat confirms the selection. This is the fastest tested layout; small timing differences are not a guarantee across all workloads.

| Model | Native smoke seconds/update | Projected compute hours | Lane/order | Reload/inference |
|---|---:|---:|---|---|
|p0c6s6_k4|1.8839|11.36|A/1|passed|
|p5c6s1_k4|1.8764|11.31|A/2|passed|
|p1c9s2_k3|1.8747|11.30|A/3|passed|
|p1c6s5_k4|1.8804|11.33|B/1|passed|
|p6c6s0_k4|1.8690|11.27|B/2|passed|
|p4c3s5_k7|1.8758|11.31|B/3|passed|

Native smoke projections total **33.97 h for lane A** and **33.91 h for lane B**, before setup, checkpoint and upload overhead. Budget **about 12 hours for the first pair** and **35–38 hours for all six**. Estimated final window: **2026-10-28 05:50 to 2026-10-28 08:50 UTC**, assuming uninterrupted allocations and similar sustained throughput. These are forecasts, not measured completed runtimes.

Final regression evidence: [CPU](../evidence/partition_centered_20261026/cpu_final_tests.log), [GPU](../evidence/partition_centered_20261026/gpu_final_tests.log), [native exact donor audit](../evidence/partition_centered_20261026/native_donor_audit.json). No new final weights or simulator SR are available at this snapshot.
