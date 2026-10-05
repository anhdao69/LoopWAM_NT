# LoopWAM v0 speed optimization and fresh restart

## Selected setup

Two H10080GB GPUs in interactive allocation4689 only. SimpleMemVLN in4659 is unchanged. DDP with FP32 policy/AdamW states, BF16 compute, fused AdamW, gradient bucket views, per-GPU microbatch8, accumulation8, global batch128. Canonical LoopWAM-S v0 architecture/objective/data/optimizer hyperparameters are unchanged.

The user explicitly requested a **fresh restart** after benchmarking: canonical Wan donor initialization, empty optimizer state, scheduler/update/data cursor zero, full ten epochs. No old or benchmark policy checkpoint is loaded. The original run stopped at update161; its update100 checkpoint and logs remain available.

## Measurements

Trials are sequential on both GPUs, with real LIBERO-Long windows. Initial trials use five optimizer updates and discard the first two for timing. Cached trials use ten updates and discard the first two. Timings include data acquisition and model/optimizer work; data loading is not the primary bottleneck. Memory below is maximum allocated across both ranks, in decimalGB.

| Trial | Microbatch/GPU | Accumulation | Seconds/update | Allocated GB |
|---|---:|---:|---:|---:|
| baseline/ddp_mb2 | 2 | 32 | 9.4927 | 24.78 |
| baseline/ddp_mb2_unfused | 2 | 32 | 9.6822 | 24.78 |
| baseline/ddp_mb4 | 4 | 16 | 7.1064 | 36.12 |
| baseline/ddp_mb8 | 8 | 8 | 6.2659 | 58.58 |
| baseline/zero1_mb8 | 8 | 8 | 6.3112 | 55.36 |
| baseline/zero2_mb8 | 8 | 8 | 6.4209 | 53.99 |
| cached/fill_ddp_mb8 | 8 | 8 | 6.2654 | 59.29 |
| cached/warm_ddp_mb8 | 8 | 8 | 3.8113 | 59.48 |
| cached/warm_zero1_mb8 | 8 | 8 | 3.9012 | 56.31 |
| cached/warm_zero2_mb8 | 8 | 8 | 4.0732 | 54.90 |
| structured/ddp_mb8 | 8 | 8 | 6.1486 | 59.39 |
| structured/zero1_mb8 | 8 | 8 | 6.1646 | 56.18 |
| structured/zero2_mb8 | 8 | 8 | 6.3115 | 54.82 |

Microbatch16 failed with CUDA OOM under DDP, ZeRO1 and ZeRO2 in the initial matrix; structured DDP16 also failed. Those attempts are not successful speed results. All successful comparisons preserve global128 and FP32 master/moment/communication precision. ZeRO1/2 are actual DeepSpeed0.18.7 engines, not simulated labels.

**Projected ten-epoch training time: 8.17hours**, using725×(6.2654+9×3.8113)/3600. This includes the first epoch's lazy cache population; it does not incorrectly extrapolate warm-only throughput to all ten epochs. Checkpoint, validation and setup overhead are additional. The original run projected about19.6hours; this is roughly2.4×faster overall. A fresh production cache is used, so benchmark cache population is not counted as completed training.

## Verified changes

- Larger microbatch reduces accumulation32→8. FP32 fused AdamW, DDP bucket views, grouped metric reductions, and one all-rank finite-loss guard per optimizer update.
- Frozen VAE temporal caches are cleared after encoding; native repeated clip/image outputs remain equal.
- Canonical attention mask is exactly decomposed into observed→observed, future→allvideo, action→observed+action calls. Runtime mask validation rejects incompatible masks. Text padding masks, v2 XSA, and virtual-layer inference caches retain their semantics. GPU tests force mask-free FlashAttention at native token/head dimensions and compare outputs/gradients.
- Lazy per-training-window VAE cache stores exact BF16 bits (~14GB for92678windows), uses immutable training indices, verifies source/VAE/transform/split/normalization provenance, flushes payload before publishing validity, never caches dummy indices, and regenerates noise/timesteps every update. Validation uses online encoding. Caches have one training-job writer set at a time.
- DDP input auto-transfer is disabled with device_ids=None; model input handling explicitly transfers compute tensors, while indices remain on CPU. Cache hits avoid unused RGB GPU transfers.

CPU acceptance:87passed,7CUDAcases skipped. GPU attention/cache/policy gate:46passed. Actual two-rank DDP/ZeRO1/ZeRO2 full and partial-batch acceptance passed: maximum parameter difference1.49e-8, first moment2.33e-10, second moment3.64e-12. Scalar/fused AdamW maximum parameter difference1.19e-7 over seven updates. A rank-local NaN was rejected on every rank with zero optimizer updates. Final all-tests GPU gate: **94passed in22.50seconds**, recorded in ../evidence/loopwam_speed/cached/final_gpu_tests.log. Fixed-noise cache miss/hit tests match prepared tensors, losses, policy/proprio gradients and padding exactly.

The first real cache attempt exposed DDP auto-transfer of CPU index metadata and failed before any optimizer update. The corrected attempt is speed_cached_retry1; failed attempts are retained remotely. Cache counters in benchmark JSON are rank0-local, not global coverage claims. The production runner checks full cache coverage after a fresh complete epoch.

## Fresh production run

Output: runs/loopwam_nt/v0_scratch_fast_bs128_20261005

```bash
bash scripts/launch_loopwam_slurm.sh 4689 runs/loopwam_nt/v0_scratch_fast_bs128_20261005 \
  --config configs/loopwam_s_v0_fast.yaml \
  --latent-cache-dir runs/loopwam_nt/v0_scratch_fast_bs128_20261005/latent_cache --smoke
```

The manifest must show resume=null, initialization_mode=canonical_wan_artifact_fresh_optimizer, optimizer state entries before training0, and planned_updates7250. Each epoch covers92678real windows exactly, including the final six-window batch. Ten epochs cover926780real windows. First-epoch timing.json initially projects current cold throughput across all epochs; use the mixed cold/warm estimate above until later epochs provide production cache-hit timings.

Verified fresh production startup at 2026-10-05T22:56:07.597216+00:00: **Slurm step4689.61**, source commitc35e9d880a5f85d5dd4bee588f8a0cd7fdc777d0, optimizer state entries0 before training, resume=null, first update1/windows128. At verification, update12/7250 and 1536windows completed with finite losses/gradients. Mean observed cold production update time6.351s; combining that with measured warm throughput gives approximately8.19hours plus checkpoint/validation/setup overhead. Sampled SM utilization averaged97.2% across both GPUs (10samples/GPU), with approximately60GB framework-reserved memory/GPU. The fresh cache was empty at startup. Training continues under nohup while allocation4689 remains alive. Timing is an estimate; training completion and manipulation success are not claimed.
