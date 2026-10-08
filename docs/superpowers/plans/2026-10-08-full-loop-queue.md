# Full LIBERO loop ablation implementation plan

**Goal:** Fresh full-suite training for 2/2, 4/1, 1/4, 3/3 in two two-H100 jobs, each train/eval/train/eval.
**Architecture:** Persistent independent simulator/model workers dynamically consume episode jobs. Benchmark replica counts without changing per-episode inference. Native training uses a provenance-checked frozen latent cache and the fastest valid measured backend/batch. Fail closed before downstream stages.
**Tech Stack:** Python, PyTorch, CUDA, LIBERO, Slurm.
**Spec:** User request and clarification: same settings as full 4/4, three repetitions of seed 42.

## Constraints
- Fresh canonical initialization, 10 epochs, global 128, seed42, all 1712 demos and 277713 windows, 21700 updates.
- Same FP32 parameters/moments and BF16 compute. Same optimization and normalization.
- Each evaluation: all 4 suites, 10 tasks each, 10 episodes each, 700 steps, 30 settle, chunk32/replan10/denoise10/CFG1. Repeat seed42 three times.
- Only allocated GPUs; no changes to other workloads. Source pinned for each job.

## Tasks
- [x] Add persistent rollout pool and grid/result identity tests.
- [ ] GPU concurrency benchmark with identical episode action traces; reject incompatible outputs or failures.
- [ ] Full-data throughput trials and native training/evaluation smoke; validate fresh optimizer and exact coverage.
- [ ] Two Slurm files with preflight, release gate, sequential train/eval/train/eval and final completeness gates.
- [ ] Submit, inspect first finite training updates, document measured settings and finish estimates.

## Review focus
Worker failures, duplicate/missing episodes, stale checkpoints/cache, GPU oversubscription, incorrect epoch completion must fail explicitly. Repeated same-seed results are not independent statistical evidence. Two-GPU random-number consumption may differ from original four-GPU training.
