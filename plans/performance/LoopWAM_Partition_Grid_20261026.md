# Full LIBERO LoopWAM partition ablation — 2026-10-26

Status: **all preflight gates passed; production running in Slurm step5248.63** on worker-2. Both first experiments were verified at update23 with finite losses/gradients. No final training or SR results are claimed.

## Code and reviewed context

Fetched the latest remote refs. This experiment extends the latest KV_concat implementation, base `8484ec2`, in isolated branch `partition_grid_20261026`. The newer chrono branch is a separate RoboTwin architecture and is not substituted for the requested LoopWAM v0. The existing residual_trans and RoboTwin checkouts remain untouched. Implementation commits: `905897d` (model/contracts/tests), `9f2e844` (queue, checkpoint inference gate, exports), `fb81dee` (current-proprioception inference helper fix).

Reviewed the consolidated October 8 report, KV-concat implementation/run report, latest `evaluation_report/final/seed_results.csv`, full-suite runner/trainer/model/initialization/evaluator code, and live original full-suite manifest/timing. The latest expanded full 4/4 baseline is 5,695/6,000 successes (94.92%) across four suites and seeds42/43/44, 500 episodes per suite/seed. This differs from the older 388/400 report because evaluation coverage changed. This campaign schedules training only, following the most recent training-only queue instruction; it does not claim new SR.

## Design and fairness

| Prelude | Core | Coda | Video/action loops | Stored blocks/expert | Applications/expert |
|---:|---:|---:|---:|---:|---:|
|0|6|6|4 / 4|12|30|
|1|6|5|4 / 4|12|30|
|3|6|3|4 / 4|12|30|
|5|6|1|4 / 4|12|30|
|6|6|0|4 / 4|12|30|
|1|9|2|3 / 3|12|30|
|4|3|5|7 / 7|12|30|

All are v0, aligned KV, final-exit supervision, 584,536,135 policy parameters. Keep the same twelve physical donor blocks in order `[0,1,2,9,10,11,12,13,14,27,28,29]`; only the contiguous stage boundaries and repetitions change. This controls initialization across the partition ablation. Selecting different donors per stage would test initialization and partition jointly. The 3/6/3 entry is a fresh control; existing weights are preserved.

Ten epochs, seed42, global batch128, fresh canonical donor initialization and AdamW optimizer. Full four-suite dataset means all 1,712 locally available demonstrations, 277,713 windows/epoch, 21,700 updates, 2,777,130 real windows. No held-out demonstration split in this all-training protocol. AdamW lr1e-4, betas(.9,.95), eps1e-8, decay.01, 5% warmup/cosine, clip1. FP32 weights and moments; BF16 compute. Same data/normalization/assets and complete frozen latent cache as the earlier full-suite run. Benchmark data manifests matched exactly except output-local normalization paths.

Default model outputs are bit-identical with explicit3/6/3. Non-default checkpoints use `loopwam-s-partition-v1`, so old loaders reject them rather than silently applying3/6/3. Updated loaders preserve/validate `block_partition`, loop capacity, training contract and state dicts. Legacy checkpoints infer their original partition. Custom partitions are restricted to aligned v0.

## Validation

Final CPU project suite: **449 passed, 16 skipped** (GPU-only). Final GPU project suite: **465 passed**. Tests cover explicit 30-layer unroll outputs and all block gradients, empty prelude/coda, checkpointed gradients, future-video isolation, cached inference on CPU/GPU, seven-loop loss, factory/checkpoint round trips, mismatched partitions and release guards.

Failures diagnosed and retained: initial missing partition API (expected TDD red); a test helper hard-coded3/6/3; three validation-message regressions; missing LIBERO import path in the initial test environment. Unrestricted `pytest` also collected unrelated vendored RoboTwin utilities and failed for missing `openai`/`sapien`; those optional utilities are not part of the passing `pytest tests` suite. The first native training smokes passed, but inference gating rejected the helper passing a full32-state proprioception sequence. Fixed it to use only the current state, added a future-proprioception isolation regression test, and restarted all native smokes in fresh `preflight_r2` directories. No tolerance relaxed. Independent review identified smoke/source/config binding, stale export reuse and incomplete remote metadata verification; all three were fixed before production. Added checkpoint-format and training-contract corruption guards.

## Measured throughput and queue

Both pairs benchmarked the same full-data 3/6/3 reference, eight updates each; steady timing excludes warm-up.

| Layout on two H100s | Pair A s/update | Pair B s/update | Result |
|---|---:|---:|---|
|DDP mb8, accumulation8|3.7367|3.7369|Selected|
|DDP mb16, accumulation4|—|—|CUDA OOM|
|DDP mb16, accumulation4, activation checkpointing|4.4177|4.4297|Slower|

Use the same selected layout for all seven runs to preserve random-number consumption/layout as far as the architectural change permits. Fused AdamW, gradient bucket views, canonical structured attention, warm VAE cache. Earlier backend sweeps favored DDP; this campaign remeasured batch/checkpointing candidates, not ZeRO.

Allocation5248, worker-2, four H100s; two isolated device pairs inside one Slurm step. Allocation4925 and the user's interactive shell are untouched. Lane A: 0/6/6 → 5/6/1 → 1/9/2 → fresh3/6/3. Lane B: 1/6/5 → 6/6/0 → 4/3/5. Seven equal-cost runs require four sequential stages on one pair and three on the other. Estimated reference training22.5h/run before checkpoint/setup overhead; final estimate will use native smoke and production observations. All data/checkpoint/log outputs reside under `/mnt/data`, not home.

Every experiment starts fresh after all preflight smokes pass, fails closed on training errors, checks finite losses/gradients/window accounting and checkpoint completeness, and checkpoints every500updates and at epoch ends. Final local checkpoints retain optimizer state. Upload retries do not discard completed training.

## Deliverables

Server evidence root: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_grid_20261026`.
Production and accepted preflight root: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_grid_20261026/preflight_r2`.
Source: `/mnt/data/vmo-ai-task/anhdh35/LoopWAM_partition_grid`.
HF destinations: `anhdao69/LoopWAM_NT/libero-all-suites/partition-grid-20261026/<label>/`.
Each release contains `policy.pt` (FP32 policy without optimizer), normalization, data manifest, training manifest/timing, fairness evidence and provenance. Remote weight LFS SHA and downloaded metadata hashes must match local exports before marking a release verified. Cached exports must match the original source-checkpoint hash. No new weight has been published at this launch snapshot; uploads occur after complete ten-epoch training. Worker-to-Hugging-Face connectivity and account access were checked; the cached token includes repo.write for anhdao69.

## Accepted preflight and production launch

All seven fresh native10-update runs passed exact baseline data/asset/normalization/init/seed/update-budget checks, finite losses and gradients, checkpoint metadata validation and strict native factory loading. Each reloaded checkpoint produced finite32x7 actions twice with bit-identical repeated-seed outputs. These are inference execution checks, not simulator SR evaluations. The inference helper uses only the current proprioception state.

| Configuration | Native seconds/update | Projected training hours | Lane/order |
|---|---:|---:|---|
|p0c6s6_k4|3.6846|22.21|A/1|
|p5c6s1_k4|3.6861|22.22|A/2|
|p1c9s2_k3|3.6924|22.26|A/3|
|p3c6s3_k4|3.6629|22.08|A/4|
|p1c6s5_k4|3.7052|22.33|B/1|
|p6c6s0_k4|3.6959|22.28|B/2|
|p4c3s5_k7|3.6993|22.30|B/3|

Released at **2026-10-26 15:12:10 UTC / 11:12:10 EDT**, pinned source `fb81deecac719b1c29eae5c219c773ba92d2cb5a`. Device sets are disjoint: laneA uses0,1; laneB uses2,3. At15:14:18UTC both production runs had23updates, about3.73s/update, finite losses/gradients, and GPU utilization100% on all four devices (~58,590MiB used per GPU). Manifests confirm fresh canonical initialization, no resume, global128, ten epochs,21,700updates and2,777,130real windows.

Native projection totals: laneA **88.76h**, laneB **66.91h**, excluding setup/checkpoint/upload overhead. Allow **90–96h for the entire queue**, approximately **October30, 05:12–11:12 EDT** (16:12–22:12 Vietnam time). First two final checkpoints are expected in about23hours, around October27 morning EDT. These are forecasts; live timing takes precedence. The existing interactive allocation must remain active for the queue to continue. It is not automatically canceled by this pipeline.

Raw evidence: [release](../evidence/partition_grid_20261026/preflight_r2/release.json), [production start](../evidence/partition_grid_20261026/preflight_r2/production_start_verified.json), [CPU tests](../evidence/partition_grid_20261026/cpu_helper_fix_tests.log), [GPU tests](../evidence/partition_grid_20261026/gpu_helper_fix_tests.log). Source and report branch: [partition_grid_20261026](https://github.com/anhdao69/LoopWAM_NT/tree/partition_grid_20261026). Existing branches were not merged or rewritten.
