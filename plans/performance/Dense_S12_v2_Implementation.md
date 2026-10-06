# Dense-S12 and v2 four-H100 implementation plan

**Goal:** In existing interactive allocation 4659, run fresh Dense-S12 training, 100 LIBERO-Long rollouts with videos, fresh v2 training, then the same evaluation.

**Spec:** `plans/LoopWAM_S_Technical_Report.md`, especially 7.7, 10, and 13. Dense-S12 uses the same canonical selected twelve donor pairs, one middle traversal, final-only supervision, no XSA, and exactly 584,536,135 trainable policy parameters. v2 retains four shared-core repetitions, joint all-exit supervision, and core-only XSA.

## Constraints and design

- Keep existing v0/v1 jobs unchanged. Job 4659 was inspected: four H100s idle, no SimpleMemVLN processes remain.
- Work in an isolated local git worktree; launch long jobs from a pinned remote checkout.
- Both models start from the canonical Wan initialization with empty optimizers, global batch 128, ten epochs, 7250 updates, 926780 real training windows.
- Preserve native VAE, text, precision, split, losses, AdamW settings, exact tail weighting, and evaluation protocol.
- Measure DDP and ZeRO-1/2, increasing per-GPU microbatch to reduce accumulation. Expected CUDA OOM rejects a candidate; any other error aborts.
- Only select a production backend after real forward/backward, cold-cache memory, checkpoint reload, and simulator smoke tests pass. If ZeRO wins, implement and verify its full production path before selecting it.
- Dense and v2 run sequentially and share their own immutable-preprocessing latent cache, with provenance validation and no concurrent writers. Neither shares caches with active v0/v1.
- Benchmark and smoke outputs are separate from both fresh training runs. Final evaluation requires complete training; v2 starts only after dense final evaluation succeeds.
- Use 100 episodes per model (ten per task), four GPU workers, saved videos, 700 policy steps, 30 settling steps, ten denoising steps, action replanning every ten steps.

## Tasks and verification

- [x] Add explicit `dense_s12` model version, one-traversal enforcement, and correct checkpoint reconstruction. Test exact dense/v0-K1 forward and gradient equivalence, twelve unique blocks, final loss, and invalid depth rejection.
- [x] Generalize evaluator checkpoint guards and architecture metadata to dense and v2, with correct depth and four-rank task coverage. Test complete-state, normalization, and version/depth mismatch rejection.
- [x] Extend benchmark/trainer version handling and four-rank backend configuration. Benchmark warm steady state and test selected cold native-VAE path.
- [x] Add sequential fail-closed orchestration, fresh-output guards, stage duration and runtime estimates, and independently tested complete-training/evaluation gates.
- [x] Run CPU suite, native four-GPU model training smokes and checkpoint simulator smokes. Review integrated diff before production.
- [ ] Pin and synchronize source, start a detached Slurm step in allocation4659, verify fresh dense production updates, save measured runtime report, and push locally to authorized private repository.

## Review focus

1. Dense accidentally runs four loops or has deep-supervision/XSA enabled.
2. Final evaluation accepts partial/wrong-version checkpoints or wrong normalization.
3. Global batch or tail-window weighting changes with four ranks or backend.
4. Benchmark/smoke optimizer weights leak into production initialization.
5. A failed stage starts the next model, or concurrent processes write the same cache.

## Evidence log

- Initial remote inspection: 4659 worker-3, four idle H100 GPUs, no compute processes; allocation has a 365-day limit. v0 allocation4689 and v1 job4691 remain running.

- CPU verification: `python -m pytest -q tests` →149 passed,7 skipped (CUDA-only),3 upstream robosuite deprecation warnings. Bare repository-wide discovery also collects unrelated vendored RoboTwin scripts and fails on absent optional `openai`/`sapien`; no dependencies changed for those external tools.
- Read-only model/pipeline/backend reviews found one provenance gap: benchmark model/backend source hashes were not compared. Added a failing regression, implemented hash binding and unchanged-source preflight guard; scoped tests passed.
- Four-GPU benchmarks compare actual Dense-S12 and v2, global128, FP32 policy/master/moments and BF16 compute. More workers and repeated finalists are included because dense throughput shows data-loader stalls.

- Four-rank production backend gate passed for real global group sizes128,18,128,6. Maximum DDP-vs-ZeRO1/2 parameter difference1.49e-8; first/second-moment differences7.45e-9/1.86e-9. Portable and native weights/moments/LR/step reload verified exactly.
- Recovery test diagnosis: DeepSpeed0.18.7 restores native optimizer states but initializes optimizer-fragment inspection views lazily; querying `safe_get_full_optimizer_state` immediately after load raised a test-only TypeError and delayed NCCL teardown. The test now initializes those views and compares actual moments against CPU checkpoint shards. No production-code correction was required.
- CUDA-specific attention/cache/checkpoint checks:7 passed.
- Longer15-update finalist measurements selected Dense-S12 ZeRO1/microbatch32/workers8 (0.892858s/update), v2 DDP/microbatch8/workers4 (2.560990s/update pooled warm measurements). Native full-size smoke and simulator preflights follow before production.

- Native preflight completed successfully in4659.16: dense and v2 each passed ten cold-cache and ten warm-cache full-size training updates, then four simulator episodes of100 policy steps with ten replans and videos. These smoke rollouts validate execution, not learned success. Source hashes were unchanged across preflight.
