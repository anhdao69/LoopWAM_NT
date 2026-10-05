# LoopWAM NT Implementation Plan

Spec: LoopWAM_S_Technical_Report.md. User explicitly requests implementation of v0/v1/v2, GPU smoke tests, and v0 LIBERO-Long training for ten epochs at global batch 128 on branch LoopWAM_NT.

## Constraints
- Base 7faa71108368fbb3b6885649f112af607427a2d4; preserve pre-existing environment edits.
- Wan2.1 native source; video 1536/6144, action 512/2048; 12 heads of 128; physical depths 3/6/3, K=4.
- Same canonical initialization, no robot teacher. Source indices 0,1,2,9..14,27..29.
- v0 final loss; v1 exits 1/6,1/6,1/6,1/2; v2 same plus core-only FP32 XSA.
- Native masks, timesteps, RoPE, 32 actions, online frozen VAE initially; cached text with masks.
- Ten real data passes, global batch 128, AdamW FP32 states, BF16 compute; record actual windows/updates and timings.

## Tasks
- [x] 1. Native source validation, canonical compact donor conversion and strict target loading. Files loopwam_init.py and preprocessing CLI. Tests for paired FFN selection, interpolation, source coverage and metadata.
- [x] 2. LoopMoT schedule and intermediate exits; virtual cache prefill/action inference and core XSA. Files loop_mot.py and tests. Tests K1 dense/K4 unrolled, gradients, coda isolation, causality, cache equivalence, XSA zero values.
- [x] 3. LoopWAM policy, loss reductions, builder/configs and strict checkpoint metadata. Files loopwam.py, runtime/config additions, tests. Tests padded losses, independent noise, scheduler sign, parameter count and reload.
- [x] 4. LIBERO Long deterministic episode split, train-only normalization, budget and distributed runner. Validate timestamps, real shapes and VAE anchor; smoke v0/v1/v2 with finite gradients and optimizer updates.
- [x] 5. Independent integrated review, fix material findings, launch v0 ten-epoch runs in available allocations. Record measured timing and extrapolation distinctly; preserve logs and reproducible commands.

## Review focus
- No clean/future video leakage into action predictions.
- Shared parameters counted and optimized once; proprio encoder included.
- Last accumulation group and distributed sampler padding explicitly handled.
- Saved student reconstructs compact model without native preset override.
- Existing occupied jobs must not be interrupted without permission.

## Execution record
Initial inspection: SSH login-0 works; interactive 4689 worker-2 2 GPUs and 4659 worker-3 4 GPUs accessible. 4659 occupied by SimpleMemVLN; permission pending. 4657 is an unrelated batch job.
Ruling: user supplied detailed specification and explicitly authorized implementation; execute that brief without reopening design approvals. Work on requested branch in existing checkout to preserve environment preparation.

Verification record: CPU suite45 passed,2 CUDA checks skipped; both CUDA checks subsequently passed. Native artifact strict loading and native VAE verified. Multi-GPU real-data smoke passes v0/v1/v2 global128,3 updates each. Full-size checkpoint reconstruction and fixed held-out inference verified. Two-GPU optimizer/RNG continuation verified; direct uninterrupted comparison pending final check.
Independent review: initialization agent reviewed integration, trainer, data and recurrence it did not author. No critical training defect found. Fixed inference default20->10, required proprioception, explicit32-action contract, and added code/model/data hashes. Final fresh reviewer spawn was rejected by the harness thread limit; independent integration review reused an existing agent.
Ruling: use replicated FP32 AdamW with DDP instead of initial proposed ZeRO-1 — measured fit25.9GB at microbatch2; preserves full precision/master state and avoids unnecessary sharding complexity. Cost is higher optimizer residency than ZeRO-1.
Ruling: dataset actually388 episodes; stratified90/10 produces344/44 instead of assumed450/50 — use existing supplied data and report its size. Cost is reduced data exposure relative to hypothetical500-demo setup.
Ruling: preserve online VAE encoding for this run to retain validated anchoring semantics. Cost is extra compute compared with a validated latent cache.

Final acceptance: 48/48 tests passed on H100, including both BF16 CUDA cases (runs/loopwam_nt/continuous_smoke.log). Two-GPU resumed and uninterrupted update2 have identical reported action/video losses, learning rate, and windows seen. Full-size student reconstructs without donors and performs held-out10-step action inference.
Production launch: source commit ca8df93a59a527d2cdd660e1a437e53c84d86ac2, branch LoopWAM_NT both hosts. Job4689 step4689.49 on worker-2; twoH100s, microbatch2, accumulation32, global128,10epochs. Output runs/loopwam_nt/v0_long_bs128_20261005. First production update verified finite; run continues independently of SSH.
