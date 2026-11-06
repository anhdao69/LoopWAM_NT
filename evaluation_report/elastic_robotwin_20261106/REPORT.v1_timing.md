# RT50-E0/E1/E2 timing on H100 (job 5758, worker-2), 2026-11-06

Code: /mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_rt, branch elastic_robotwin @ 496aa51 (local commits, not pushed).
Teacher = student init: RoboTwin v0 4/4 epoch_10.pt (sha 70cf5af7..., = HF robotwin/v0-video4-action4/epoch_10/policy.pt).
Data: RoboTwin-50 v0 training windows (6,075,103) + v0 latent cache (NFS, read-only). Global batch 128 -> 47,462 updates/epoch.
Speed flags: --bf16-frozen --teacher-graph --no-fill-uninitialized --gc-freeze, no block checkpointing, 15 NUMA-local CPUs/rank.

## Correctness checks (all passed)
- 86 CPU tests (new RoboTwin tests + existing elastic/transport tests: LIBERO behaviour unchanged)
- validation/preflight_core.json: teacher + data identity; dataset == RoboTwin trainer fields on 450 windows (incl. padded ends);
  teacher rollout bit-equal to deployed infer_action (4/4 windows); param counts; common_v1 init + trainable counts
- validation/preflight_speed_E{0,1,2}.json: each speed option and all together bit-exact (loss and gradients) for E0/E1/E2
- Timing runs: finite losses, grip 0, every 9-update block a full FLAT9 permutation, data wait <= 2 ms
- Same layout repeated (incl. GPUs 0-3 vs 4-7, under concurrent load): all 108 updates bit-identical
- 8-GPU mb16 vs 4-GPU mb32: identical schedules/windows, update-1 losses equal to 6 decimals, rounding-level differences
  afterwards (largest relative only on ~1e-5 losses), no drift by update 108

## Timing (108 updates/run, first 9 = warm-up excluded; hours include a ~2 s recovery save per 100 updates)
| layout | run | GPUs | mb | s/update (wall) | save s | data wait max | peak GB | 1 epoch h | 21,700 upd h | 10 epochs h |
|---|---|---|---|---|---|---|---|---|---|---|
| w4_mb32 | RT50-E0 | 0,1,2,3 | 32 | 0.975 (0.978) | 2.0 | 0.001 | 70.5 | 13.2 | 6.0 | 131.6 |
| w4_mb32 | RT50-E1 | 0,1,2,3 | 32 | 0.965 (0.970) | 1.8 | 0.000 | 70.5 | 13.0 | 6.0 | 130.3 |
| w4_mb32 | RT50-E2 | 0,1,2,3 | 32 | 0.985 (0.986) | 2.3 | 0.001 | 70.4 | 13.3 | 6.1 | 133.1 |
| w4x2_mb32 | RT50-E0 | 0,1,2,3 | 32 | 0.974 (0.976) | 1.9 | 0.001 | 70.5 | 13.1 | 6.0 | 131.1 |
| w4x2_mb32 | RT50-E1 | 4,5,6,7 | 32 | 0.970 (0.973) | 2.1 | 0.001 | 70.5 | 13.1 | 6.0 | 131.1 |
| w8_mb16 | RT50-E0 | 0,1,2,3,4,5,6,7 | 16 | 0.749 (0.754) | 2.2 | 0.001 | 37.0 | 10.2 | 4.7 | 102.3 |
| w8_mb16 | RT50-E1 | 0,1,2,3,4,5,6,7 | 16 | 0.760 (0.765) | 1.8 | 0.002 | 37.0 | 10.3 | 4.7 | 103.2 |
| w8_mb16 | RT50-E2 | 0,1,2,3,4,5,6,7 | 16 | 0.767 (0.771) | 2.2 | 0.001 | 37.0 | 10.5 | 4.8 | 104.6 |
