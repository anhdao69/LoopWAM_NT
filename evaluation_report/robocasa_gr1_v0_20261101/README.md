# LoopWAM v0 4/4 on RoboCasa-GR1 tabletop: training and official evaluation

**Status: complete.** Training verified complete 2026-11-03 23:11 UTC: 117,580 updates (5 epochs, global batch 256, 4 H100), 0 restarts. Official evaluation (epoch 3 on ws; epochs 4 and 5 on both H100 and ws) and an epoch-4 action-step sweep on ws finished 2026-11-05. Code: branch `robocasa_gr1_v0` (eval with `--render egl`, commit d937167).

Protocol: starVLA `examples/simBenchmarks/Robocasa_tabletop` (24 tasks x 50 rollouts, any-step success, 720 steps, n_action_steps 12, n_envs 1, seed 0); references StarVLA-OFT-Qwen3 48.8 and GR00T-N1.6 47.6 from that README @13d8ad4.

## Results

| checkpoint (HF `anhdao69/LoopWAM_NT` Robocasa-GR1/) | update | host | success |
|---|---|---|---|
| epoch_03.pt `a8d7ec33913d...` | 70,548 | ws (RTX PRO 6000) | 32.67% (392/1200) |
| epoch_04.pt `b62483a91bca...` | 94,064 | H100 | 44.08% (529/1200) |
| epoch_04.pt `b62483a91bca...` | 94,064 | ws (RTX PRO 6000) | 46.25% (555/1200) |
| epoch_05.pt `1597816e9cc1...` | 117,580 | H100 | 44.58% (535/1200) |
| epoch_05.pt `1597816e9cc1...` | 117,580 | ws (RTX PRO 6000) | 44.50% (534/1200) |

Pooled over both hosts: epoch 4 45.17% (1084/2400), epoch 5 44.54% (1069/2400). Per-task table: [official_results_all.md](eval_ws/results/official_results_all.md) (H100-only: [official_results_h100.md](eval_h100/official_results_h100.md)).

Epoch-4 action-step sweep on ws (the model always predicts its 32-action chunk; n = steps executed per query): n 8 42.17%, n 12 (official) 46.25%, n 16 45.08%, n 24 42.08%, n 32 43.08% ([sweep_epoch04_report.md](eval_ws/results/sweep_epoch04/sweep_epoch04_report.md)).

**Independent check** ([independent_check.json](independent_check.json)): PASSED - per-episode successes == summary.json == the published tables, for all 9 runs.

## Contents

- `eval_h100/`: H100 official epoch 4/5 runs (per-task JSON, logs, validation), HF upload receipts and logs, render check, smoke, and the aborted time-sliced first attempt (`aborted_20261103_timeslice/`, superseded, not a result).
- `eval_ws/`: ws official epoch 3/4/5 runs and the sweep (`results/`), driver/render logs (`logs/`), the ws eval and validation scripts (`tools/`), and `media_manifest.tsv` listing the 8,400 rollout videos (+3 render .npz) kept on ws.
- `training/`: production manifest, precision, timing, COMPLETE, `metrics.jsonl.gz`, `production.log.gz`, gates, timing campaigns, smoke and reload/resume checks. `setup/`: eval checks a/b/c, benchmarks, setup logs, environment.

Copied 2026-11-06 20:50 UTC (H100) / 20:52 UTC (ws) without modifying any source. `SOURCES.tsv` lists every stored file with its source host/path and the SHA-256 and size of the original bytes (also for `.gz` files and tar members). Not copied: model checkpoints (`*.pt`), rollout videos, copied EGL libraries and latent caches.
