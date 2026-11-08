# LoopWAM v0 (4/4 loops, 3/6/3) — RoboCasa-GR1 tabletop (24 tasks)

LoopWAM v0 trained on the RoboCasa GR1 tabletop simulation data (starVLA mixture `fourier_gr1_unified_1000`,
24 tasks x 1000 demos, GR1 arms + waist + Fourier hands, 29-D joint actions, ego view
`ego_view_bg_crop_pad_res256_freq20` at 256x256).

| file | content |
|---|---|
| `epoch_03.pt` | end of epoch 3 (update 70,548) |
| `epoch_04.pt` | end of epoch 4 (update 94,064) |
| `epoch_05.pt` | end of epoch 5 (update 117,580) |
| `epoch_09.pt` | continuation epoch 9 (update 211,644); official eval 52.17% (626/1200) (best evaluated epoch) |
| `epoch_10.pt` | continuation epoch 10 (update 235,160); official eval 51.67% (620/1200) |
| `epoch_NN.pt.uploaded.json` | upload receipt (sha256, size, HF revision) |
| `dataset_stats.json` | 29-D min/max normalization used in training (required at inference) |
| `data/data_manifest.json` | training data manifest |
| `manifest.json` | training run manifest and contract (epochs 1-5) |
| `continuation_manifest.json` | continuation run manifest and contract (epochs 6-10) |
| `eval/official_epochNN_validation.json` | official eval validation, per-task successes |

Training contract (same recipe as `robotwin/v0-video4-action4`, global batch 256): 5 epochs over 6,020,058 windows
(117,580 updates), AdamW lr 1e-4 (backbone and robot I/O), betas (0.9, 0.95), weight decay 0.01, clip 1.0, cosine
schedule with 5% warmup and floor 0.01, seed 42, FP32 weights/moments with BF16 compute, 4 x H100.
Inference: 32-action chunk, 10 denoising steps, CFG 1, fixed noise seed 42 per query.

Checkpoints load with `fastwam.models.wan22.loopwam_gr1.create_loopwam_gr1` (branch `robocasa_gr1_v0` of LoopWAM_NT)
and are served to the starVLA RoboCasa-GR1 tabletop evaluation by `experiments/robocasa_gr1/loopwam_policy_server.py`.
Evaluation follows starVLA `examples/simBenchmarks/Robocasa_tabletop` (24 tasks x 50 episodes, n_envs 1,
max_episode_steps 720, n_action_steps 12, state sent).

## Continuation: epochs 6-10

`epoch_09.pt` and `epoch_10.pt` come from a continuation of the same run for five more epochs (6-10), started from the
end-of-epoch-5 training state `latest.pt` (update 117,580, sha256 `e9fa70080709d251ce8e946808ddcfea9df66b007f7389f34d18e4b37a6a1a58`) with a new learning-rate phase:
AdamW lr 3e-5 (backbone and robot I/O), no warmup, cosine to 1% (3e-7) at update 235,160; otherwise the same contract
(same data and normalization, global batch 256, 23,516 updates per epoch, seed 42, 4 x H100). The full continuation
manifest and contract are in `continuation_manifest.json`.

Official evaluation (protocol above, 24 tasks x 50 episodes, H100, EGL rendering, env seed 0): epoch 9 = **52.17%
(626/1200)**, epoch 10 = 51.67% (620/1200). Per-task results: `eval/official_epochNN_validation.json`.
