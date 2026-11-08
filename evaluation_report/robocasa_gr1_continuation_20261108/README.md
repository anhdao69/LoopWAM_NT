# RoboCasa-GR1 v0: epochs 6-10 continuation, final results

**Status: complete.** Epochs 6-10 trained from 2026-11-06 13:49 UTC to 2026-11-08 10:24 UTC: 117,580 updates (117,581-235,160), one attempt per epoch, no restart, 141,904 s (39.4 h) of training. Every epoch was evaluated with the official protocol while training paused (75-77 min per evaluation); the last evaluation finished 2026-11-08 11:41 UTC. This folder supersedes the in-progress snapshot [robocasa_gr1_continuation_20261106](../robocasa_gr1_continuation_20261106/README.md), which is kept unchanged.

Continuation of the 5-epoch run in [robocasa_gr1_v0_20261101](../robocasa_gr1_v0_20261101/README.md) from its final training state `latest.pt` (update 117,580, sha256 `e9fa70080709...`): AdamW lr 3e-5 (backbone and robot I/O), no warmup, cosine to 1% (3e-7) at update 235,160; otherwise the same contract (same data and normalization, global batch 256, 23,516 updates per epoch, seed 42). 4 H100 (job 5758, GPUs 4-7). Code: branch `robocasa_gr1_continuation_20261106` (= server branch `gr1_cont_run` @b748d92).

Protocol (same as the v0 evaluations): starVLA `examples/simBenchmarks/Robocasa_tabletop`, 24 tasks x 50 rollouts, any-step success, 720 steps, n_action_steps 12, n_envs 1, seed 0, EGL rendering; H100 GPUs 4-7, 2 policy servers per GPU, 24 parallel clients. References from that README @13d8ad4: StarVLA-OFT-Qwen3 48.8, GR00T-N1.6 47.6.

## Results

| epoch | update | checkpoint sha256 | Hugging Face `anhdao69/LoopWAM_NT` | official success (H100) |
|---|---|---|---|---|
| 6 | 141,096 | `0910b8406af4` | not published | 47.33% (568/1200) |
| 7 | 164,612 | `28ecabf5d4fb` | not published | 46.50% (558/1200) |
| 8 | 188,128 | `7d9ae9e78738` | not published | 48.50% (582/1200) |
| 9 | 211,644 | `694a679b4c07` | `Robocasa-GR1/epoch_09.pt` | **52.17% (626/1200)** |
| 10 | 235,160 | `cb7b9c196903` | `Robocasa-GR1/epoch_10.pt` | 51.67% (620/1200) |

Per-task table with the H100 epoch 4/5 columns and the references: [official_results_continuation.md](eval/official_results_continuation.md). Full checkpoint hashes: [training/checkpoints_sha256.json](training/checkpoints_sha256.json).

- Best evaluated epoch: **epoch 9, 52.17%**, above both references (48.8, 47.6). Epochs 9 and 10 are 7.6 and 7.1 points above epoch 5 on the same host (44.58%, 535/1200); one 1,200-episode run has a standard error of about 1.4 points, so the 0.5-point gap between epochs 9 and 10 is within noise.
- Training loss (mean of the last 2,000 updates of each epoch): 0.0290, 0.0277, 0.0266, 0.0255, 0.0251 for epochs 6-10 (`loss` in training/metrics.jsonl.gz).

## All RoboCasa-GR1 v0 official results

| epoch | update | host | success | folder |
|---|---|---|---|---|
| 3 | 70,548 | ws (RTX PRO 6000) | 32.67% (392/1200) | [robocasa_gr1_v0_20261101](../robocasa_gr1_v0_20261101/README.md) |
| 4 | 94,064 | H100 | 44.08% (529/1200) | [robocasa_gr1_v0_20261101](../robocasa_gr1_v0_20261101/README.md) |
| 4 | 94,064 | ws (RTX PRO 6000) | 46.25% (555/1200) | [robocasa_gr1_v0_20261101](../robocasa_gr1_v0_20261101/README.md) |
| 5 | 117,580 | H100 | 44.58% (535/1200) | [robocasa_gr1_v0_20261101](../robocasa_gr1_v0_20261101/README.md) |
| 5 | 117,580 | ws (RTX PRO 6000) | 44.50% (534/1200) | [robocasa_gr1_v0_20261101](../robocasa_gr1_v0_20261101/README.md) |
| 6 | 141,096 | H100 | 47.33% (568/1200) | this folder |
| 7 | 164,612 | H100 | 46.50% (558/1200) | this folder |
| 8 | 188,128 | H100 | 48.50% (582/1200) | this folder |
| 9 | 211,644 | H100 | **52.17% (626/1200)** | this folder |
| 10 | 235,160 | H100 | 51.67% (620/1200) | this folder |

The epoch-4 action-step sweep (n 8 / 12 / 16 / 24 / 32) is in [robocasa_gr1_v0_20261101](../robocasa_gr1_v0_20261101/README.md).

## Publication and follow-up

- Hugging Face `anhdao69/LoopWAM_NT`, folder `Robocasa-GR1/`: `epoch_09.pt` and `epoch_10.pt` uploaded 2026-11-08 (commit `ebfa5c43`) with `continuation_manifest.json`, both evaluation validations and README rows; the remote LFS sha256 and size of every file were checked against the local bytes, then receipts were committed. Records: [hf/](hf/). Epochs 6-8 are not published.
- Teacher of the GR1-E0 elastic distillation: epoch 10 only if its success count is strictly greater than epoch 9's; 620 <= 626, so **epoch 9** was selected ([followup_gr1_e0/teacher_selection.json](followup_gr1_e0/teacher_selection.json)). GR1-E0 (5 epochs, GPUs 4-7) started 2026-11-08 11:50 UTC and is in progress; it has no result in this snapshot.

**Independent check** ([independent_check.json](independent_check.json), script [../verify_gr1_continuation_20261108.py](../verify_gr1_continuation_20261108.py)): for each of epochs 6-10, per-episode successes == summary.json == validation JSON == per-task table == orchestrator log == the table above; the evaluation config and the policy server name the right checkpoint and step; the table's epoch 4/5 columns match the v0 per-episode evidence; training metrics cover every update once; the teacher selection and the Hugging Face receipts match.

## Contents

- `record/`: README, final STATUS, orchestrator and keeper logs, gates (unit tests, dry run, continuation and resume gates, eval smoke), tools, and logs (training log of each epoch, evaluation and validation logs).
- `training/`: manifest and contract, COMPLETE, timing, precision, resume audits and manifests at every epoch start, data manifest and normalization, `metrics.jsonl.gz` (one row per update), `checkpoints_sha256.json`.
- `eval/`: `official_epoch06` ... `official_epoch10` (per-task episode JSON, summary, config, driver and server logs), their validations, `official_results_continuation.md`, and `media_manifest.tsv` listing the 6,000 rollout videos (32.9 GB) kept on the H100.
- `hf/`: Hugging Face upload receipts, upload and dry-run logs, README draft, and the upload scripts.
- `followup_gr1_e0/`: the GR1-E0 teacher selection record.

Copied 2026-11-08 16:44 UTC from the H100 without modifying any source. `SOURCES.tsv` lists every stored file with its source host/path and the SHA-256 and size of the original bytes (also for the `.gz` file). Not copied: model checkpoints (`*.pt`; hashes in `training/checkpoints_sha256.json`), rollout videos and latent caches.
