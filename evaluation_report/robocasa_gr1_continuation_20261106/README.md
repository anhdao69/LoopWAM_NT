# RoboCasa-GR1 v0: epochs 6-10 continuation with per-epoch official evaluation

**Status at snapshot: in progress** - phase `training` (epoch 6 (to update 141096), attempt 1 fresh continuation); update 138,331 at 2026-11-06 20:46 UTC, 1.19 s/update; epoch 6 ends at update 141,096 (~2026-11-06 21:41 UTC), then its official evaluation runs. No epoch 6-10 result exists yet; nothing is published to Hugging Face.

Continuation of the 5-epoch run in [robocasa_gr1_v0_20261101](../robocasa_gr1_v0_20261101/README.md) from its final `latest.pt` (update 117,580), lr cosine 3e-5 -> 3e-7 without warmup, 4 H100 (job 5758 GPUs 4-7); training pauses at each epoch end for the official 24-task evaluation. Code: branch `robocasa_gr1_continuation_20261106` (= server branch `gr1_cont_run` @b748d92). Full description: [record/README.md](record/README.md).

## Contents

- `record/`: README, STATUS, orchestrator/keeper logs, gates (unit tests, dry run, continuation + resume gates, eval smoke), tools, logs (the running training log is a snapshot).
- `train_snapshot/`: manifest, precision, resume audit, timing of the running attempt (metrics.jsonl not copied while live).
- `stopped_attempt_20261106T130635Z/`: the first attempt, stopped at update 117,596 at the user's request for the RoboTwin elastic timing test; superseded by the restart from update 117,580.

Copied 2026-11-06 20:50 UTC (H100) / 20:52 UTC (ws) without modifying any source. `SOURCES.tsv` lists every stored file with its source host/path and the SHA-256 and size of the original bytes (also for `.gz` files and tar members). Not copied: model checkpoints (`*.pt`), rollout videos, copied EGL libraries and latent caches.
