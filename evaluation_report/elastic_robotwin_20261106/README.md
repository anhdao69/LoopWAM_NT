# RoboTwin-50 elastic distillation (RT50 E0/E1/E2)

**Status at snapshot: in progress.** Main report: [REPORT.md](REPORT.md) (generated 2026-11-06 20:42 UTC).

The elastic E0/E1/E2 recipes ported to RoboTwin-50 (endpoint loss over the 24 executed actions, no binarized gripper loss). Two parts:

1. Training-speed test with the RoboTwin v0 epoch-10 teacher on 8 and 4 H100 (`timing/`, `validation/`; previous report version [REPORT.v1_timing.md](REPORT.v1_timing.md)).
2. One-epoch E0 -> E1 -> E2 queue distilled from the RoboTwin v0 **epoch-5** teacher (HF `robotwin/v0-video4-action4/policy.pt`) on 4 H100 (`ep5/`): pre-launch validation in `ep5/prelaunch/`, live status in `ep5/STATUS.json`.

Progress at snapshot: RT50-ep5-E0: update **18,323 / 47,462** (38.6%), started 2026-11-06 15:39 UTC, 1 launch(es), 0 resume(s). No task evaluation of the distilled students has been run yet. Code: branch `elastic_robotwin` (production run at commit 2c78ff6). Nothing is published to Hugging Face.

`tools/make_report.py` regenerates REPORT.md on the H100 login node from the run files. The running E0 `metrics.jsonl` is not copied while live; its train log is a snapshot.

Copied 2026-11-06 20:50 UTC (H100) / 20:52 UTC (ws) without modifying any source. `SOURCES.tsv` lists every stored file with its source host/path and the SHA-256 and size of the original bytes (also for `.gz` files and tar members). Not copied: model checkpoints (`*.pt`), rollout videos, copied EGL libraries and latent caches.
