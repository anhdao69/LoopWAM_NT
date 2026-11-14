# v0-video2-action2 epoch10 — evaluation setup

Prepared 2026-11-13T03:12:03.490111+07:00.

Checkpoint: https://huggingface.co/anhdao69/LoopWAM_NT/tree/b72f355505d98c473fb91e45d10e16d4cc7ce536/robotwin-10/v0-video2-action2/epoch_10

Policy SHA256: `89a87a875e92a9e19ab255609116aca3a86a31f654d93b88a5dd63c630a09786`. File size2,338,752,101bytes. Strict native variant `v0-video2-action2`, version `v0`, video/action loops2/2. Epoch10, update90,890;12 stored blocks per expert;584,567,886 policy parameters.

Training parent checkpoint SHA256: `9b6b7c1375cfee3888746bbe67897f2a4cf6716457abd2a9c485e5a4e09c2d51`. Phase starts at epoch5/update45,445. Training provenance is preserved from the release and is not assumed to identify a previously evaluated epoch5 export.

Run directory: `/data14t/anhdh35/loopwam_robotwin10_epoch10_queue_20261113/runs/v0_2x2`. Source copy: `/data14t/anhdh35/loopwam_robotwin10_epoch10_queue_20261113/runs/v0_2x2/repo`.

## Evaluation protocol

Seed 42; 10 tasks; demo_clean + demo_randomized; 100 accepted episodes per task/setting (2,000 scored episodes/model); NFE 10; CFG 1; predict 32 actions, execute 24 before replanning; 14-dimensional actions/proprioception; FP32 parameters and BF16 autocast.

Tasks: `adjust_bottle`, `grab_roller`, `hanging_mug`, `move_stapler_pad`, `open_microwave`, `press_stapler`, `scan_object`, `stack_blocks_two`, `stamp_seal`, `turn_switch`.

The native evaluator, deployed policy, task admission, task horizons, image handling, action normalization, scene configuration and rendering are copied from the previous successful local evaluation. Source commit is `7a2181d0ecc8e30d64806e94535b347bc9e1da2e`; dirty/frozen evaluator changes are bound by SHA256, so the commit alone is not the full provenance. Each run has an isolated source copy and a fixed `source_snapshot.json`. No inference/model source is modified for this campaign. The runtime also verifies the inherited environment and shared assets.

Both completed epoch10 baselines (v0 4/4 and v0 4/1) are checked for matching protocol fields and policy/harness hashes. Each checkpoint has the same parsed normalization and exact task list. The training export identifies its training source as dirty; the eight recorded model/data source hashes match the frozen evaluation source. Its continuation training script differs from the frozen training script and is not imported by evaluation.

Images follow released FastWAM PIL/uint8/BF16 preprocessing. Query noise is generated on CPU with a fixed seed 42 for every policy query. Instructions use the released unseen instruction mechanism, including unseeded Python template sampling. Thus repeated seed-42 runs match the evaluation protocol but are not guaranteed to use identical instructions or exactly paired trajectories. Actual accepted scene seeds, instructions, action-trace hashes and video hashes are retained per episode.

Rendering remains ray-traced RT32, depth 8, with OIDN 2.3.3 for Blackwell compatibility, matching the previous local runs. Expert scene admission and task horizons remain unchanged. Environment admission failures do not become scored policy failures. Interrupted worker attempts are retained, excluded from official scoring, and listed in the final audit.

## Speed and queue behavior

Only one model evaluates at a time on local GPU 0 (RTX PRO 6000 Blackwell Max-Q). Each uses four simulator workers and one shared T5 service. CUDA graphs are admitted only after bit-exact comparison against native inference on eight captured real observations, checked in forward and reverse order. This gate runs NFE10 with validation-only seeds42/43/44; official evaluation is seed42 only. No precision reduction, shorter horizon, extra early success shortcut, or skipped task is used for speed.

Each model runs 16 unscored smoke episodes: adjust_bottle, grab_roller, press_stapler, scan_object × clean/randomized × 2. Admission requires at least 60,000 MiB free before GPU work and at least 10,000 MiB observed smoke memory margin. Higher worker counts previously caused sustained-run renderer/resource failures, so four workers is the proven configuration. Smoke time is not treated as an accurate full-benchmark ETA.

The campaign and each model use independent file locks. Inherited ownership tokens and PID-start identities permit cleanup after a supervisor crashes; a real CPU-process regression verifies cleanup of a separately sessioned descendant while preserving unrelated processes. The queue advances only after a successful final audit with 20 task/settings and 2,000 verified scored videos/episode traces for the expected checkpoint. A failed model stops the campaign and publishes the error; it does not silently skip to the next model. Native worker failures may retry at most 3 attempts per cell, with a 1,800-second no-progress timeout. Logs and valid completed cells support controlled resumption; gates and runtime hashes are rechecked.

## Records and interpretation

Before scoring: checksum receipt, pinned HF manifest, native tests, CPU validation, source check, immutable runtime manifest, CUDA graph gate, smoke calibration, and preflight. During scoring: queue/launch state, per-worker logs/videos/traces, progress JSON/Markdown. After scoring: final_results.json, task_results.csv, episodes.csv, failed_attempts.json, protocol_comparison.json, final_audit.json, COMPLETE.json and final_report.md. Partial SR includes currently logged episodes and is provisional; the official final result requires complete audited cells.

Dense-S12 epoch10 already has a prior result (1,517/2,000 =75.85%, clean75.40%, randomized76.30%). This campaign is a fresh rerun in a separate directory. Its unchanged policy file is reused only after matching the newly pinned HF LFS SHA256; prior results remain intact. The previous v0 4/4 epoch10 result is80.15% (clean79.50%, randomized80.80%); v0 4/1 epoch10 is80.95% (clean80.00%, randomized81.90%). No previous, smoke, failed, or seed43 result is blended into the new score.

## Time estimate

Previous four-worker evaluation durations: v0 4/2 about 3.21 hours, v0 2/2 about 3.35 hours, Dense-S12 epoch10 about 3.26 hours on this local machine; v0 1/4 about 3.53 hours on the comparable remote RTX PRO6000. These are historical scoring durations, not guarantees. Budget roughly 3–4 hours/model and 14–16 hours for the full sequence including gates and audits, assuming no competing GPU job, retries, or hardware/network interruption. ETA will be revised from actual scored progress.

## Locations

- Campaign: `/data14t/anhdh35/loopwam_robotwin10_epoch10_queue_20261113`
- Reports: `/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_epoch10_queue_20261113`
- Live queue: [progress.md](progress.md) and [status.json](status.json)
