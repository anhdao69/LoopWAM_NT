# dense-s12 RoboTwin-10 evaluation — 2026-11-05

Host: local RTX PRO 6000 Blackwell Max-Q, 96 GB. Idle GPU verified before setup. Separate run directory: `/data14t/anhdh35/loopwam_robotwin10_dense_s12_20261105`.

## Release and architecture

HF repository `anhdao69/LoopWAM_NT`, directory `robotwin-10/dense-s12`, pinned revision `dbb527e8f83fbf00fcbbc017f886122d5f4ec4df`. Checkpoint SHA256: `c0c0fbb7b4cc607c00f7ee783c1f0e43815daa9fa09b1141c0b456dec255d064`. Epoch 5, update 45,445; 584,567,886 policy parameters; 12 stored block pairs; native video/action loops 1/1; architecture version `dense_s12`. Dense-s12 is a dense 12-block model, without recurrent looping. All nine training source hashes match the existing frozen evaluator at commit 7a2181d0ecc8e30d64806e94535b347bc9e1da2e. No model/harness source changes are required.

## Scored protocol

Seed 42, NFE 10. All ten release tasks, clean and randomized, 100 accepted episodes per task/setting: **2,000 scored episodes**. Unseen instructions, CFG 1, predict 32 / execute 24, absolute 14-D qpos, original PIL uint8 three-camera packing at 384×320, checkpoint global z-score normalization, FP32 weights / BF16 compute. CPU action noise is reset to seed 42 for every query. Original task horizons and expert scene-plan/success acceptance. Rendering RT 32 / depth 8 / OIDN 2.3.3. Candidate scene seeds begin at 4,300,000 before expert-rejected scenes are skipped.

Tasks: adjust_bottle, grab_roller, hanging_mug, move_stapler_pad, open_microwave, press_stapler, scan_object, stack_blocks_two, stamp_seal, turn_switch.

The released instruction-template Python RNG remains unseeded. Same-seed runs do not guarantee identical instruction strings. Actual scene seeds, instructions, actions and videos are logged. No seed mixing, score-based checkpoint selection or best-run substitution.

## Verification and execution

Verify all release files against pinned HF LFS/Git blob hashes and SHA256SUMS. Strict CPU loading checks native architecture, finite weights, parameter count, training state, normalization and VAE. Run 38 existing evaluator tests freshly on this host. Hash-bind source/runtime assets. Check native versus CUDA-graph actions exactly on eight real observations, seeds 42/43/44 and forward/reverse order.

After the user requested maximum speed, the small initial sweep was superseded before scoring. Expanded unscored profiles: native four workers, then graph four/eight/twelve/ten workers; all ten tasks, both settings, one accepted episode per cell (20 episodes per profile, up to 100 total). Select the fastest passing graph profile by wall time per episode, with at least 12,000 MiB measured memory headroom. Eight-worker admission requires at least 85,000 MiB free. Ten/twelve-worker admission uses measured four/eight-worker peak growth plus 15% incremental-growth allowance and 10,000 MiB reserve; skip profiles that cannot fit. The first native four-worker and graph four-worker profiles are mandatory correctness gates. Eleven additional safety/selection tests passed on each host. Full evaluation requires three stable free-memory samples and rechecks runtime hashes. One shared T5 server; no changes to task horizons or NFE for speed. Calibration episodes do not enter SR.

Persistent tmux queue survives disconnects (not reboot), maximum three attempts per cell, 1,800-second no-progress timeout. Report failed attempts and exclude their partial episodes. After all 20 cells finish, automatically audit the checkpoint, source/runtime hashes, counters and all 2,000 videos, then publish final JSON, CSV and Markdown reports. Prior model results remain separate.

Temporary/cache and bulk output paths use the spacious data disk. Existing environment and assets are reused. Remote portability changes from the prior setup are limited to absolute asset paths; provenance and hashes are checked again. Cross-host timing is reported as observed, not assumed identical.

[Live report](report.md) · [Queue status](queue_status.json) · [Final audit](final_audit.json) when complete.

## Speed-request provenance

The initial preflight was stopped intentionally before any scored episode. Earlier calibration files and runtime bindings are preserved under `preflight_before_speed_request`; a new runtime manifest binds the expanded sweep and its helper. No checkpoint, model, simulator, render quality, NFE or scored task/episode setting changed. An interrupted calibration is not a scored failure. The queues were restarted after syntax checks and eleven targeted memory/selection tests passed. All profile timings include startup and simulator execution; they are throughput evidence, not isolated model latency or guaranteed full-run duration.

The expanded sweep received an independent read-only review. Two pre-scoring edge cases were fixed: unfinished sweeps now fail closed instead of retiming cached/partial episodes, and profile selection requires both 12,000 MiB observed headroom and enough current free memory for final admission. Interrupted pre-review profiles are preserved under `preflight_before_review_fixes`. All 11 focused regression checks pass on both hosts. A completed calibration can resume unchanged; an unfinished sweep must be archived deliberately before restarting.

## Sustained-run failure and four-worker recovery

The selected ten-worker profile passed a short one-episode-per-cell calibration but proved unsafe for sustained 100-episode cells. Full-run logs showed GPU out-of-memory, OIDN filter/resource errors, Vulkan device errors, and retry exhaustion. Some completed cells also contained OIDN memory/filter errors, making image integrity uncertain. Consequently **the entire ten-worker scored attempt is invalidated and excluded**, including completed cells; no episodes are carried forward.

All original results, videos, launch timing, runtime manifest, calibration selection and failures are retained under `invalid_10worker_run` in the bulk directory. A corresponding report snapshot and incident JSON preserve the evidence. The fresh run uses four CUDA-graph workers, the same checkpoint, seed 42, NFE 10, tasks, renderer settings, and 100 episodes per cell. The four-worker all-task calibration had no renderer/resource errors and matches concurrency used by previous successful full runs. This operational override replaces the short-profile speed choice.

The final auditor now additionally rejects any accepted completed-cell log containing OIDN errors, CUDA OOM/resource allocation errors, buffer-allocation failures, or Vulkan device-loss errors. Two regression tests (including seven error patterns) passed on both hosts. An actual previously completed but corrupted cell was rejected, while a clean actual cell was accepted. The checkpoint/model/benchmark code is unchanged; the new finalizer hash is bound before restart.

[Recovery incident](recovery_incident.json) · [Invalidated attempt snapshot](invalid_10worker_run/progress.json).
