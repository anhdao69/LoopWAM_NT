# RoboTwin-10 v0 4/1 — epoch 10

**Current configuration: four workers.** The initial eight-worker short-trial choice was superseded on 2026-11-09 after GPU/renderer memory failures during the full evaluation. Two clean completed cells (156/200 successes) were independently re-audited and retained; all failed/interrupted attempts are excluded and rerun. See [resource recovery](resource_recovery.md).

Requested on 2026-11-08. This is an independent evaluation of the epoch-10 checkpoint under `robotwin-10/v0-video4-action1/epoch_10`, on the free local RTX PRO 6000 Blackwell 96 GB GPU. The existing 4/4 epoch-10 evaluation continues on server137. Previous outputs remain intact.

## Checkpoint identity

- [Pinned release](https://huggingface.co/anhdao69/LoopWAM_NT/tree/539a5a26d01054262220bd33236b0d3a03a91890/robotwin-10/v0-video4-action1/epoch_10), repository revision `539a5a26d01054262220bd33236b0d3a03a91890`.
- Policy SHA-256: `2407441a5f15bc19c5327cf79454d96cc8097b15a194b2ac0c0a50e7b3404af8`; 2,338,752,101 bytes.
- Epoch 10; update 90,890; 11,633,510 training-window presentations. This uses the nested epoch-10 release, not the parent directory's earlier checkpoint.
- Native `loopwam-s-robot-v1`, variant `v0-video4-action1`: **four video loops, one action loop**, aligned/late KV, partition 3/6/3, 12 stored blocks per expert, 584,567,886 policy parameters, no history or memory tokens, 14-D action/proprioception.
- All 11 release files downloaded through server137 over verified HTTPS, verified against pinned HF git/LFS metadata and the published SHA256SUMS, transferred over SSH, and rehashed on the evaluation host. The local download receipt retains remote relay provenance.
- CPU strict load confirms the exact variant, loop counts, training step/cursor/contract, finite FP32 policy weights, normalization and VAE. No optimizer payload is used for inference.

## Benchmark protocol

| Setting | Value |
| --- | --- |
| Tasks | adjust_bottle, grab_roller, hanging_mug, move_stapler_pad, open_microwave, press_stapler, scan_object, stack_blocks_two, stamp_seal, turn_switch |
| Environments | demo_clean and demo_randomized |
| Episodes | 100 accepted episodes/task/setting; 20 cells; 2,000 scored episodes |
| Seed | 42; original fixed CPU noise seed for every policy query |
| Diffusion | NFE 10, CFG 1.0 |
| Action horizon | Predict 32, execute 24, then replan |
| Action type | Absolute 14-D joint positions |
| Precision | FP32 policy weights, BF16 compute |
| Images | Original released FastWAM PIL/uint8 camera packing, 384×320, BF16 normalization |
| Instructions | Unseen instruction pool; original Python instruction-template RNG remains unseeded |
| Scene admission/horizons | Unchanged original expert-plan/success filter and task horizons |
| Rendering | RT, 32 samples, depth 8, OIDN 2.3.3 |
| Retries | Maximum three attempts/cell, 1,800-second no-progress timeout |

The one action **loop** is an internal model setting. It does not change the 32/24 prediction/execution horizon. RoboCasa's separate execution prefix of 12 does not apply to this RoboTwin run.

The native evaluator is the unchanged `7a2181d0ecc8e30d64806e94535b347bc9e1da2e` worktree at `/storage/anhdh35/LoopWAM-robotwin10-dense-eval`. Its directory name does not choose the model: the native loader reads the checkpoint's v0 4/1 metadata. All eight released model/data source hashes must match. The continuation-training script differs from the original training script and is not used by evaluation.

The inherited local 4/1 epoch-5 runtime manifest binds 21,150 files (approximately 28.7 GB), including assets, model/runtime dependencies and native evaluator sources. Those bytes are verified before launch. New control scripts, checkpoint metadata and planned protocol are added to the bound runtime. The same-host 4/1 epoch-5 result is the automatic final comparison baseline; epoch and training step differ, while task/seed/inference/environment settings and normalization must match. The separate 4/4 epoch-10 run is available in the adjacent report folder. Unseeded instruction-template generation prevents claiming exactly paired trajectories across independent runs.

## Speed selection and gates

The idle local GPU is used immediately. CUDA graphs are allowed only after eager/graph action outputs match bit-for-bit across eight captured real observations, tested forward and reverse using noise seeds 42/43/44. Those extra seeds are validation inputs only; the scored benchmark uses seed42.

Concurrency is measured using **four versus eight workers**, each running the same 16 unscored episodes: adjust_bottle, grab_roller, press_stapler and scan_object, clean/randomized, two episodes per cell. A shared T5 server avoids per-worker encoder duplication. GPU memory is sampled during each trial; the trial stops if free memory falls below 8,000 MiB. Only fully validated profiles with at least 10,000 MiB observed total-memory headroom are eligible. The fastest eligible profile is selected. Eight workers are skipped if fewer than85,000MiB are free before their trial. An invalid eight-worker trial falls back to a validated four-worker profile. The chosen worker count, measured times and peak memory are written to calibration.json and preflight.json.

Old trials lacking a successful timing receipt are preserved under calibration_abandoned and rerun from scratch; cached completed cells are never timed as fresh inference. Short trials and unseeded instructions limit speed extrapolation. All calibration episodes are excluded from official SR.

## Validation and automatic operation

All 38 native evaluator/model tests passed in 3.94 seconds. Two calibration regression tests, two subprocess cleanup/memory-guard tests and two renderer-error tests passed. Independent review is clear after fixing fresh-trial resume, eight-worker validation fallback, and task-list report sourcing. See queue_review.json.

The persistent supervisor is `scripts/queue_pipeline.py`, launched by `start_queue.sh`, with a lock and recorded process identities. It verifies inputs/runtime, requires strict model loading and graph parity, performs calibration, waits for GPU headroom, launches scored evaluation, then runs the final audit automatically. Graceful stop propagates through the controller, native evaluator, simulator workers and T5 server. Only complete validated cells under the identical manifest are reused. Interrupted attempts and their logs/videos are preserved and excluded from final scoring.

Final audit rechecks all 2,000 episode videos with ffprobe, unique accepted scene seeds, instruction records, ordered action-query traces, cumulative successes, checkpoint/source/runtime hashes, protocol consistency and renderer error markers. Failed and interrupted attempts are separately recorded. Final SR is published only after this audit passes.

Run root: `/data14t/anhdh35/loopwam_robotwin10_v0_4x1_epoch10_20261108`.
Reports are written directly to this evaluation_report folder; no remote synchronization is needed. `queue_status.json` records the stage, and `progress.json` / `report.md` update every ten seconds during scoring. `final_results.json`, `final_report.md`, `task_results.csv`, `episodes.csv`, `final_audit.json` and `protocol_comparison.json` are produced automatically after completion.

## Initial short-trial selection — superseded by four-worker recovery

Both fresh calibration profiles passed all16 episodes and clean renderer/artifact checks. Four workers took 238.04s with peak 50,641MiB; eight took 193.71s with peak 72,518MiB and 25,369MiB total-memory margin. **Eight workers were selected**, reducing this trial's elapsed time by 18.62%. This short calibration does not establish the same speedup for every benchmark task. All32 calibration episodes are excluded from the 2,000-episode scored evaluation.

Graph parity passed with zero absolute/relative tolerance. Mean test-query time was0.08565s eager and0.07460s with CUDA graphs. The persistent supervisor was launched with PID1174420; launch.json records the scored evaluator's identity.
