# Dense-S12 epoch 10 — RoboTwin-10 evaluation

Updated: 2026-11-09T07:03:28+07:00 (Vietnam time).

## Status and scope

The detached queue has started on the idle local NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB). Checkpoint, code, renderer, CUDA-graph and all 16 smoke-episode checks passed. The scored evaluator is running with four workers; actual policy queries and 9 completed scored episodes were observed at 2026-11-09T07:07:06+07:00. See `queue_status.json` for current stage and `report.md` / `progress.json` for scored progress once available. Queue launch does not mean benchmark completion.

This deployment is **Dense-S12 epoch 10 only**, ten tasks, both settings. No other epoch was downloaded or scheduled. The Elastic E2 evaluation on server 10.148.182.137 remains separate.

## Checkpoint identity

| Field | Value |
|---|---|
| Release | [robotwin-10/dense-s12/epoch_10](https://huggingface.co/anhdao69/LoopWAM_NT/tree/794de7069cf7a05b77808b0b55af0290b5b0bf0f/robotwin-10/dense-s12/epoch_10) |
| Pinned HF revision | `794de7069cf7a05b77808b0b55af0290b5b0bf0f` |
| Epoch / update | **10 / 90,890** |
| Variant / architecture | `dense-s12` / `dense_s12` |
| Structure | 12 stored block pairs; 1 video pass and 1 action pass |
| Policy parameters | 584,567,886 |
| Checkpoint bytes | 2,338,752,101 |
| policy.pt SHA256 | `a222704c10688bf42a531a4a4175fe45a1b172c437cf9b9efa435c4b4fae1ac8` |
| Local checkpoint | `/data14t/anhdh35/loopwam_robotwin10_dense_s12_epoch10_20261109/checkpoint/policy.pt` |

Eleven release files were downloaded through server 10.148.182.137 using verified HTTPS, transferred locally, and independently rechecked. HF LFS/Git blob checksums and SHA256SUMS were verified. The manifest contains only the epoch_10 prefix. Receipts preserve relay and final local locations. Published metadata describes continuation from epoch 5 through epoch 10; only the epoch-10 export is loaded here.

## Evaluation protocol

| Item | Value |
|---|---|
| Tasks | adjust_bottle, grab_roller, hanging_mug, move_stapler_pad, open_microwave, press_stapler, scan_object, stack_blocks_two, stamp_seal, turn_switch |
| Conditions | `demo_clean`, `demo_randomized` |
| Episodes | 100 accepted episodes per task and condition; **2,000 scored episodes** |
| Evaluation seed | **42 only** |
| Inference | **NFE 10**, CFG 1 |
| Action prediction / execution | **32 predicted / 24 executed** before replanning |
| Actions | 14D absolute joint positions, original z-score normalization |
| Precision | FP32 weights, BF16 compute |
| Cameras | Released FastWAM PIL uint8 resizing; packed 384 × 320 input |
| History / memory tokens | 0 / 0 |
| Instructions | Unseen templates, released evaluator selection behavior |
| Query noise | CPU action noise; fixed seed 42 reset per query |
| Scene admission | Original expert-planning/success checks and task horizons |
| Renderer | RT, 32 samples per pixel, depth 8, OIDN 2.3.3 |
| Saved evidence | Videos plus episode/query/action/scene/instruction records |

Python instruction-template selection retains the released evaluator's unseeded behavior. Matching evaluation seed and protocol therefore does **not** guarantee identical instruction selections or exactly paired trajectories across independently launched runs. Results are not replaced task-by-task based on success rate.

## Fair comparison and source verification

The validated Dense-S12 epoch-5 run is the reference. Tasks, conditions, episode count, seed, NFE, prediction/execution length, image preprocessing, instruction behavior, renderer, VAE and normalization were checked against it. Architecture and normalization match. That run completed at **1,444 / 2,000 = 72.20%** (clean 71.90%, randomized 72.50%). The epoch-10 result must finish and pass its final audit before a full comparison.

Evaluation checkout: `/storage/anhdh35/LoopWAM-robotwin10-dense-eval`, commit `7a2181d0ecc8e30d64806e94535b347bc9e1da2e`. Shared native model/evaluator source was not changed for this launch. Eight native model/data source files match checkpoint provenance. The continuation-training script differs and is not imported by evaluation.

The deployment verifies 21,152 inherited runtime file/asset entries and explicitly pins **781 baseline source files**, including evaluator, CUDA-graph helper, task and policy code, into the runtime manifest. The policy link is checked against this checkout. The controller verifies the frozen files before scoring and the finalizer checks them again. Independent review identified a missing explicit baseline source binding; it was corrected and reviewed before launch. No actual source drift was found.

Canonical normalization hash: `38ed9c8d466356fb8f4688da470b14a036dbc51e0fc5484cc5959b507adbcd48`.
VAE SHA256: `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`.

## Speed and checks

The queue uses **four concurrent workers**, shared T5 resources and CUDA graphs. Four workers completed previous sustained runs reliably. Short higher-concurrency profiles underestimated later GPU-memory/renderer failures; this deployment therefore fixes concurrency at four. Smoke monitoring requires at least 10,000 MiB memory margin and stops below 8,000 MiB free. Scoring waits for the measured memory admission threshold.

- **38 native model/evaluator tests passed**.
- **8 launcher/source-binding tests passed**, including source-drift and invalid cached-concurrency rejection.
- **2 renderer-guard tests passed**.
- Strict CPU checkpoint loading passed; finite weights, epoch, update, architecture, parameter count, normalization and VAE were verified.
- CUDA-graph output equivalence passed on **eight recorded real inputs**, with **zero maximum absolute difference** and zero tolerance. Gate seeds 42/43/44 are validation cases; scored evaluation remains seed 42 only.
- Measured native query: **73.87 ms**; graph query: **60.88 ms**; **17.6% lower query latency** in this gate. This is not an end-to-end simulation speed measurement.

The completed smoke run consists of **16 unscored episodes**: adjust_bottle, grab_roller, press_stapler and scan_object; both conditions; two episodes per cell. All eight smoke task/setting cells completed with no infrastructure failures in 223.9 seconds. Peak measured GPU memory was 49,973 MiB, leaving 47,914 MiB margin by total-minus-peak accounting. Smoke records remain separate from the 2,000 scored episodes. Partial success rates are provisional until the final audit passes.

## Automatic queue and report files

Deployment: `/data14t/anhdh35/loopwam_robotwin10_dense_s12_epoch10_20261109`.
Reports: `/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_dense_s12_epoch10_20261109`.
Queue PID: 1574568, with process start ticks recorded in `queue_launch.json`.

Stages: runtime verification → renderer tests → graph gate → smoke → scored evaluation → final audit. The detached process survives terminal disconnection; machine-reboot restart is not configured. Each task/condition cell allows up to three attempts with a 1,800-second no-progress timeout. Failed/interrupted attempts are retained for diagnosis and excluded from valid final totals. Gate failure stops scoring and records the error.

`queue_status.json` records stage; `progress.json` and `report.md` update during scoring. `launch.json` records exact native command/process. `runtime_manifest.json`, `source_check.json`, `cpu_validation.json`, `graph_gate.json`, `calibration.json`, `preflight.json`, review receipts and test logs preserve validation evidence. On completion, the finalizer produces `final_report.md`, `final_results.json` and CSV outputs after checking 20 completed cells and 2,000 episode/video records. Reports update directly into the requested repository folder through the deployment's `report` symlink.
