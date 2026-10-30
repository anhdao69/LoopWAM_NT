# LiLa-WAM trained on our LIBERO data (baseline, paper recipe)

These are **not official LiLa-WAM weights.** They are a baseline we trained ourselves.

- **Model and code:** the public LiLa-WAM code (github.com/teee000/LiLa-WAM @ `339860f`; paper arXiv 2608.03701).
- **Data:** the same LIBERO training data our LoopWAM/FastWAM models use.
- **Why we trained it:** the authors released no LIBERO checkpoint and no LIBERO training code; only a RoboTwin 2.0 model is public.
- **Recipe:** the paper's LIBERO recipe, as closely as the paper specifies it.

## Files

| Path | Content |
|---|---|
| `<suite>/stage2_epoch2_final.pt` | **Final policy for the suite**: the end of stage 2. Use this one. |
| `<suite>/stage1_epoch5.pt` | End of stage 1, which initializes stage 2. |
| `<suite>/config_stage{1,2}.yaml` | The exact configs that were trained, saved by `train.py`. |
| `<suite>/norm_stats.json` | Action/state min-max for that suite. The top-level key `robotwin2` is required by LiLa-WAM's loader. |
| `<suite>/logs/` | Full training logs, plus per-update loss CSVs. |
| `task_cond/<suite>/<task>/task_cond.npy` | The task-condition vector (1024-D) for each of the 40 tasks. |
| `code/` | Data converter, LIBERO dataset adapter, training wrapper, stats and task-condition scripts, base config, launch script. |
| `manifest.json` | sha256, size, epochs and final loss of every uploaded checkpoint, filled in per suite as each upload is verified. |

**Checkpoint format.** Each `.pt` is the dict that `train.py` saves: `epoch`, `global_step`, `model_state_dict` (the 212.1M-parameter action model, bf16), `optimizer_state_dict`, `scheduler_state_dict` and `loss`. The frozen DINOv3 ViT-L/16 encoder is **not** included. Get it from `facebook/dinov3-vitl16-pretrain-lvd1689m` (DINOv3 license).

## Training data

The data is our LIBERO set:
- the FastWAM LIBERO regeneration with MuJoCo 3.3.2 and no-op actions removed, in LeRobot v2.1 format;
- 1,712 demonstrations and 277,713 frames over 40 tasks;
- per suite: libero_10 388 demos / 104,280 frames, goal 433 / 52,895, object 457 / 67,309, spatial 434 / 53,229.

It was converted to the authors' LIBERO HDF5 layout (`code/convert_ours_to_lilawam_hdf5.py`) as follows:
- **Images:** the 512×512 agent-view and wrist frames, resized to 256×256 (INTER_AREA) and stored as PNG. The frames are upright, the same orientation as our LeRobot data.
- **Action:** 6 end-effector deltas, unchanged, plus the gripper converted to raw LIBERO (−1 open, +1 close; = 1 − 2·ours). A policy therefore outputs raw environment actions; do **not** apply FastWAM's gripper flip.
- **Proprio:** our 8-D `observation.state` (eef position 3, axis-angle 3, gripper qpos 2).
- **Cleaning:** none. Every frame of every demonstration is used.

## Recipe

| Item | Paper (Sec. 4.1–4.2) | This training |
|---|---|---|
| Policies | one per suite | one per suite |
| Views | third-person + wrist, 256×256 | the same; side by side as one 512×256 image `[agentview \| wrist]` (fusion not described in the paper; the public model takes one image) |
| Action chunk | 12 | 12 |
| Vision encoder | frozen DINOv3 ViT-L/16 | the same (layers −12/−8/−4 + CLS/registers, public config) |
| Optimizer | AdamW (0.9, 0.99), wd 0.01 | the same |
| Batch, clipping | 128, 1.0 | 128, 1.0 |
| Loss | flow matching + λff=0.5 future-feature loss, logit-normal t | the same |
| LR schedule | stage-wise cosine, peak 2e-4 then 5e-5 | stage 1: cosine 2e-4→5e-5; stage 2 (`--init_from`, fresh optimizer and schedule): cosine 5e-5→5e-6 |
| Epochs | not stated | stage 1: 5 epochs, stage 2: 2 epochs, per suite |
| Task condition | task vector from DINOv3 | the same: last-layer CLS of the last frame minus the first frame, averaged over a task's demonstrations, on the same composite |

Each run is the public `train.py`, unchanged except that its dataset factory reads LIBERO (`code/train_libero.py`, `code/libero_data.py`). It ran on one H100 80GB at 0.61 s/update, in interactive job 1094056.

## Inference inputs (must match training)

- **Image:** raw LIBERO observations are upside down, so flip both views with `[::-1, ::-1]`. Resize each to 256×256 (INTER_AREA), then join them with `code/libero_data.composite(agent, wrist)`. Feed RGB with ImageNet normalization.
- **State:** the 8-D state as above. Use the suite's `norm_stats.json` and the task's `task_cond.npy`. Run 10 Euler flow steps (public config).

## Status

- **LIBERO success rates under our matched protocol are pending:** 4 suites × 10 tasks × 50 trials × seeds 42/43/44.
- **Paper reference:** the paper reports 98.0 / 98.8 / 97.2 / 94.2 (Spatial / Object / Goal / Long), 97.1 average. Those come from the authors' own data and training length, not from these weights.
- **License:** the LiLa-WAM ModelScope release states Apache-2.0. DINOv3 is under Meta's DINOv3 license and is not redistributed here.

## Results so far (generated from the result files)

### Training

| Suite | Stage-1 epoch mean loss (5 epochs) | Stage-2 epoch mean loss (2 epochs) | Final file | sha256 (HF LFS verified) |
|---|---|---|---|---|
| Long (libero_10) | 0.2737 → 0.1731 → 0.1546 → 0.1430 → 0.1371 | 0.1348 → 0.1318 | `libero_10/stage2_epoch2_final.pt` | `27ad99a458d97a68…` (match) |
| Goal | 0.3463 → 0.1971 → 0.1710 → 0.1591 → 0.1520 | 0.1484 → 0.1456 | `libero_goal/stage2_epoch2_final.pt` | `5ac2bfd3aad9c2ca…` (match) |
| Object | 0.2800 → 0.1623 → 0.1444 → 0.1352 → 0.1299 | 0.1277 → 0.1247 | `libero_object/stage2_epoch2_final.pt` | `4571a5f45f02dc61…` (match) |
| Spatial | 0.3419 → 0.1951 → 0.1742 → 0.1626 → 0.1555 | 0.1528 → 0.1503 | `libero_spatial/stage2_epoch2_final.pt` | `0836ecc4ceefb7fe…` (match) |

### Open-loop check on held-in demonstration frames (not a success rate)

512 random training samples per suite, seed 0, public sampling procedure (10 Euler steps). MAE in raw action units over the 6 end-effector deltas; the baseline always predicts the suite's mean action.

| Suite | Checkpoint | MAE model | MAE baseline | Gripper sign agreement model / baseline |
|---|---|---|---|---|
| Long (libero_10) | stage-1 epoch 5 | 0.0729 | 0.1452 | 0.937 / 0.534 |
| Long (libero_10) | final | 0.0702 | 0.1452 | 0.949 / 0.534 |
| Goal | stage-1 epoch 5 | 0.0889 | 0.1845 | 0.948 / 0.633 |
| Goal | final | 0.0853 | 0.1845 | 0.956 / 0.633 |
| Object | stage-1 epoch 5 | 0.0709 | 0.1600 | 0.939 / 0.537 |
| Object | final | 0.0682 | 0.1600 | 0.939 / 0.537 |
| Spatial | stage-1 epoch 5 | 0.0885 | 0.1939 | 0.949 / 0.536 |
| Spatial | final | 0.0855 | 0.1939 | 0.950 / 0.536 |

### Latency

Batch 1 on an H100 80GB (evc103, GPU otherwise idle), 500 recorded raw LIBERO observations after 20 warmup queries, from raw observation to the smoothed host action chunk (12×7): mean 52.0 ms, p50 51.1, p95 61.0, peak 1.12 GB; 0.515B parameters including the frozen DINOv3 (212.1M trainable action model).

### LIBERO success under our matched protocol

Protocol lightwam_20261009_matched: 4 suites × 10 tasks × official initial states 0–49 × seeds 42/43/44; 30 settling steps; at most 400 policy steps (700 for Long); success = environment done flag. Each suite uses its own final policy. 6 of 12 predicted actions are executed per query, and the gripper is executed as sign(g). Inference is batched, in bf16.

| Suite | Successes / episodes | Success % | Seeds 42 / 43 / 44 % |
|---|---|---|---|
| Spatial | 554 / 1500 | 36.93 | 9.6 / 68.6 / 32.6 |
| Object | 293 / 1500 | 19.53 | 18.6 / 21.6 / 18.4 |
| Goal | 495 / 1500 | 33.00 | 11.0 / 60.8 / 27.2 |
| Long (libero_10) | 27 / 1500 | 1.80 | 0.2 / 5.2 / 0.0 |
| **All** | 1369 / 6000 | **22.82** | 9.8 / 39.1 / 19.6 |