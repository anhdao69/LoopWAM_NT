# Public WAM baselines on LIBERO: latency of the released models, and LiLa-WAM trained on our data (2026-10-30)

This report covers two pieces of work on H100 80GB HBM3. Branch `elastic`.

1. **Latency of four public world-action models**, each run through **its own public code with its own released weights**, unoptimized: FastWAM, FasterWAM, Light-WAM and LiLa-WAM.
2. **LiLa-WAM trained on our LIBERO data**, because the authors released no LIBERO model. It uses the paper's LIBERO recipe. The weights are on Hugging Face under `baselines/`, along with an open-loop check and the trained model's latency.

**Training: all four suites finished both stages.** LIBERO success of the trained LiLa-WAM under our matched protocol (Section 6.9): **22.82% (1369/6000)**.

Where things are:
- **Data and scripts:** `plans/performance/public_wam_latency_20261030/` (latency) and `plans/performance/lilawam_libero_20261030/` (LiLa-WAM training).
- **Our own models:** compared in `LoopWAM_All_Models_Latency_20261030.md`.

## 1. Summary

- **Latency of the released models, as their own LIBERO eval runs them.** Batch 1, mean over 500 queries, node evc102:
  - Light-WAM: 49.1 ms;
  - LiLa-WAM (RoboTwin release): 49.5 ms;
  - FastWAM: 73.9 ms;
  - FasterWAM: 193.1 ms.
- **With the language prompt encoded once per task** (our matched boundary):
  - Light-WAM: 49.4 ms;
  - FastWAM: 59.3 ms;
  - FasterWAM: 180.8 ms.

  The second node, evc103, agrees within 5% for every model (Section 4).
- **FastWAM's speed comes from `torch.compile`.** Its public LIBERO config turns compile on. With compile off it takes 232.5 ms instead of 59.3 ms (3.9–4.1×).
  - Uncompiled vs uncompiled, FasterWAM is 1.27–1.29× faster than FastWAM (180.8 vs 232.5 ms).
  - FasterWAM's own latency script agrees: 178.5 vs 244.7 ms on random-weight architectures.
- **Text encoding cost.** FastWAM and FasterWAM run the 5.7B-parameter T5-XXL text encoder inside every query, which adds 12–18 ms. Light-WAM encodes once per task. LiLa-WAM has no language input.
- **Prompt caching changes no actions.** Encoding the prompt once per task gave bit-identical actions on 500/500 queries for every model.
- **Peak GPU memory:**
  - FastWAM and FasterWAM: about 25 GB, because they keep T5-XXL resident;
  - Light-WAM: 4.3 GB, because it moves T5 to the CPU after encoding;
  - LiLa-WAM: 1.1 GB (0.52B parameters, no language model).
- **Checkpoints are verified.** All timed checkpoints match their published sha256. The FastWAM file is the exact checkpoint our matched evaluation scored (96.63%).
- **LiLa-WAM has no public LIBERO model**, so we trained one on our LIBERO data with the public code and the paper's LIBERO recipe:
  - one policy per suite;
  - third-person + wrist views at 256×256;
  - chunks of 12 actions;
  - two-stage cosine learning rate, 2e-4 then 5e-5.

  The paper does not give the number of epochs, so 5 + 2 epochs per suite were chosen to fit one H100 in job 1094056.
  - The data conversion was verified exactly: 24 randomly chosen demonstrations (3,876 frames), all checks passed: True.
  - Training progress and losses: Section 6.5. Weights and their sha256 verification: Section 6.6. Open-loop check: Section 6.7.
  - Latency of the trained LIBERO policy: 52.0 ms mean (p50 51.1, p95 61.0), 1.12 GB peak, measured on evc103 after training finished with the GPU otherwise idle.

## 2. What was run

| Model | Public code (commit) | Released weights | Verified |
|---|---|---|---|
| FastWAM | github.com/yuantianyuan01/FastWAM (`7faa711`) | `libero_uncond_2cam224.pt`, the released LIBERO checkpoint, one model for all four suites | sha256 `1000437cfcf55c00…`: the file our matched evaluation scored |
| FasterWAM | github.com/hustvl/FasterWAM (`8366781`) | HF `hustvl/FasterWAM` `libero/step_021700.pt` (rev `6bf9471c`) | sha256 `01be4efca565d189…` = HF LFS sha256 |
| Light-WAM | github.com/L1ziang/Light-WAM (`db7d05e`) | HF `l1ziang/lightwam-checkpoints` `lightwam_libero_10/checkpoints/weights/step_080000.pt` (rev `7cc8593f`); one checkpoint per suite | sha256 `93c816ed1e751271…` = HF LFS sha256 |
| LiLa-WAM | github.com/teee000/LiLa-WAM (`339860f`) | ModelScope `yangfan97/LiLa-WAM_RoboTwin2_0` `checkpoint.pt`, plus DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) | sha256 `40e8aba09a6caeb6…` = ModelScope sha256 |
| LiLa-WAM (ours, LIBERO) | the same LiLa-WAM code + our LIBERO adapter | trained here (Section 6) | sha256 of each uploaded file checked against HF LFS (Section 6.6) |

**Light-WAM.** The four per-suite Light-WAM checkpoints share one architecture and one config apart from training steps, so the LIBERO-Long checkpoint was timed.

**LiLa-WAM's public release.** Its only public weights are a RoboTwin 2.0 model: one 320×240 head camera, a 16-D dual-arm state and 32×14 action chunks. Its latency row times that model on RoboTwin-shaped inputs built from the same LIBERO frames. The LIBERO model we trained has its own row.

## 3. Latency protocol

- **Inputs.** Every model replays the same 500 recorded raw LIBERO observations as the all-models report:
  - sha256 `5815c6fd…`;
  - task 0 of each of the four suites, 125 consecutive steps per suite;
  - 20 warmup queries, then all 500 timed queries in order, batch 1.
- **One query** is the repository's own per-replan call, from the raw simulator observation to the host action array:
  - FastWAM, FasterWAM and Light-WAM: `experiments/libero/eval_libero_single._predict_action_chunk`. It applies the repo's own image and proprio preprocessing, its native inference method and its denormalization, and returns LIBERO actions.
  - LiLa-WAM (RoboTwin release): `RobotWinInference.update_state_buffer` + `_predict_chunk`, which preprocesses, extracts DINOv3 features, runs 10 flow steps, denormalizes and applies B-spline smoothing.
  - LiLa-WAM (ours): the same `_predict_chunk` procedure on LIBERO inputs, built by `libero_obs_to_inputs` (Section 6.3).
  - `torch.cuda.synchronize()` runs before the clock starts. The clock stops when the host action array exists.
- **Model construction.** Each model is built exactly as its repo's `eval_single_process` builds it: its Hydra `sim_libero` config, `instantiate(cfg.model, model_dtype, device)`, `_load_model_checkpoint`, and its processor with the release's dataset stats. Light-WAM also merges the checkpoint's training config, as its eval does.
  - The only change: `model.redirect_common_files=false` makes FastWAM/FasterWAM load the original Wan2.2 `.pth` T5/VAE files instead of DiffSynth-Studio's safetensors conversions. The weights are the same; only the file format differs.
- **Prompt modes.**
  - `native`: what the repo's LIBERO eval does. FastWAM and FasterWAM re-encode the prompt with T5-XXL inside every query; Light-WAM pre-encodes it once per task (`EVALUATION.preencode_prompt_context=true`).
  - `prompt_cached`: T5 runs once per task outside the clock, by memoizing the model's own `encode_prompt`. That is our matched protocol's boundary. Every action chunk was compared with `native`: identical on 500/500 for every model.
- **Compile.** FastWAM's public `sim_libero.yaml` sets `EVALUATION.compile_action_infer: true`, so its native path is `torch.compile`d. A second FastWAM row sets it to false. FasterWAM and Light-WAM have no compile option.
- **Nodes.** Your two interactive jobs, each with one H100: 1094057 on evc102 and 1094056 on evc103. Every released model was timed on both, one model at a time per GPU, as independent repeats.
  - The trained LiLa-WAM was timed only on evc103, after its training ended; evc102 had ended by then.
- **Software.**
  - FastWAM conda env: torch 2.7.1+cu128, bf16.
  - LiLa-WAM: needs transformers 5.0.0rc0 for DINOv3 (plus h5py for training), taken from an isolated overlay `envs/lilawam_venv` on PYTHONPATH. The env itself is unchanged.

## 4. Latency results

### Per-query latency (batch 1, H100 80GB HBM3), ms

Each cell: mean (p50 / p95) over the timed queries, after 20 warmup queries. Hz = 1000 / mean. Two jobs on two nodes measured every model independently.

| Model | Prompt mode | evc102 mean (p50 / p95) | evc103 mean (p50 / p95) | Hz (evc102; the other node where it is -) | Peak GPU GB | Identical to native |
|---|---|---|---|---|---|---|
| FastWAM | native | 73.9 (73.7 / 75.1) | 74.3 (74.0 / 75.5) | 13.5 | 25.0 |  |
| FastWAM | prompt_cached | 59.3 (59.1 / 60.4) | 56.5 (56.4 / 56.8) | 16.9 | 25.0 | 500/500 |
| FastWAM (compile_action_infer=false) | native | 248.0 (247.8 / 250.5) | 249.2 (247.4 / 265.7) | 4.0 | 25.0 |  |
| FastWAM (compile_action_infer=false) | prompt_cached | 232.5 (232.5 / 235.3) | 233.7 (232.8 / 240.4) | 4.3 | 25.0 | 500/500 |
| FasterWAM | native | 193.1 (192.1 / 199.1) | 196.2 (194.1 / 215.2) | 5.2 | 24.1 |  |
| FasterWAM | prompt_cached | 180.8 (179.8 / 186.3) | 183.7 (181.9 / 193.5) | 5.5 | 24.1 | 500/500 |
| Light-WAM (libero_10 release) | native | 49.1 (48.9 / 49.9) | 49.8 (49.7 / 51.2) | 20.4 | 4.3 |  |
| Light-WAM (libero_10 release) | prompt_cached | 49.4 (49.2 / 50.5) | 49.8 (49.7 / 50.5) | 20.3 | 4.3 | 500/500 |
| LiLa-WAM (RoboTwin 2.0 release) | native | 49.5 (49.4 / 50.0) | 50.8 (49.9 / 59.7) | 20.2 | 1.1 |  |
| LiLa-WAM (ours, LIBERO paper recipe, libero_10 policy) | native | - | 52.0 (51.1 / 61.0) | 19.2 | 1.1 |  |

### Configuration actually executed (read back from each run)

| Model | Repo @ commit | Checkpoint | Weights dtype | Denoising steps | Inference path | Prompt handling (native) | Action chunk | Parameters (incl. text encoder, VAE) | Load s |
|---|---|---|---|---|---|---|---|---|---|
| FastWAM | FastWAM @ `7faa711` | `libero_uncond_2cam224.pt` | torch.bfloat16 | 10 | infer_action + torch.compile (EVALUATION.compile_action_infer=true, the repo default) | T5 prompt encoding inside every query | [32, 7] | 12.41 B | 101 |
| FastWAM (compile_action_infer=false) | FastWAM @ `7faa711` | `libero_uncond_2cam224.pt` | torch.bfloat16 | 10 | infer_action (EVALUATION.compile_action_infer=false) | T5 prompt encoding inside every query | [32, 7] | 12.41 B | 96 |
| FasterWAM | FasterWAM @ `8366781` | `step_021700.pt` | torch.bfloat16 | 10 | one_pass_future_cache | T5 prompt encoding inside every query | [32, 7] | 11.94 B | 90 |
| Light-WAM (libero_10 release) | Light-WAM @ `db7d05e` | `step_080000.pt` | torch.bfloat16 | 10 | infer_action | pre-encoded once per task (EVALUATION.preencode_prompt_context=true) | [32, 7] | 7.67 B | 146 |
| LiLa-WAM (RoboTwin 2.0 release) | LiLa-WAM @ `339860f` | `checkpoint.pt` | bfloat16 | 10 | RobotWinInference._predict_chunk (DINOv3 features + flow steps + B-spline smoothing) | no language input; precomputed task-condition vector | [32, 14] | 0.52 B | 7 |
| LiLa-WAM (ours, LIBERO paper recipe, libero_10 policy) | LiLa-WAM @ `339860f` | `checkpoint_epoch_2.pt` | bfloat16 | 10 | public _predict_chunk procedure on LIBERO inputs ([agentview | wrist] 512x256, 8-D state; DINOv3 features + flow steps + B-spline smoothing) | no language input; task-condition vector | [12, 7] | 0.52 B | 8 |

### 4.1 Full statistics of every run (ms)

| Model | Node (job) | Prompt mode | n | mean | std | p5 | p50 | p95 | min | max | Hz | Peak GB | Load s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FastWAM | evc102 (1094057) | native | 500 | 73.89 | 0.62 | 73.11 | 73.75 | 75.11 | 72.77 | 77.94 | 13.5 | 25.00 | 101 |
| FastWAM | evc102 (1094057) | prompt_cached | 500 | 59.30 | 0.51 | 58.89 | 59.12 | 60.41 | 58.52 | 61.90 | 16.9 | 25.00 | 101 |
| FastWAM | evc103 (1094056) | native | 500 | 74.31 | 6.46 | 70.45 | 73.96 | 75.54 | 70.32 | 114.85 | 13.5 | 25.00 | 87 |
| FastWAM | evc103 (1094056) | prompt_cached | 500 | 56.47 | 0.27 | 56.31 | 56.41 | 56.76 | 56.18 | 58.27 | 17.7 | 25.00 | 87 |
| FastWAM (compile_action_infer=false) | evc102 (1094057) | native | 500 | 248.03 | 1.41 | 246.33 | 247.75 | 250.54 | 245.69 | 255.39 | 4.0 | 25.00 | 96 |
| FastWAM (compile_action_infer=false) | evc102 (1094057) | prompt_cached | 500 | 232.53 | 1.60 | 230.33 | 232.48 | 235.26 | 229.26 | 242.76 | 4.3 | 25.00 | 96 |
| FastWAM (compile_action_infer=false) | evc103 (1094056) | native | 500 | 249.23 | 6.36 | 245.10 | 247.44 | 265.72 | 244.46 | 321.97 | 4.0 | 25.00 | 77 |
| FastWAM (compile_action_infer=false) | evc103 (1094056) | prompt_cached | 500 | 233.72 | 6.36 | 230.11 | 232.79 | 240.37 | 229.37 | 337.87 | 4.3 | 25.00 | 77 |
| FasterWAM | evc102 (1094057) | native | 500 | 193.07 | 3.23 | 190.72 | 192.10 | 199.13 | 189.00 | 223.16 | 5.2 | 24.07 | 90 |
| FasterWAM | evc102 (1094057) | prompt_cached | 500 | 180.75 | 2.86 | 178.48 | 179.79 | 186.31 | 177.86 | 209.10 | 5.5 | 24.08 | 90 |
| FasterWAM | evc103 (1094056) | native | 500 | 196.17 | 6.70 | 192.46 | 194.11 | 215.16 | 191.78 | 259.63 | 5.1 | 24.07 | 86 |
| FasterWAM | evc103 (1094056) | prompt_cached | 500 | 183.73 | 4.66 | 180.66 | 181.86 | 193.51 | 179.55 | 210.83 | 5.4 | 24.08 | 86 |
| Light-WAM (libero_10 release) | evc102 (1094057) | native | 500 | 49.06 | 0.45 | 48.67 | 48.93 | 49.91 | 48.51 | 52.81 | 20.4 | 4.34 | 146 |
| Light-WAM (libero_10 release) | evc102 (1094057) | prompt_cached | 500 | 49.36 | 0.54 | 48.80 | 49.21 | 50.48 | 48.46 | 52.48 | 20.3 | 4.34 | 146 |
| Light-WAM (libero_10 release) | evc103 (1094056) | native | 500 | 49.84 | 0.65 | 49.36 | 49.71 | 51.19 | 49.11 | 55.48 | 20.1 | 4.34 | 60 |
| Light-WAM (libero_10 release) | evc103 (1094056) | prompt_cached | 500 | 49.79 | 0.56 | 49.39 | 49.67 | 50.49 | 49.15 | 54.38 | 20.1 | 4.34 | 60 |
| LiLa-WAM (RoboTwin 2.0 release) | evc102 (1094057) | native | 500 | 49.51 | 0.31 | 49.23 | 49.42 | 49.99 | 48.90 | 51.81 | 20.2 | 1.10 | 7 |
| LiLa-WAM (RoboTwin 2.0 release) | evc103 (1094056) | native | 500 | 50.78 | 2.64 | 49.55 | 49.90 | 59.73 | 49.31 | 62.25 | 19.7 | 1.10 | 22 |
| LiLa-WAM (ours, LIBERO paper recipe, libero_10 policy) | evc103 (1094056) | native | 500 | 52.00 | 2.94 | 50.44 | 51.08 | 60.98 | 50.31 | 63.41 | 19.2 | 1.12 | 8 |

### 4.2 Cost of encoding the prompt inside every query

Native minus prompt_cached, mean ms. Light-WAM pre-encodes per task already, so its difference is noise.

| Model | evc102: native − prompt_cached (ms) | evc103: native − prompt_cached (ms) | Share of native (evc102) |
|---|---|---|---|
| FastWAM | +14.6 | +17.8 | 19.7% |
| FastWAM (compile_action_infer=false) | +15.5 | +15.5 | 6.2% |
| FasterWAM | +12.3 | +12.4 | 6.4% |
| Light-WAM (libero_10 release) | -0.3 | +0.1 | -0.6% |

### 4.3 FasterWAM's official latency script (cross-check)

`experiments/latency/measure_latency.py` was run unmodified with its defaults on evc103 (job 1094056):
- random-initialized architectures from the repo configs, with synthetic 224×448 inputs;
- no text encoder;
- BF16, batch 1, 5 warmup and 10 GPU-synchronized iterations.

Values are mean ± population std, in ms.

| Model | Parameters | VAE encode | Visual branch | Action branch | Model infer |
|---|---|---:|---:|---:|---:|
| Joint-WAM | 6.725B | 4.838 ± 0.068 | N/A | N/A | 427.924 ± 1.439 |
| Fast-WAM (no compile) | 6.725B | 4.697 ± 0.043 | 20.846 ± 0.405 | 214.925 ± 0.418 | 244.709 ± 0.786 |
| Faster-WAM | 6.263B | 4.795 ± 0.049 | 22.124 ± 0.123 | 147.169 ± 0.759 | 178.542 ± 0.774 |

These agree with the measured `prompt_cached` uncompiled numbers above (FasterWAM 180.8–183.7 ms, FastWAM 232.5–233.7 ms). FasterWAM's gain comes from the action branch: its one-pass future cache does not re-run the visual branch at each denoising step.

## 5. Reading the latency numbers, and comparison with our models

- **Which number to quote.**
  - `native` answers "how fast does the released system run its own LIBERO eval".
  - `prompt_cached` is the number to compare with our models, because our matched protocol also excludes T5 (our evaluations use a T5 cache).
- **Our checkpoints**, from the all-models report: same 500 observations, same H100 type, T5 cached, our exact `fast` backend. These are the points on our success-latency front.

| Model | Config | LIBERO success % (6,000 episodes, matched) | Mean ms | p95 ms |
|---|---|---|---|---|
| elastic SP11 | [1,1] | 93.77 | 23.4 | 24.9 |
| elastic E0 | [1,1] | 94.33 | 24.7 | 24.8 |
| elastic E1 | [2,2] | 94.58 | 27.6 | 27.7 |
| elastic E2 | [2,2] | 94.75 | 27.7 | 27.8 |
| RT-A2 | S7 [4,2] | 94.93 | 31.5 | 33.9 |
| elastic E0 | [4,4] | 94.97 | 33.2 | 33.3 |
| elastic E2 | [4,4] | 95.27 | 33.4 | 33.4 |
| elastic F0-PB | [4,4] | 95.88 | 37.0 | 37.6 |
| ChronoLoop CL-A a4 | native | 96.18 | 92.1 | 99.6 |
| all-suites v0 4/4 | native | 94.92 | 99.3 | 99.6 |
| FastWAM (released) through our exact harness | native | 96.63 | 107.5 | 108.9 |

- **FastWAM through our harness vs its own default.** Our harness gives 107.5 ms; the public default gives 56–59 ms. Two things differ:
  - The public default stores weights in BF16 and runs Inductor `torch.compile`.
  - Our harness keeps the FP32-stored weights under BF16 autocast, the setup our matched evaluation scored, and requires bit-exact actions.

  The all-models report measured Inductor compile at 87.3 ms on our FP32-weight setup, with 0/50 chunks bit-identical to eager. So the public default's actions are not the ones that scored 96.63%. Uncompiled, the public BF16 path takes 232.5 ms; our eager path took 264.2 ms.
- **The fastest public baselines vs our models** (prompt cached):
  - Light-WAM (49.4 ms) and the LiLa-WAM RoboTwin model (49.5 ms) are about 2× slower than our [1,1] Elastic points (23–25 ms) and 1.3× slower than F0-PB [4,4] (37.0 ms).
  - Compiled FastWAM (56–59 ms) is 1.5–1.6× slower than F0-PB.
  - FasterWAM (181–184 ms) is 4.9–5.0× slower.
  - The success rates are in Section 7.
- **Noise.** Repeated runs on these GPUs can differ by about 5–10% (two latency levels, see the all-models report). Treat differences under about 10% as ties.

## 6. LiLa-WAM trained on our LIBERO data

### 6.1 Why

The paper (arXiv 2608.03701) reports LIBERO results: Spatial 98.0, Object 98.8, Goal 97.2, Long 94.2, average 97.1, with one policy per suite and 50 rollouts per task. But the public code (`339860f`) contains only the RoboTwin pipeline:
- no LIBERO config, dataloader or evaluation;
- no LIBERO weights.

The authors' ModelScope repo `yangfan97/LiLa-WAM_Libero` holds LIBERO **data** in HDF5. We inspected one file to copy its layout. We did not use that data.

### 6.2 Data (ours)

- **Source.** Our LIBERO training data: the FastWAM regeneration with MuJoCo 3.3.2 and no-op actions removed, in LeRobot v2.1 format (`data/FastWAM/libero_mujoco3.3.2/*_no_noops_lerobot`). All four suites: 1,712 demonstrations, 277,713 frames, 40 tasks.
- **Conversion** to the authors' HDF5 layout (`convert_ours_to_lilawam_hdf5.py`), one file per demonstration (`<suite>/<task>/demo_clean/data/episode_XXXXX.hdf5`), 36 GB in total:
  - `observation/agentview/rgb` and `observation/robot0_eye_in_hand/rgb`: our 512×512 frames resized to 256×256 (INTER_AREA) and stored as PNG. The frames are upright, like our LeRobot data and like the authors' files.
  - `action` (T,7): our six end-effector deltas, unchanged, plus the gripper converted to raw LIBERO (−1 open, +1 close; = 1 − 2·ours). The trained policy therefore outputs raw environment actions.
  - `proprio` (T,8): our `observation.state` (eef position 3, axis-angle 3, gripper qpos 2). This is the same state our FastWAM/LoopWAM evaluations feed; the authors' proprio is position + quaternion + 1.
  - No cleaning: every frame of every demonstration is used.
- **Verification** (`verify_conversion.py`): 24 randomly chosen demonstrations (3,876 frames), all checks passed: True. On each of those demonstrations:
  - stored actions, gripper mapping and states equal the source exactly;
  - frame counts equal the parquet rows for both cameras;
  - every stored image equals the re-decoded, resized source frame pixel for pixel.

### 6.3 Model inputs, normalization and the adapter

The adapter is in `lilawam_libero_20261030/`. Only data access is new: `train_libero.py` runs the public `train.py` unchanged, with its dataset factory swapped for `libero_data.create_dataset`.
- **Views.** The paper feeds both views at 256×256 but does not say how they are fused, and the public model takes one image. So the two views are placed side by side: `[agentview | wrist]`, one 512×256 RGB image.
  - The frozen DINOv3 ViT-L/16 (RoPE) sees 32×16 patches.
  - The future-feature target is the same composite 12 steps ahead.
  - Images are decoded to RGB explicitly; the public `_decode` returns BGR.
- **Normalization.** Action and state min-max per suite, computed over every frame (`stats_libero.py`). LiLa-WAM's loader reads them from the top-level key `robotwin2`.
- **Task condition.** As in the public `precompute_task_cond.py`: the last-layer DINOv3 CLS of a demonstration's last frame minus its first frame, averaged over the task's demonstrations, computed on the same composite. That gives 40 vectors of 1024-D, keyed by `<suite>/<task>`.
- **At evaluation time:**
  - `libero_obs_to_inputs` rotates both raw views by 180°, as FastWAM's `get_libero_image` does, because our training frames are upright. It resizes each view to 256×256 and builds the composite.
  - It builds the state as eef pos + `quat2axisangle(quat)` + gripper qpos. Its `quat2axisangle` gives exactly the same output as FastWAM's (max difference 0.0 over the 500 observations).

### 6.4 Training recipe: the paper vs this run

| Item | Paper (Sec. 4.1–4.2) | This run |
|---|---|---|
| Policies | one per suite (4) | one per suite (4) |
| Views | third-person + wrist, 256×256 | the same, fused side by side (fusion not described in the paper) |
| Action chunk | 12 | 12 |
| Vision encoder | frozen DINOv3 ViT-L/16 | the same; features from layers −12/−8/−4 + CLS/registers (public config) |
| Action model | DiT action expert | public config: depth 12, hidden 768, 8 heads, 64 adapter queries; 212.1M trainable parameters |
| Loss | flow matching + λff = 0.5 future-feature loss; logit-normal t | the same |
| Optimizer | AdamW (0.9, 0.99), wd 0.01, batch 128, clip 1.0 | the same |
| LR schedule | "stage-wise cosine, peak 2e-4 in the early stage and 5e-5 in the later stage" | stage 1: cosine 2e-4 → 5e-5 (public config floor). Stage 2: the public `--init_from` (weights only; fresh optimizer and schedule), cosine 5e-5 → 5e-6 |
| Epochs | not stated | stage 1: 5, stage 2: 2, per suite, sized to fit job 1094056 |
| Data cleaning | not stated | none |
| Precision / GPU | bf16, "fits a 24 GB GPU" | bf16 autocast, one H100 80GB (23.6 GB used), 0.61 s/update, 12 data workers |

### 6.5 Training runs

- **Where it ran.** Interactive job 1094056 on evc103, step 7. The steps for each suite were:
  1. stage 1 (`configs/<suite>_stage1.yaml`);
  2. stage 2 initialized from stage-1 epoch 5 (`configs/<suite>_stage2.yaml`).
- **Order:** libero_10, goal, object, spatial.
- **Robustness.** The srun client runs on the login node, so training does not depend on a Claude session. Every epoch is checkpointed.
- **Wall-clock times** below are evc103 local time.

| Suite | Demos | Frames | Updates / epoch | Stage | LR (cosine) | Epochs done | Per-epoch mean loss | Epoch wall s | Start → end | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| Long (libero_10) | 388 | 104,280 | 814 | stage1 | 2e-04 → 5e-05 | 5/5 | 0.2737 → 0.1731 → 0.1546 → 0.1430 → 0.1371 | 507 / 499 / 499 / 500 / 503 | 09:37:17 → 10:19:54 | complete |
| Long (libero_10) | 388 | 104,280 | 814 | stage2 | 5e-05 → 5e-06 | 2/2 | 0.1348 → 0.1318 | 502 / 498 | 10:19:54 → 10:36:54 | complete (init from stage-1 epoch 5) |
| Goal | 433 | 52,895 | 413 | stage1 | 2e-04 → 5e-05 | 5/5 | 0.3463 → 0.1971 → 0.1710 → 0.1591 → 0.1520 | 265 / 254 / 254 / 255 / 254 | 10:36:54 → 10:59:01 | complete |
| Goal | 433 | 52,895 | 413 | stage2 | 5e-05 → 5e-06 | 2/2 | 0.1484 → 0.1456 | 276 / 318 | 10:59:01 → 11:09:28 | complete (init from stage-1 epoch 5) |
| Object | 457 | 67,309 | 525 | stage1 | 2e-04 → 5e-05 | 5/5 | 0.2800 → 0.1623 → 0.1444 → 0.1352 → 0.1299 | 348 / 323 / 322 / 322 / 324 | 11:09:28 → 11:38:04 | complete |
| Object | 457 | 67,309 | 525 | stage2 | 5e-05 → 5e-06 | 2/2 | 0.1277 → 0.1247 | 352 / 324 | 11:38:04 → 11:51:36 | complete (init from stage-1 epoch 5) |
| Spatial | 434 | 53,229 | 415 | stage1 | 2e-04 → 5e-05 | 5/5 | 0.3419 → 0.1951 → 0.1742 → 0.1626 → 0.1555 | 277 / 257 / 255 / 257 / 256 | 11:51:36 → 12:14:32 | complete |
| Spatial | 434 | 53,229 | 415 | stage2 | 5e-05 → 5e-06 | 2/2 | 0.1528 → 0.1503 | 258 / 257 | 12:14:32 → 12:23:40 | complete (init from stage-1 epoch 5) |

A joint four-suite run started earlier was cancelled after about 6 minutes, before saving any checkpoint, when the paper recipe (one policy per suite) was requested. Its folder is marked SUPERSEDED.

### 6.6 Checkpoints on Hugging Face

- **Location:** `anhdao69/LoopWAM_NT` (public), folder `baselines/LiLa-WAM_LIBERO_ours/`.
- **Upload process.** `hf_upload_watcher.py`, running on the login node, uploads each suite as soon as its stage 2 finishes, in one commit per suite. It then reads back the HF LFS sha256 of every checkpoint and compares it with the local file.
- **What else is uploaded with each suite:** both stage configs, the suite's norm stats, the training logs and per-update loss CSVs, and the 40 task vectors. The first commit also adds the code and a model card.
- **Checkpoint contents.** Each `.pt` holds the 212.1M-parameter action model (bf16), the optimizer and scheduler state, the epoch and the loss. The frozen DINOv3 encoder is not included.

| Suite | File on Hugging Face (`anhdao69/LoopWAM_NT`) | Epoch / update | Epoch mean loss | Size GB | sha256 | HF LFS sha256 = local | Commit |
|---|---|---|---|---|---|---|---|
| Long (libero_10) | `baselines/LiLa-WAM_LIBERO_ours/libero_10/stage2_epoch2_final.pt` | 2 / 1628 | 0.1318 | 1.27 | `27ad99a458d97a68…` | yes | `4544aebb8c` |
| Long (libero_10) | `baselines/LiLa-WAM_LIBERO_ours/libero_10/stage1_epoch5.pt` | 5 / 4070 | 0.1371 | 1.27 | `05cdc8445b3f16ca…` | yes | `4544aebb8c` |
| Goal | `baselines/LiLa-WAM_LIBERO_ours/libero_goal/stage2_epoch2_final.pt` | 2 / 826 | 0.1456 | 1.27 | `5ac2bfd3aad9c2ca…` | yes | `595b21b8b5` |
| Goal | `baselines/LiLa-WAM_LIBERO_ours/libero_goal/stage1_epoch5.pt` | 5 / 2065 | 0.1520 | 1.27 | `2a7e5fc01631e54e…` | yes | `595b21b8b5` |
| Object | `baselines/LiLa-WAM_LIBERO_ours/libero_object/stage2_epoch2_final.pt` | 2 / 1050 | 0.1247 | 1.27 | `4571a5f45f02dc61…` | yes | `efda9c4ebf` |
| Object | `baselines/LiLa-WAM_LIBERO_ours/libero_object/stage1_epoch5.pt` | 5 / 2625 | 0.1299 | 1.27 | `5564d2cd2ff4b04a…` | yes | `efda9c4ebf` |
| Spatial | `baselines/LiLa-WAM_LIBERO_ours/libero_spatial/stage2_epoch2_final.pt` | 2 / 830 | 0.1503 | 1.27 | `0836ecc4ceefb7fe…` | yes | `8703bb9113` |
| Spatial | `baselines/LiLa-WAM_LIBERO_ours/libero_spatial/stage1_epoch5.pt` | 5 / 2075 | 0.1555 | 1.27 | `5496e10a9303367e…` | yes | `8703bb9113` |

`manifest.json` in that folder lists the full sha256 values.

**Uploader incident (no data affected).** The first uploader instance crashed at 10:37 before uploading anything. `import torch` fails on the login node (`libnccl.so.2: failed to map segment`), and it had used torch only to read epoch, step and loss from each checkpoint.
- It now reads them from the training logs. For both libero_10 checkpoints the values are identical to the checkpoint contents read with torch: stage 1 epoch 5 is step 4070, loss 0.1371; stage 2 epoch 2 is step 1628, loss 0.1318.
- It was restarted at 10:38, and libero_10 was uploaded and verified at 10:39. The crash log is in `upload/hf_upload_watcher_crash1.log`.

### 6.7 Open-loop sanity check (not a success rate)

- **What it runs.** `openloop_check.py` builds the model exactly as the public `RobotWinInference` does (strict state-dict load: 0 missing, 0 unexpected keys). It samples 12-step chunks with the public procedure (Gaussian start, 10 Euler steps, denormalize).
- **What it is tested on.** 512 random training samples of each suite, seed 0. These are **held-in** frames, so this checks that the weights, stats, task vectors and inference path reproduce the demonstrations. It does not measure generalization or task success.
- **Baseline:** always predicting the suite's mean action.
- **Units:** errors are mean absolute error in raw action units, over valid (non-padded) steps.

| Suite | Checkpoint | Samples (valid steps) | MAE, 6 EEF deltas: model | smoothed | mean-action baseline | First-step MAE: model / baseline | Gripper sign agreement: model / baseline | max abs(raw − smoothed) |
|---|---|---|---|---|---|---|---|---|
| Long (libero_10) | stage-1 epoch 5 | 512 (6030) | 0.0729 | 0.0729 | 0.1452 | 0.0759 / 0.1454 | 0.937 / 0.534 | 1.1661 |
| Long (libero_10) | final (stage-2 epoch 2) | 512 (6030) | 0.0702 | 0.0702 | 0.1452 | 0.0730 / 0.1454 | 0.949 / 0.534 | 1.1441 |
| Goal | stage-1 epoch 5 | 512 (5852) | 0.0889 | 0.0889 | 0.1845 | 0.0895 / 0.1844 | 0.948 / 0.633 | 1.1998 |
| Goal | final (stage-2 epoch 2) | 512 (5852) | 0.0853 | 0.0853 | 0.1845 | 0.0864 / 0.1844 | 0.956 / 0.633 | 0.9327 |
| Object | stage-1 epoch 5 | 512 (5884) | 0.0709 | 0.0709 | 0.1600 | 0.0714 / 0.1591 | 0.939 / 0.537 | 0.9206 |
| Object | final (stage-2 epoch 2) | 512 (5884) | 0.0682 | 0.0682 | 0.1600 | 0.0686 / 0.1591 | 0.939 / 0.537 | 1.0651 |
| Spatial | stage-1 epoch 5 | 512 (5839) | 0.0885 | 0.0884 | 0.1939 | 0.0909 / 0.1975 | 0.949 / 0.536 | 1.1657 |
| Spatial | final (stage-2 epoch 2) | 512 (5839) | 0.0855 | 0.0855 | 0.1939 | 0.0880 / 0.1975 | 0.950 / 0.536 | 1.1563 |

The public B-spline smoothing (8 control points over 12 steps) barely changes the predicted chunks, which are already smooth. On random 12×7 input, the same function changes values by up to 1.13. So it is active, just not needed here.

### 6.8 Latency of the trained LIBERO policy

52.0 ms mean (p50 51.1, p95 61.0), 1.12 GB peak, measured on evc103 after training finished with the GPU otherwise idle. It is the `lilawam_libero` row in Section 4.
- One query is the public `_predict_chunk` procedure on LIBERO inputs: two-view composite, 8-D state, DINOv3 features, 10 flow steps, B-spline smoothing.
- The libero_10 policy was timed on all 500 observations; the four suite policies share one architecture.

### 6.9 LIBERO success rate under our matched protocol

- **Protocol:** `lightwam_20261009_matched`, the same as every matched model in our reports. The runner is `evaluate_lilawam_matched.py`. It imports the constants, group claims and summary code of the LoopWAM runner `evaluate_v0_matched.py` unchanged.
- **Grid:** 4 suites × 10 tasks × official initial states 0–49 × seeds 42/43/44 = 6,000 episodes.
- **Per task/seed group:**
  - one environment, created with `env.seed(seed)` and reused for its 50 initial states in order;
  - `set_global_seed(seed)` once;
  - 30 settling actions `[0,0,0,0,0,0,-1]`, with `done` ignored;
  - then at most 400 policy steps (700 for Long);
  - success = the environment's `done` flag.
- **LiLa-WAM-specific choices** (upstream has no LIBERO eval; all are recorded in `eval_matched/manifest.json`):
  - each suite runs its own final policy;
  - inputs come from `libero_obs_to_inputs`, the training format;
  - sampling uses the public procedure: 10 Euler steps, then B-spline smoothing;
  - every query of a group starts from the same seed noise;
  - **6 of the 12 predicted actions are executed per query** (half the chunk, as the public RoboTwin config executes 16 of 32);
  - the gripper is executed as sign(predicted raw gripper).
- **Speed setup.** One batched bf16 inference server per GPU (Long + Goal on one GPU, Spatial + Object on the other) serves many CPU-only simulator workers, each with one environment.
  - Batching changes only the batch size of the same computation. It is not bit-identical to batch-1 inference.

- **Pre-checks before the full run** (evc103, GPU shared with training, seed 42, 2 episodes per task; not part of the result). These were run to separate a pipeline bug from a weak policy:
  - The eval image and state match the training format pixel-for-pixel in layout (`lilawam/eval_debug/compare_eval_top_train_bottom.png`).
  - The sampling code equals the public `_predict_chunk`, and the public processor's resize is a no-op on our 512×256 composite.
  - The action conventions match our LoopWAM eval: EEF deltas unchanged; gripper open = −1.
  - All 40 task names map to their task vectors.
  - Closed loop: **Long 0/20** (10 tasks). **Goal 1/20**: "put the bowl on top of the cabinet" succeeded in 101 steps.
  - The robot reaches toward the right objects but rarely grasps (`lilawam/eval_debug/closed_loop_montage.png`).
- **What the pre-checks did and did not show.** They ruled out a broken pipeline: a task was completed, and the inputs and sampling match training. They covered only Long and Goal with 2 episodes per task, so they say nothing about Spatial or Object. The full 6,000-episode table below is the result.
- **Training budget, for reading the table:**
  - 5 + 2 epochs per suite: about 5.7k updates for Long, about 3k for Goal, and similar for Object and Spatial (Section 6.5);
  - training loss still falling at the end of every stage.
  - The paper does not state its epoch count. Where a suite falls well short of the paper's 94–99%, longer training is the first thing to try.
- **Shared encoder.** Each GPU server encodes a batch once with the frozen pretrained DINOv3 and gives the features to each suite's action model. This is exact:
  - all four suite configs use the same DINOv3 checkpoint, layers (−12/−8/−4) and token settings;
  - the trained checkpoints contain only the action model (loaded with `strict=True`), which holds the per-suite feature projection.

| Suite | Successes / episodes | Success % | Seed 42 / 43 / 44 % | FastWAM (ours, matched) % | Paper (LiLa-WAM) % |
|---|---|---|---|---|---|
| Spatial | 554 / 1500 | **36.93** | 9.6 / 68.6 / 32.6 | 97.47 | 98.0 |
| Object | 293 / 1500 | **19.53** | 18.6 / 21.6 / 18.4 | 99.27 | 98.8 |
| Goal | 495 / 1500 | **33.00** | 11.0 / 60.8 / 27.2 | 96.27 | 97.2 |
| Long (libero_10) | 27 / 1500 | **1.80** | 0.2 / 5.2 / 0.0 | 93.53 | 94.2 |
| **All** | 1369 / 6000 | **22.82** | 9.85 / 39.05 / 19.55 | 96.63 | 97.1 |

- **Completed:** 6000 episodes, 2,488,241 policy steps, 416,811 policy queries.
- **Runner time:** 42.5 min wall across 1 runner invocation(s), in job 1102933 on evc103. GPUs: 0,1; 80 simulator workers.
- **Last group completed:** 2026-10-30 13:53:26.
- **Checkpoint identity:** combined sha256 of the four policies `5f73fc1a4b850e6d…`. Per-suite sha256 values are in `eval_matched/manifest.json` and match Section 6.6.

- **Seed spread is large.** See the per-seed column, e.g. Spatial across seeds 42/43/44. The seed sets three things:
  - the environment seed;
  - the global RNG;
  - the starting noise of *every* policy query in that task/seed group. Every query uses the same 12×7 Gaussian draw from the base seed, mirroring the LoopWAM protocol's "every query uses the base seed".
- For a flow policy this short-trained, that one fixed noise draw evidently shifts behavior across all 50 episodes of a group. The public LiLa-WAM inference (`_predict_chunk`) draws fresh noise per query instead. A fresh-noise run would likely give a different, less seed-sensitive number; it was not run.
- **Episode lengths.** Successful episodes end early, so the run used 2.49M policy steps instead of the ~2.9M worst case.

## 7. Success-rate context

None of these numbers come from the latency runs.

| Model | LIBERO success (ours, matched protocol) | Episodes | Notes |
|---|---|---|---|
| FastWAM (released) | **96.63%**: Long 93.53 (1403/1500), Goal 96.27 (1444/1500), Object 99.27 (1489/1500), Spatial 97.47 (1462/1500) | 6,000 (seeds 42/43/44) | the same checkpoint file timed here (sha256 match) |
| Light-WAM | Long only: **90.87%** (1363/1500); seeds 90.4 / 90.0 / 92.2 | 1,500 | the earlier evaluation scored `lightwam_libero_10/.../step_080000.pt`, the file timed here by name. Its sha256 was not recorded in that evaluation. |
| FasterWAM | not evaluated by us | – | |
| LiLa-WAM (RoboTwin release) | not applicable: a RoboTwin model | – | |
| LiLa-WAM (ours, paper recipe) | **22.82% (1369/6000)** (Section 6.9) | 6,000 (seeds 42/43/44) | the paper reports 98.0 / 98.8 / 97.2 / 94.2 (Spatial / Object / Goal / Long), 97.1 average, with the authors' own data and training length |

## 8. Caveats

- **Different query boundaries.** This report times each repo's own query function end to end. The all-models report times our adapters. `prompt_cached` uses the same text boundary, but preprocessing code differs between repos (PIL vs torch resizing, CPU vs GPU), so the numbers compare closely but not identically.
- **Peak GPU memory** is `torch.cuda.max_memory_allocated` during the timed queries. It includes resident encoders: T5-XXL for FastWAM and FasterWAM, and DINOv3 for LiLa-WAM.
- **Outliers on evc103.** A few queries on evc103 show outliers: FastWAM native max 114.9 ms; FastWAM uncompiled max 322–338 ms. They raise the std and p95 there, but medians agree across nodes within 5%.
- **Our LiLa-WAM is not the authors' LIBERO model.** It differs in:
  - training data: ours;
  - epochs: the paper gives none;
  - stage-2 LR floor;
  - view fusion: side by side;
  - no data cleaning.

  Its success rate, once measured, is a result for this recipe on our data, not a reproduction of the paper's numbers.
- **Comparable cost, but a different policy.** The released RoboTwin LiLa-WAM row (49.5 ms) and our LIBERO model's row differ in inputs: one 320×240 view vs a 512×256 composite, and 32×14 vs 12×7 actions. The architecture and the number of flow steps are the same.

## 9. Files and reproduction

- **GitHub, branch `elastic`:**
  - `plans/performance/public_wam_latency_20261030/`: result JSONs with per-query times, the harness, `env.sh`, `run_models.sh`, `build_tables.py`, `make_report.py`, the FasterWAM official-script output, checkpoint hashes and node logs.
  - `plans/performance/lilawam_libero_20261030/`: converter, adapter, training wrapper, configs, per-suite norm stats, job scripts, the HF upload watcher, `manifest.json`, upload verification records, open-loop results, conversion verification and HANDOFF.
- **Hugging Face:** `anhdao69/LoopWAM_NT` under `baselines/LiLa-WAM_LIBERO_ours/`.
- **Cluster:**
  - code clones: `/groups/yshang/an221229/code/wam_baselines_20261030/`;
  - weights and latency: `/groups/yshang/an221229/checkpoints/wam_baselines/`;
  - converted data: `/groups/yshang/an221229/data/LiLaWAM_libero_ours_hdf5` (36 GB);
  - training run: `/groups/yshang/an221229/checkpoints/wam_baselines/lilawam/libero_ours_paper_recipe/`.
- **Reproduce latency:** `source env.sh; bash run_models.sh <tag> fastwam fastwam_nocompile fasterwam lightwam; bash run_lilawam.sh <tag>`, then `measure_lilawam_libero_latency.py --config <stage2 yaml> --ckpt <final .pt> --stats <suite stats> --out results/lilawam_libero_<node>.json`, then `make_report.py <out.md>`.
- **Reproduce training:**
  1. `convert_ours_to_lilawam_hdf5.py <out>`;
  2. `stats_libero.py <out> <stats.json> <suite>`;
  3. `task_cond_libero.py <out> <taskcond dir> <dinov3 dir>`;
  4. `train_job.sh`, inside a GPU allocation.
