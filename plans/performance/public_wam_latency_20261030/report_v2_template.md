# Public WAM baselines on LIBERO: latency of the released models, and LiLa-WAM trained on our data (2026-10-30)

This report covers two pieces of work on H100 80GB HBM3. Branch `elastic`.

1. **Latency of four public world-action models**, each run through **its own public code with its own released weights**, unoptimized: FastWAM, FasterWAM, Light-WAM and LiLa-WAM.
2. **LiLa-WAM trained on our LIBERO data**, because the authors released no LIBERO model. It uses the paper's LIBERO recipe. The weights are on Hugging Face under `baselines/`, along with an open-loop check and the trained model's latency.

{{TRAIN_STATUS}} The LIBERO success-rate evaluation of the trained LiLa-WAM under our matched protocol has **not** been run yet (Section 6.9).

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
  - The data conversion was verified exactly: {{CONV}}.
  - Training progress and losses: Section 6.5. Weights and their sha256 verification: Section 6.6. Open-loop check: Section 6.7.
  - Latency of the trained LIBERO policy: {{LAT_TRAINED}}.

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

{{LAT_TABLES}}

### 4.1 Full statistics of every run (ms)

{{LAT_FULL}}

### 4.2 Cost of encoding the prompt inside every query

Native minus prompt_cached, mean ms. Light-WAM pre-encodes per task already, so its difference is noise.

{{T5}}

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
- **Verification** (`verify_conversion.py`): {{CONV}}. On each of those demonstrations:
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

{{TRAIN_TABLE}}

A joint four-suite run started earlier was cancelled after about 6 minutes, before saving any checkpoint, when the paper recipe (one policy per suite) was requested. Its folder is marked SUPERSEDED.

### 6.6 Checkpoints on Hugging Face

- **Location:** `anhdao69/LoopWAM_NT` (public), folder `baselines/LiLa-WAM_LIBERO_ours/`.
- **Upload process.** `hf_upload_watcher.py`, running on the login node, uploads each suite as soon as its stage 2 finishes, in one commit per suite. It then reads back the HF LFS sha256 of every checkpoint and compares it with the local file.
- **What else is uploaded with each suite:** both stage configs, the suite's norm stats, the training logs and per-update loss CSVs, and the 40 task vectors. The first commit also adds the code and a model card.
- **Checkpoint contents.** Each `.pt` holds the 212.1M-parameter action model (bf16), the optimizer and scheduler state, the epoch and the loss. The frozen DINOv3 encoder is not included.

{{CKPT_TABLE}}

`manifest.json` in that folder lists the full sha256 values.

### 6.7 Open-loop sanity check (not a success rate)

- **What it runs.** `openloop_check.py` builds the model exactly as the public `RobotWinInference` does (strict state-dict load: 0 missing, 0 unexpected keys). It samples 12-step chunks with the public procedure (Gaussian start, 10 Euler steps, denormalize).
- **What it is tested on.** 512 random training samples of each suite, seed 0. These are **held-in** frames, so this checks that the weights, stats, task vectors and inference path reproduce the demonstrations. It does not measure generalization or task success.
- **Baseline:** always predicting the suite's mean action.
- **Units:** errors are mean absolute error in raw action units, over valid (non-padded) steps.

{{OPENLOOP}}

The public B-spline smoothing (8 control points over 12 steps) barely changes the predicted chunks, which are already smooth. On random 12×7 input, the same function changes values by up to 1.13. So it is active, just not needed here.

### 6.8 Latency of the trained LIBERO policy

{{LAT_TRAINED}}. It is the `lilawam_libero` row in Section 4.
- One query is the public `_predict_chunk` procedure on LIBERO inputs: two-view composite, 8-D state, DINOv3 features, 10 flow steps, B-spline smoothing.
- The libero_10 policy was timed on all 500 observations; the four suite policies share one architecture.

### 6.9 Not done yet: LIBERO success rate under our matched protocol

The evaluation needs a GPU allocation and a LiLa-WAM policy adapter for our LIBERO runner. The protocol is 4 suites × 10 tasks × 50 trials × seeds 42/43/44 = 6,000 episodes. The adapter should:
- for each episode, load the suite's final checkpoint, its config, its `norm_stats.json` and the task's `task_cond.npy`;
- build inputs with `libero_data.libero_obs_to_inputs`;
- sample with the public procedure;
- send actions to the environment as-is, **without** FastWAM's gripper flip, because the policy outputs raw LIBERO actions;
- use an execution horizon that must be decided and recorded. The paper gives none for LIBERO; the public RoboTwin setting executes 16 of 32 steps.

`plans/performance/lilawam_libero_20261030/HANDOFF.md` lists these steps.

## 7. Success-rate context

None of these numbers come from the latency runs.

| Model | LIBERO success (ours, matched protocol) | Episodes | Notes |
|---|---|---|---|
| FastWAM (released) | **96.63%**: Long 93.53 (1403/1500), Goal 96.27 (1444/1500), Object 99.27 (1489/1500), Spatial 97.47 (1462/1500) | 6,000 (seeds 42/43/44) | the same checkpoint file timed here (sha256 match) |
| Light-WAM | Long only: **90.87%** (1363/1500); seeds 90.4 / 90.0 / 92.2 | 1,500 | the earlier evaluation scored `lightwam_libero_10/.../step_080000.pt`, the file timed here by name. Its sha256 was not recorded in that evaluation. |
| FasterWAM | not evaluated by us | – | |
| LiLa-WAM (RoboTwin release) | not applicable: a RoboTwin model | – | |
| LiLa-WAM (ours, paper recipe) | **pending** (Section 6.9) | – | the paper reports 98.0 / 98.8 / 97.2 / 94.2 (Spatial / Object / Goal / Long), 97.1 average, with the authors' own data and training length |

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
