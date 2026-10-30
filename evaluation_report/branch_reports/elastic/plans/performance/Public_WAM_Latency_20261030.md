# Public WAM baselines on LIBERO: latency of the released models (2026-10-30)

This report measures **the latency of four public world-action models**, each run through **its own public code with its own released weights**, unoptimized: FastWAM, FasterWAM, Light-WAM and LiLa-WAM. Hardware: H100 80GB HBM3. Branch `elastic`.

Where things are:
- **Data and scripts:** `plans/performance/public_wam_latency_20261030/`.
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

## 2. What was run

| Model | Public code (commit) | Released weights | Verified |
|---|---|---|---|
| FastWAM | github.com/yuantianyuan01/FastWAM (`7faa711`) | `libero_uncond_2cam224.pt`, the released LIBERO checkpoint, one model for all four suites | sha256 `1000437cfcf55c00…`: the file our matched evaluation scored |
| FasterWAM | github.com/hustvl/FasterWAM (`8366781`) | HF `hustvl/FasterWAM` `libero/step_021700.pt` (rev `6bf9471c`) | sha256 `01be4efca565d189…` = HF LFS sha256 |
| Light-WAM | github.com/L1ziang/Light-WAM (`db7d05e`) | HF `l1ziang/lightwam-checkpoints` `lightwam_libero_10/checkpoints/weights/step_080000.pt` (rev `7cc8593f`); one checkpoint per suite | sha256 `93c816ed1e751271…` = HF LFS sha256 |
| LiLa-WAM | github.com/teee000/LiLa-WAM (`339860f`) | ModelScope `yangfan97/LiLa-WAM_RoboTwin2_0` `checkpoint.pt`, plus DINOv3 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`) | sha256 `40e8aba09a6caeb6…` = ModelScope sha256 |

**Light-WAM.** The four per-suite Light-WAM checkpoints share one architecture and one config apart from training steps, so the LIBERO-Long checkpoint was timed.

**LiLa-WAM's public release.** Its only public weights are a RoboTwin 2.0 model: one 320×240 head camera, a 16-D dual-arm state and 32×14 action chunks. Its latency row times that model on RoboTwin-shaped inputs built from the same LIBERO frames.

## 3. Latency protocol

- **Inputs.** Every model replays the same 500 recorded raw LIBERO observations as the all-models report:
  - sha256 `5815c6fd…`;
  - task 0 of each of the four suites, 125 consecutive steps per suite;
  - 20 warmup queries, then all 500 timed queries in order, batch 1.
- **One query** is the repository's own per-replan call, from the raw simulator observation to the host action array:
  - FastWAM, FasterWAM and Light-WAM: `experiments/libero/eval_libero_single._predict_action_chunk`. It applies the repo's own image and proprio preprocessing, its native inference method and its denormalization, and returns LIBERO actions.
  - LiLa-WAM (RoboTwin release): `RobotWinInference.update_state_buffer` + `_predict_chunk`, which preprocesses, extracts DINOv3 features, runs 10 flow steps, denormalizes and applies B-spline smoothing.
  - `torch.cuda.synchronize()` runs before the clock starts. The clock stops when the host action array exists.
- **Model construction.** Each model is built exactly as its repo's `eval_single_process` builds it: its Hydra `sim_libero` config, `instantiate(cfg.model, model_dtype, device)`, `_load_model_checkpoint`, and its processor with the release's dataset stats. Light-WAM also merges the checkpoint's training config, as its eval does.
  - The only change: `model.redirect_common_files=false` makes FastWAM/FasterWAM load the original Wan2.2 `.pth` T5/VAE files instead of DiffSynth-Studio's safetensors conversions. The weights are the same; only the file format differs.
- **Prompt modes.**
  - `native`: what the repo's LIBERO eval does. FastWAM and FasterWAM re-encode the prompt with T5-XXL inside every query; Light-WAM pre-encodes it once per task (`EVALUATION.preencode_prompt_context=true`).
  - `prompt_cached`: T5 runs once per task outside the clock, by memoizing the model's own `encode_prompt`. That is our matched protocol's boundary. Every action chunk was compared with `native`: identical on 500/500 for every model.
- **Compile.** FastWAM's public `sim_libero.yaml` sets `EVALUATION.compile_action_infer: true`, so its native path is `torch.compile`d. A second FastWAM row sets it to false. FasterWAM and Light-WAM have no compile option.
- **Nodes.** Your two interactive jobs, each with one H100: 1094057 on evc102 and 1094056 on evc103. Every released model was timed on both, one model at a time per GPU, as independent repeats.
- **Software.**
  - FastWAM conda env: torch 2.7.1+cu128, bf16.
  - LiLa-WAM: needs transformers 5.0.0rc0 for DINOv3, taken from an isolated overlay `envs/lilawam_venv` on PYTHONPATH. The env itself is unchanged.

## 4. Latency results

### Per-query latency (batch 1, H100 80GB HBM3), ms

Each cell: mean (p50 / p95) over the timed queries, after 20 warmup queries. Hz = 1000 / mean. Two jobs on two nodes measured every model independently.

| Model | Prompt mode | evc102 mean (p50 / p95) | evc103 mean (p50 / p95) | Hz (evc102) | Peak GPU GB | Identical to native |
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

### Configuration actually executed (read back from each run)

| Model | Repo @ commit | Checkpoint | Weights dtype | Denoising steps | Inference path | Prompt handling (native) | Action chunk | Parameters (incl. text encoder, VAE) | Load s |
|---|---|---|---|---|---|---|---|---|---|
| FastWAM | FastWAM @ `7faa711` | `libero_uncond_2cam224.pt` | torch.bfloat16 | 10 | infer_action + torch.compile (EVALUATION.compile_action_infer=true, the repo default) | T5 prompt encoding inside every query | [32, 7] | 12.41 B | 101 |
| FastWAM (compile_action_infer=false) | FastWAM @ `7faa711` | `libero_uncond_2cam224.pt` | torch.bfloat16 | 10 | infer_action (EVALUATION.compile_action_infer=false) | T5 prompt encoding inside every query | [32, 7] | 12.41 B | 96 |
| FasterWAM | FasterWAM @ `8366781` | `step_021700.pt` | torch.bfloat16 | 10 | one_pass_future_cache | T5 prompt encoding inside every query | [32, 7] | 11.94 B | 90 |
| Light-WAM (libero_10 release) | Light-WAM @ `db7d05e` | `step_080000.pt` | torch.bfloat16 | 10 | infer_action | pre-encoded once per task (EVALUATION.preencode_prompt_context=true) | [32, 7] | 7.67 B | 146 |
| LiLa-WAM (RoboTwin 2.0 release) | LiLa-WAM @ `339860f` | `checkpoint.pt` | bfloat16 | 10 | RobotWinInference._predict_chunk (DINOv3 features + flow steps + B-spline smoothing) | no language input; precomputed task-condition vector | [32, 14] | 0.52 B | 7 |

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
  - The success rates are in Section 6.
- **Noise.** Repeated runs on these GPUs can differ by about 5–10% (two latency levels, see the all-models report). Treat differences under about 10% as ties.

## 6. Success-rate context

None of these numbers come from the latency runs.

| Model | LIBERO success (ours, matched protocol) | Episodes | Notes |
|---|---|---|---|
| FastWAM (released) | **96.63%**: Long 93.53 (1403/1500), Goal 96.27 (1444/1500), Object 99.27 (1489/1500), Spatial 97.47 (1462/1500) | 6,000 (seeds 42/43/44) | the same checkpoint file timed here (sha256 match) |
| Light-WAM | Long only: **90.87%** (1363/1500); seeds 90.4 / 90.0 / 92.2 | 1,500 | the earlier evaluation scored `lightwam_libero_10/.../step_080000.pt`, the file timed here by name. Its sha256 was not recorded in that evaluation. |
| FasterWAM | not evaluated by us | – | |
| LiLa-WAM (RoboTwin release) | not applicable: a RoboTwin model | – | |

## 7. Caveats

- **Different query boundaries.** This report times each repo's own query function end to end. The all-models report times our adapters. `prompt_cached` uses the same text boundary, but preprocessing code differs between repos (PIL vs torch resizing, CPU vs GPU), so the numbers compare closely but not identically.
- **Peak GPU memory** is `torch.cuda.max_memory_allocated` during the timed queries. It includes resident encoders: T5-XXL for FastWAM and FasterWAM, and DINOv3 for LiLa-WAM.
- **Outliers on evc103.** A few queries on evc103 show outliers: FastWAM native max 114.9 ms; FastWAM uncompiled max 322–338 ms. They raise the std and p95 there, but medians agree across nodes within 5%.

## 8. Files and reproduction

- **GitHub, branch `elastic`:**
  - `plans/performance/public_wam_latency_20261030/`: result JSONs with per-query times, the harness, `env.sh`, `run_models.sh`, `build_tables.py`, `make_report.py`, the FasterWAM official-script output, checkpoint hashes and node logs.
- **Cluster:**
  - code clones: `/groups/yshang/an221229/code/wam_baselines_20261030/`;
  - weights and latency: `/groups/yshang/an221229/checkpoints/wam_baselines/`.
- **Reproduce latency:** `source env.sh; bash run_models.sh <tag> fastwam fastwam_nocompile fasterwam lightwam; bash run_lilawam.sh <tag>`, then `make_report.py <out.md>`.
