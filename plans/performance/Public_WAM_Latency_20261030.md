# Public WAM baselines: batch-one LIBERO inference latency (2026-10-30)

FastWAM, FasterWAM, Light-WAM and LiLa-WAM, each run through **its own public code with its own released weights**, unoptimized, on H100 80GB HBM3.
- **Data:** `plans/performance/public_wam_latency_20261030/`, with per-query timings in every result file.
- **Harness:** `measure_public_latency.py` and `measure_lilawam_latency.py` in the same folder.
- **Comparison report:** the all-models report for our own checkpoints is `LoopWAM_All_Models_Latency_20261030.md`.

## 1. Summary

- **Per-query latency on H100** (prompt context cached per task, the same text boundary as our matched protocol). Each figure is the mean over 500 queries, with node evc102 first and evc103 second:
  - Light-WAM: 49.4 / 49.8 ms (about 20 Hz), 4.3 GB peak GPU memory.
  - LiLa-WAM: 49.5 / 50.8 ms, 1.1 GB. This is its RoboTwin release; no LIBERO model exists.
  - FastWAM, as its repo runs it with `torch.compile`: 59.3 / 56.5 ms, 25.0 GB.
  - FasterWAM: 180.8 / 183.7 ms, 24.1 GB.
  - FastWAM with its compile flag off: 232.5 / 233.7 ms.
- **Natively, exactly as each repo's LIBERO eval runs**, the times are as follows (evc102):
  - FastWAM 73.9 ms;
  - FasterWAM 193.1 ms;
  - Light-WAM 49.1 ms;
  - LiLa-WAM 49.5 ms.

  FastWAM and FasterWAM re-encode the prompt with T5-XXL inside every query, which costs 12–18 ms per query. Light-WAM pre-encodes it once per task.
- **Compile is the biggest single factor.** FastWAM's public LIBERO config turns on `torch.compile`, which makes it 3.9–4.1× faster than its own uncompiled path. FasterWAM has no compile option.
  - Uncompiled vs uncompiled, FasterWAM is 1.29× faster than FastWAM: 180.8 vs 232.5 ms. Its one-pass future cache shortens the action branch.
  - FasterWAM's own official latency script independently agrees. It times random-initialized architectures without the text encoder, and reports FasterWAM 178.5, FastWAM 244.7 and Joint-WAM 427.9 ms.
- **Reliability checks.**
  - The two nodes agree within 5% for every model.
  - Caching the prompt never changed an action: identical in 500/500 queries for every model.
  - Every checkpoint matches its published sha256.
- **Memory.** FastWAM and FasterWAM keep the 5.7B-parameter T5-XXL encoder on the GPU (about 25 GB peak). Light-WAM offloads T5 after pre-encoding (4.3 GB). LiLa-WAM is a 0.5B model with a DINOv3 ViT-L encoder and no language model (1.1 GB).

## 2. What was run

| Model | Public code | Released weights (verified sha256) |
|---|---|---|
| FastWAM | github.com/yuantianyuan01/FastWAM | `libero_uncond_2cam224.pt` (the released LIBERO checkpoint; sha256 `1000437c…`, the file our matched evaluation scored) |
| FasterWAM | github.com/hustvl/FasterWAM | HF `hustvl/FasterWAM` `libero/step_021700.pt` (rev `6bf9471c`; sha256 matches HF LFS) |
| Light-WAM | github.com/L1ziang/Light-WAM | HF `l1ziang/lightwam-checkpoints` `lightwam_libero_10/.../step_080000.pt` (rev `7cc8593f`; sha256 matches HF LFS) |
| LiLa-WAM | github.com/teee000/LiLa-WAM | ModelScope `yangfan97/LiLa-WAM_RoboTwin2_0` `checkpoint.pt` (sha256 `40e8aba0…` matches ModelScope), DINOv3 ViT-L/16 from HF |

**Light-WAM** releases one checkpoint per LIBERO suite. All four share one architecture and one config apart from training steps, so the LIBERO-Long (`libero_10`) checkpoint was timed.

**LiLa-WAM** releases **no LIBERO checkpoint and no LIBERO code** (no LIBERO config, dataloader or evaluation). Its only public weights are the RoboTwin 2.0 model:
- one 320×240 head camera, a 16-D dual-arm state and 32×14 actions;
- its row therefore times that release on RoboTwin-shaped inputs, not a LIBERO policy;
- as agreed, it has no LIBERO success evaluation.

## 3. Protocol

- **Inputs.** All models replay the same 500 recorded raw LIBERO observations as the all-models report (sha256 `5815c6fd…`; 125 consecutive steps of task 0 in each of the four suites). Every model runs 20 warmup queries, then all 500 timed queries in order.
- **What one query is.** The query is exactly the repository's own per-replan call:
  - FastWAM, FasterWAM and Light-WAM: `experiments/libero/eval_libero_single._predict_action_chunk`. It takes the raw simulator observation through the repo's own image and proprio preprocessing, its native inference method and its denormalization, and returns host LIBERO actions.
  - LiLa-WAM: `RobotWinInference.update_state_buffer` + `_predict_chunk`, which preprocesses the input, extracts DINOv3 features, runs ten flow steps, denormalizes and applies B-spline smoothing. That is what its `step()` runs when it re-plans.
  - In every case `torch.cuda.synchronize()` runs before the clock starts, and the clock stops when the host action array exists.
- **Model construction.** Models are built exactly as each repo's `eval_single_process` builds them: its Hydra `sim_libero` config, `instantiate(cfg.model, model_dtype, device)`, its `_load_model_checkpoint`, and its processor with the release's dataset stats. Light-WAM's eval also merges the checkpoint's training config, and so did this run.
  - No settings were changed, except that `model.redirect_common_files=false` makes FastWAM/FasterWAM load the original Wan2.2 `.pth` T5/VAE files instead of their safetensors conversions from DiffSynth-Studio. The weights are the same; only the file format differs.
  - Each model uses the repo's own seed and noise settings.
- **Prompt modes.**
  - `native`: whatever the repo's LIBERO evaluation does.
    - FastWAM and FasterWAM re-encode the language prompt with the 5.7B-parameter T5-XXL encoder **inside every query**.
    - Light-WAM pre-encodes it once per task (`EVALUATION.preencode_prompt_context=true`, its default).
  - `prompt_cached`: the T5 context is computed once per task outside the clock, which is our matched protocol's boundary. This is done by memoizing the model's own `encode_prompt` on the prompt string. The memoized function returns the same tensors, and every action chunk was compared with native: identical in 500/500 queries for every model.
- **FastWAM's compile.** FastWAM's public `sim_libero.yaml` sets `EVALUATION.compile_action_infer: true`, so its native path is torch.compile'd. A second FastWAM row turns this off through the same config key, to show the uncompiled cost. FasterWAM and Light-WAM have no compile path.
- **Two nodes.** Two of your interactive jobs each had one H100: 1094057 (evc102) and 1094056 (evc103). Every model was measured on both, one model at a time per GPU. The two columns are independent repeats on different hardware.
- **Environment.**
  - Software: the FastWAM conda env (torch 2.7.1+cu128).
  - LiLa-WAM: needs transformers 5.0.0rc0 for DINOv3, which comes from an isolated overlay (`envs/lilawam_venv`); the env itself is unchanged.
  - LIBERO is imported only for its benchmark module; no simulator runs.

## 4. Results

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

### FasterWAM's official latency script (cross-check)

`experiments/latency/measure_latency.py` was run unmodified with its defaults on evc103. It builds random-initialized architectures from the repo configs, feeds synthetic inputs and excludes the text encoder: 5 warmup and 10 timed iterations, BF16, batch 1. Its full report is in `official_fasterwam_script/`.

| Model | VAE encode | Visual branch | Action branch | Model infer (ms) |
|---|---:|---:|---:|---:|
| Joint-WAM | 4.8 | N/A | N/A | 427.9 |
| Fast-WAM (no compile) | 4.7 | 20.8 | 214.9 | 244.7 |
| Faster-WAM | 4.8 | 22.1 | 147.2 | 178.5 |

## 5. Reading the numbers

- **Which number to quote.**
  - Use `native` to answer "how fast does the released system run its own LIBERO eval".
  - Use `prompt_cached` to compare with our models: the all-models report excludes T5 the same way, since our evaluations use a T5 cache.
- **Comparison with our checkpoints.** These figures come from the all-models report, on the same 500 observations and H100s, with T5 cached and our exact `fast` backend:
  - Elastic E0 at [1,1]: 24.7 ms, 94.33% four-suite success;
  - F0-PB at [4,4]: 37.0 ms, 95.88%;
  - all-suites v0 4/4: 99.3 ms, 94.92%;
  - the released FastWAM checkpoint through our bit-exact path: 107.5 ms, 96.63%.
- **FastWAM through our harness vs its own default.** Our harness gives 107.5 ms, while FastWAM's public default gives 56–59 ms. Two things differ:
  - the public default stores weights in BF16 and runs `torch.compile` (Inductor);
  - our harness keeps FP32-stored weights under BF16 autocast, the setup our matched evaluation scored, and requires bit-exact actions.

  The all-models report found Inductor compile not bit-exact (0/100), so the public default's actions are not identical to the scored ones.
- **Success context** (earlier matched evaluations, not these runs):
  - FastWAM 96.63% (6,000 episodes, four suites);
  - Light-WAM 90.87% on LIBERO-Long only (1,500 episodes, earlier campaign);
  - FasterWAM: not evaluated by us;
  - LiLa-WAM: no LIBERO model has been released. Training it on our LIBERO data was requested on 2026-10-30, as a separate follow-up.

## 6. Caveats

- **Different query boundaries in the matched protocol.** This report times each repo's own query function end to end. Our matched-protocol latencies time our own adapters with the T5 cache excluded. `prompt_cached` uses that same text boundary, but preprocessing code differs between repos (PIL vs torch resizing, etc.), so the numbers compare closely but not identically with the all-models report.
- **Success rates come from earlier matched evaluations, not from these runs.**
  - FastWAM: 96.63% over 6,000 episodes (four suites, seeds 42/43/44).
  - Light-WAM: 90.87% on LIBERO-Long only, over 1,500 episodes, from the earlier campaign. That campaign's checkpoint identity was not re-verified against this release.
  - FasterWAM: not evaluated by us.
  - LiLa-WAM: no LIBERO model exists.
- **LiLa-WAM inputs.** LiLa-WAM was timed on RoboTwin-shaped inputs built from the LIBERO frames: the agent-view image as head camera, and the end-effector pose for both arms. Values do not change its computation, but its single 320×240 camera differs from the other models' two 224×224 cameras.
- **Peak GPU memory** is `torch.cuda.max_memory_allocated` during the timed queries. It includes the resident T5-XXL encoder where the repo keeps it on the GPU. Light-WAM's eval moves T5 to the CPU after pre-encoding.
