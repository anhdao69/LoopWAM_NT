# Full-suite v0 latency and robustness evaluation — October 7, 2026

## Resource assignment

LIBERO-Pro is running in existing interactive allocation **4728**, worker-1, step **4728.9**, on **two H100 80 GB GPUs**. LIBERO-Plus is queued behind Pro in the same allocation. Queue watcher PID 802548 runs independently of this SSH session. It checks allocation ownership and availability, waits for successful Pro completion and step exit, rejects unexpected active steps, then launches Plus with two GPUs. Pro or allocation failure stops the queue and records the error. Plus performs its own category smoke checks before full evaluation. SimpleMemVLN remains running on worker-3; our Plus step there was cancelled during text preparation.

The allocation must remain active for the queue to proceed. Outputs are under `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/`: `libero_pro_job4728_20261007` and `libero_plus_job4728_after_pro_20261007`. Videos remain on the server.

## Completed native LIBERO results: FastWAM vs LoopWAM v0

| Model / source | Spatial SR | Object SR | Goal SR | Long SR | Average SR | Mean query latency, H100 |
|---|---:|---:|---:|---:|---:|---:|
| Original FastWAM, published | 98.2% | 100.0% | 97.0% | 95.2% | 97.6% | 279.90 ms, measured here |
| Full-four-suite LoopWAM v0, measured here | 99.0% (99/100) | 99.0% (99/100) | 94.0% (94/100) | 96.0% (96/100) | **97.0% (388/400)** | **260.26 ms**, measured here |

FastWAM SR is from the original model's [official project results](https://yuantianyuan01.github.io/FastWAM/#results-on-simulation-and-real-world-tasks), checked October 7, 2026. It is an external published baseline; we have not run a matched four-suite FastWAM simulator evaluation in this experiment. The newer Optional-IDM checkpoint's first-frame scores in the upstream README belong to a different checkpoint and are not substituted for the original unconditional release measured for latency.

LoopWAM differences relative to that published baseline are +0.8, −1.0, −3.0 and +0.8 percentage points for Spatial/Object/Goal/Long; average −0.6 percentage points. Episode counts, seeds and evaluation protocols differ, so these differences do not establish a statistically reliable ranking. Our native evaluation uses ten episodes per task, seed 42 with deterministic task/episode offsets, 30 settling steps, maximum 700 policy steps, terminal-aware success handling, four loops and ten diffusion steps. The shorter benchmark-specific horizons above apply to Pro/Plus, not to these completed native results.

The final native pipeline is **complete**. Training took **12 h 6 m 20.85 s** on four H100s; the four final suite evaluation stages took **1 h 23 m 22.56 s** combined. Including smoke checks and orchestration, pipeline runtime was **13 h 34 m 55.53 s**. All 400 episode keys match the exact 40-task × ten-trial grid. All episode checkpoint hashes match the final update-21700 checkpoint, and all 400 videos exist and are nonempty on the server. Raw final summary and pipeline timing are synced under `plans/evidence/full_four_suite_20261007/`; videos stay on the server.

The measured mean latency reduction is **7.01%**, or **1.075×** throughput per query. Both models were measured on one H100 using the protocol below, with 100 timed queries each. The FastWAM release checkpoint is `libero_uncond_2cam224.pt` (SHA-256 `1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`). Compilation was disabled for both models: these figures do not compare upstream's newest compiled deployment path. The measured latency comparison and external SR comparison have distinct provenance.

## Measured latency

Two independent trials, each with five warmups and 50 synchronized measurements per model, on the same single H100. Batch 1, two 224×224 cameras, 32-action chunks, ten diffusion steps, cached text, FP32 weights with BF16 autocast. Timing includes image preprocessing, device transfers, online VAE encoding, visual prefill, action denoising and CPU output transfer; excludes simulator, video encoding, model loading and text encoding. No compilation or latent caching.

| Model | Trial 1 mean ms | Trial 2 mean ms | Combined mean ms | Median ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| FastWAM release | 265.66 | 294.13 | 279.90 | 273.13 | 333.99 |
| Full-suite LoopWAM v0 | 263.69 | 256.83 | 260.26 | 257.18 | 290.95 |

These are policy-query latencies, each producing 32 actions. Native architectures and VAEs differ. Host workload and trial variation limit the interpretation of the approximately 7% difference in combined means. This is not a universal speedup claim. Raw timings and aggregate are in `plans/evidence/robustness_20261007/`.

## Checkpoint and evaluation protocol

Both benchmarks use the completed four-suite v0 checkpoint at training update **21700**, with SHA-256 `4ffe1b14f35484ea582189ba0f06c5a41ac199e1fa733b491719727105e5e5cc`. Training was ten epochs, global batch 128. Training normalization is retained. Evaluation uses four loops, ten diffusion steps, 32-action chunks, replanning every ten actions, 30 settling steps and suite horizons 220/280/300/520. Every episode saves a video. Simulator seeds are deterministic. Results must pass exact episode coverage, unique episode keys, matching checkpoint hashes and video existence checks before aggregation.

Pro: official [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO), revision `eafdb809426b13153aa1e4c42d6601844217dfec`; five perturbation dimensions × four suites × ten tasks × ten episodes = **2000 episodes**. The published [asset dataset](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro) supplies four dimensions. Its missing environment dimension was generated using the official environment BDDL transformation and ten simulator reset states per task, seeds 42–51. Policy instructions are read from the actual perturbed BDDL. All five dimensions passed smoke evaluation on both ranks.

Plus: official [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus), revision `4976dc30028e805ff8094b55501d532c48fec182`; **10030 tasks/episodes**, seven perturbation categories across all four suites. Official task classification and initial-state remapping are retained. All required assets were extracted and isolated dependencies loaded successfully. Its simulator smoke evaluation is pending the queued launch; the full run starts only after passing those checks.

Production code is pinned to `2b7d2c3606c26cccd2f67f6367b06b33c735c2bc`. Five aggregation integrity tests passed. Shared training dependencies and the original LIBERO installation were left intact.

## Initial runtime estimate

At approximately **00:05 EDT October 7**, Pro had completed 5/2000 episodes: a 280-step episode took 46 seconds and a 520-step episode took 81 seconds. These first tasks are insufficient for a precise ETA. Pro is provisionally **14–23 hours**, approximately afternoon/evening October 7 EDT. Plus will then start automatically: provisionally **70–120 hours on two GPUs**, including its greater per-task environment setup overhead, approximately October 10–13 EDT including the Pro wait. Plus has not started, so this estimate is especially uncertain. Replace these estimates with observed throughput once task and category coverage expands. No final robustness success rates are available yet.
