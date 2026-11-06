# ChronoLoop CL-A on RoboTwin-10 and RoboDojo: 2-GPU training-speed benchmark (October 15, 2026)

Branch `chrono`, worktree `/lustre/fs1/home/an221229/code/LoopWAM_chrono` (uncommitted changes, listed in section 7).
Interactive job 916782 (evc102), **GPUs 2 and 3 only**. GPUs 0 and 1 belong to your other run in the same job.
This is a benchmark, not training: each run did 45–60 real updates and saved nothing.

## 1. Result

Both runs use CL-A exactly as in plans/ChronoLoop.md. The only changes are the ones each robot dataset forces (section 3).
Best measured configuration for each dataset: RoboTwin-10 with the CPU/allocator settings of section 4 (+5.6%);
RoboDojo baseline (the same settings measured equal, 36.2 vs 36.8 windows/s on identical updates).
Steady state = updates after 15 (updates 1–3 are torch.compile); losses finite throughout.

| | RoboTwin-10 (LiLa-WAM subset) | RoboDojo (official train set) |
|---|---|---|
| Episodes / windows per epoch | 5,500 / 1,163,351 | 3,500 / 1,856,102 |
| Updates per epoch (128 slots each) | 10,775 | 15,405 |
| Valid windows per update (schedule mean) | 108.0 (84.4% of slots) | 120.5 (94.2%) |
| s / update, mean (median) | 3.19 (2.96) | 3.30 (3.14) |
| **Throughput, valid windows / s** | **34.1** | **36.5** |
| **Time per epoch, 2 x H100** | **9.5 h** | **14.1 h** |
| 10 epochs (the CL-A LIBERO budget) | 94.7 h (3.9 days) | 141.5 h (5.9 days) |
| FastWAM-matched budget | 5 epochs (RoboTwin recipe): 47.3 h | 5.12 M samples (batch 256 x 20K steps): 39.0 h |
| Peak memory per GPU | 26.7 GB | 28.0 GB |
| GPU util / power (steady) | 92% / 610 W | 88% / 600 W |

Projections divide each budget's valid windows by measured windows/s; masked slots are dropped, so time scales with
valid windows. The s/update projections agree within 1% (95.4 h and 141.3 h for 10 epochs).
Not included: the one-time latent cache build (section 5), startup/compile (~4–6 min per job), and checkpoint saves.

Baseline vs optimized on identical updates (same windows, same noise; losses agree within 8e-4, BF16 noise):

| | RoboTwin-10 base | RoboTwin-10 optimized | RoboDojo base | RoboDojo optimized |
|---|---|---|---|---|
| Updates compared | 16–50 | 16–50 | 16–45 | 16–45 |
| s/update mean / median / p90 | 3.37 / 3.01 / 4.54 | **3.19 / 2.96 / 3.66** | 3.27 / 3.12 / 3.75 | 3.33 / 3.10 / 3.77 |
| Valid windows / s | 32.3 | **34.1 (+5.6%)** | 36.8 | 36.2 (−1.5%, noise) |
| GPU util | 87% | 92% | 88% | 90% |

Optimized = `--text-threads 4` (T5-context loader threads per rank, was 16) and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It only helps RoboTwin, whose per-frame instructions make every
window load its own T5 file; RoboDojo has 35 prompts, all cached after the first update. Use it for both.

## 2. What runs

- **Model:** ChronoLoop CL-A — 16 memory tokens, learned source, loop write (U1, G-a gate), K_a = 4, TBPTT S = 4,
  memory horizon = whole episode. Backbone is LoopWAM-S v0 (3 pre + 6 core x 4 loops + 3 coda = 30 virtual layers).
- **Init:** the full-suite LIBERO v0 4/4 parent (`$CHRONO_PARENT`, the CL-A parent). Every backbone tensor is kept.
  The 14-D action encoder (14→512), action head (512→14) and proprio encoder (14→4096) are new (seeded nn.Linear
  default init), the same robot adaptation as your RoboTwin v0 setup. Trainable: 584,567,886 + 31,008 memory.
- **Optimizer / schedule:** CL-A's: AdamW (0.9, 0.95), clip 1.0, backbone LR 3e-5, memory LR 3e-4, 3% warmup then
  cosine to 1%. The new robot interfaces train at 3e-4 (`--lr-robot-io`, same rule as the new memory parameters).
  LR does not change speed; pick it before a real run.
- **Batch:** 128 window slots per update = 32 streams x 4 consecutive windows; 2 GPUs x 16 streams; one micro-batch.
- **Speed settings (chrono production):** torch.compile of each block function, per-block activation checkpointing
  (needed: TBPTT keeps the graph of 4 windows), BF16 autocast with FP32 master weights, fused AdamW, DDP with
  bucket views, masked slots dropped, decode-free data (frozen-VAE latents + in-memory parquet columns).

## 3. Data protocol

Both datasets are LeRobot v2.1, 3 cameras at 480x640 (AV1), 14-D absolute joints. Both use FastWAM's RoboTwin input
protocol (`configs/data/robotwin.yaml`, your v0 adapter `loopwam_robotwin.py`): every frame is an anchor; 9 frames
at offsets 0,4,…,32 and 32 actions; cameras → 240x320, high camera 256x320 over the two wrists at 128x160 = 384x320;
repeat-last padding with masked losses; global z-score; prompt = the anchor frame's task.

**RoboTwin-10.** LiLa-WAM (arXiv 2608.03701) Sec. 4.2: "a subset of 10 representative tasks": adjust_bottle,
grab_roller, hanging_mug, move_stapler_pad, open_microwave, press_stapler, scan_object, stack_blocks_two, stamp_seal,
turn_switch. The paper trains on "50 clean demonstrations and 500 randomized demonstrations" per task, jointly; the
FastWAM release holds exactly these 550 per task, in alphabetical task blocks (episodes 0–49 of a block are clean,
50–549 randomized; checked on first frames and instructions). Normalization: the published FastWAM statistics.
(The LiLa-WAM repo's ablation folder `data-200-10` suggests 50 clean + ≤200 randomized per task for the ablation
tables; the paper does not say. With that data the times above shrink by ~2.2x.)

**RoboDojo.** `RoboDojo-Benchmark/RoboDojo` `data/RoboDojo_lerobot_v21_video`, the official training set (35 task
directories: 34 tasks + the DLC task, 100 trajectories each, 25 fps, ARX X5 bimanual, mean episode 530 frames).
Normalization: global mean/std over all frames (no published stats in this export). T5 prompts were encoded for
the 35 task strings (`text_embeds_cache/robodojo`).

**Replanning stride 24** for both, the stride of the TBPTT traversals (plan 3.2): FastWAM's RoboTwin eval uses
`replan_steps: 24` (`configs/sim_robotwin.yaml`), and the XPolicyLab FastWAM adapter for RoboDojo uses the same task
config with `replan_steps 24`. LIBERO used 10.

## 4. Speed work and what it found

| Option | Result |
|---|---|
| torch.compile per block | kept (chrono: 1.8–2.2x). Compile warm-up: 3 updates, 3.5–6 min |
| 16 streams per GPU in one micro-batch | kept (largest GEMMs; 28 GB of 80 GB used) |
| Decode-free latents | kept; measured data wait 1.3 s over 60 updates (prefetcher keeps up) |
| No activation checkpointing | not possible: CL-A's TBPTT holds 4 windows x 16 streams of graph per GPU |
| cuDNN attention instead of Flash-2 | dropped: torch.compile keeps dispatching Flash; interleaved eager timings were within noise |
| Keep activations of the last 6 of 30 layers (`--uncheckpointed-layers 6`) | out of memory (80 GB): over 9 GB per layer |
| 4 T5 loader threads + expandable-segments allocator | kept: RoboTwin-10 +5.6% windows/s, p90 4.54 → 3.66 s; RoboDojo unchanged |

## 5. Before a real run: the latent cache

Training reads frozen-VAE latents; the benchmark built only the windows its updates read (7,216 and 7,982).
A full cache is one 9-frame clip per frame:

| | RoboTwin-10 | RoboDojo |
|---|---|---|
| Cache size (184 KB per window) | 214 GB | 342 GB |
| CPU per window (1 thread): AV1 decode / resize chain, 3 cameras | 6.3 / 20.2 ms | 8.6 / 18.3 ms |

| Frozen Wan2.1 VAE encode, 1 H100 (batch 16 or 32, 34 GB peak) | 30 windows/s | 30 windows/s |
| **Full cache build, 2 x H100 (VAE-bound)** | **5.4 h** | **8.6 h** |

In the full build each frame is decoded and resized once for one window (~27 ms of CPU per window).
About 2 hyperthreads therefore keep both GPUs fed, and the build is VAE-bound. The benchmark lists were CPU-bound
(~10 windows/s) only because they are sparse: every window needs its own decoded frames.

- Stage the cache in RAM (`/dev/shm`, as the LIBERO runs do). Reading 184 KB windows at random from Lustre starved
  the LIBERO runs. 214 GB or 342 GB fits on a 768 GB node, but not in the 256 GB that `chronoloop_job.sbatch` requests.
- This job has 8 hyperthreads, shared with your GPU 0/1 run. The cache build is CPU-bound here (~10 windows/s
  for the sparse benchmark lists); build it in a job with more CPUs.

## 6. Checks

- **Fields.** On 10 windows from two episodes (hanging_mug clean 5503, open_microwave randomized 9410), including
  episode-end padding, the decode-free fields equal your LeRobot RoboTwin v0 adapter exactly: action, proprio,
  all three pad masks, prompt, T5 context and mask. The preprocessed 384x320 clips are **bit-identical** (max |Δ| = 0).
- **Model.** Backbone 584,567,886 = 584,536,135 + 31,751 (the 14-D interfaces); memory 31,008, the same as LIBERO CL-A.
- **Training.** All updates finite. Fresh action head: action loss 2.6 → 2.15, video loss 0.25 → 0.18, within 60
  updates, at warm-up LR. Gradient norm 12–14 is clipped to 1.0.
- **Schedules.** 10-epoch traversal files built once (seed 42, 32 streams, S = 4, stride 24); every (demo, window)
  appears once per epoch.

## 7. Files and commands

New: `src/fastwam/datasets/chronoloop_robot.py`, `chronoloop_robot_tasks.py`,
`src/fastwam/models/wan22/chronoloop_robot.py`, `scripts/train_chronoloop_robot.py`,
`scripts/chronoloop_robot_precompute.py`, `scripts/chronoloop_robot_benchmark.sh`,
`scripts/chronoloop_robot_bench_report.py`. Changed, defaults unchanged for LIBERO: `chronoloop_data.py`
(stride argument), `chronoloop.py` (`uncheckpointed_virtual_layers`, default 0).

Artifacts: `/groups/yshang/an221229/checkpoints/ChronoLoop/robot/{robotwin10,robodojo}/` (schedules, data manifest,
stats, partial latent caches), `bench/<run>/` (metrics.jsonl, gpu.csv, bench_summary.json), `logs/`.

```bash
source scripts/chrono_env.sh
python scripts/train_chronoloop_robot.py --dataset robotwin10 --run-name prep --output-dir X --prepare-only 60 \
  --memory-tokens 16 --mem-source learned --mem-write loop            # schedule + benchmark window list (CPU)
torchrun --standalone --nproc_per_node=2 scripts/chronoloop_robot_precompute.py --dataset robotwin10 \
  --shared-dir /groups/yshang/an221229/checkpoints/ChronoLoop/robot/robotwin10 [--windows-file needed_windows_u60.npy]
bash scripts/chronoloop_robot_benchmark.sh robotwin10 base 2,3      # or robodojo; extra flags pass through
# real run (after a full cache): torchrun ... train_chronoloop_robot.py --dataset robotwin10 --run-name CL-A \
#   --memory-tokens 16 --mem-source learned --mem-write loop --action-loops 4 --stream-micro 16 --output-dir DIR
```

## 8. Caveats

- 45 steady updates per run, from the start of the schedule. Valid windows per update matched the 10-epoch
  schedule means (109.0 vs 108.0; 120.4 vs 120.5). The run-to-run spread is not measured.
- CPU: 8 hyperthreads shared with another 2-GPU run. RoboTwin's per-frame instructions make each window load its own
  T5 context (~150 KB file); its slower steps (p90 4.5 s vs RoboDojo 3.8 s) are consistent with CPU contention.
- No evaluation: there is no RoboTwin or RoboDojo harness for ChronoLoop yet, and RoboDojo needs Isaac Sim on RTX GPUs.
