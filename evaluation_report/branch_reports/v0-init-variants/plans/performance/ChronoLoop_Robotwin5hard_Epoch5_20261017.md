# ChronoLoop CL-0 and CL-A on 5 hard RoboTwin 2.0 tasks: training to epoch 5 (October 16–18, 2026)

Branch `chrono`, worktree `/lustre/fs1/home/an221229/code/LoopWAM_chrono`. Code snapshot used by every run:
`/groups/yshang/an221229/checkpoints/ChronoLoop/code/54b7c805c3bb` (commit 54b7c80).
Weights: https://huggingface.co/anhdao69/ChronoLoop_Robotwin5 (section 7). RoboTwin 2.0 evaluation: section 8.

## 1. Result

Both runs reached the end of epoch 5 (update 35,111 of the 10-epoch schedule) with no non-finite step, and saved
`epoch_05.pt`: CL-0 on October 17 at 20:36, CL-A on October 18 at 01:51.

| Run | Epoch-5 training loss (video / action) | Training time to update 35,111 | Seconds per update (2 H100s) |
|---|---|---|---|
| CL-0 (no memory) | 0.0623 / 0.0067 | 26.2 h | 2.63-2.69 |
| CL-A (mem16, learned source, loop write) | 0.0618 / 0.0051 | 31.6 h | 3.13-3.33 |

RoboTwin 2.0 evaluation of both `epoch_05.pt` (section 8): unseen instructions, 50 episodes per task and setting,
1,000 episodes in all. Mean success over the 5 tasks:

| Run | Clean | Randomized |
|---|---:|---:|
| CL-0 (no memory) | 55.2% | 50.0% |
| CL-A (mem16, learned source, loop write) | **57.6%** | **53.1%** |

On the scenes both models ran (220 per setting), CL-A is 1.3 points higher on Clean (56.3% vs 57.6%) and 5.4 points
higher on Randomized (49.3% vs 54.7%). Neither difference is statistically significant: on matched scenes, CL-A
alone succeeded 34 times against CL-0 alone 31 times on Clean (exact McNemar p = 0.80), and 28 against 17 on
Randomized (p = 0.14). By task, handover_mic is solved by both (98-100%), and hanging_mug is the hardest
(24-30%). CL-A's largest gains on matched scenes are place_can_basket Randomized (+14 points), scan_object
Randomized and hanging_mug Clean (+9 each); CL-0 is ahead on two matched cells, by one scene each.

## 2. What was trained

- **Tasks (`robotwin5hard`):** hanging_mug, blocks_ranking_size, scan_object, place_can_basket, handover_mic. These are
  the five RoboTwin 2.0 tasks with the lowest trimmed-mean success in LiLa-WAM (arXiv 2608.03701) Table 8: for each
  task, the mean of the ten clean/randomized success rates of pi0.5, Motus, X-VLA, GigaWorld-Policy and LiLa-WAM, with
  the best and worst dropped (27.6, 63.5, 65.0, 69.5, 72.2). Defined in `src/fastwam/datasets/chronoloop_robot_tasks.py`.
- **Data:** the FastWAM RoboTwin release, 550 demonstrations per task (50 clean + 500 randomized, the LiLa-WAM
  protocol): 2,750 episodes, 792,168 windows. Input protocol as in the October 15 benchmark report
  (`ChronoLoop_Robot_Benchmark_20261015.md`, section 3): 9 frames at offsets 0..32 step 4 plus 32 actions, three
  cameras packed into 384x320, 14-D joints with global z-score, per-frame instruction, replanning/TBPTT stride 24.
- **Runs:** CL-0 (no memory; stateless, the four windows of a stream are batched) and CL-A (16 memory tokens,
  learned source, loop write; U1 update, G-a gate; TBPTT S = 4). Both use K_a = 4 and start from the full-suite
  LIBERO v0 4/4 parent with new 14-D action encoder, action head and proprio encoder.
  Trainable: 584,567,886 backbone (+ 31,008 memory for CL-A).
- **Optimizer and schedule:** AdamW (0.9, 0.95), weight decay 0.01, clip 1.0; LR backbone 3e-5, memory 3e-4, robot
  interfaces 3e-4; 3% warmup, cosine to 1% over the **10-epoch** schedule (70,220 updates; 32 streams x 4 windows
  = 128 slots per update, 88.1% valid). Epoch 5 ends at update 35,111, where the LR is still 52.9% of its peak
  (backbone 1.59e-5): the epoch-5 checkpoints are the midpoint of a 10-epoch run, not an annealed 5-epoch run.
- **Speed settings:** torch.compile per block, per-block activation checkpointing, BF16 autocast with FP32 master
  weights, fused AdamW, DDP; frozen-VAE latents staged in `/dev/shm`; `--text-threads 4`,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Two H100s per run (`--stream-micro 16`).

## 3. How it ran

| When (EDT) | What |
|---|---|
| Oct 16 08:42–11:04 | Schedule (seed 42, 32 streams, S = 4, stride 24) and the full latent cache: 792,168 windows, 136 GB, built in 2.3 h on 4 H100s (interactive job 926608). This subset resizes frames on the GPU (`ROBOTWIN_PREPROCESS`; 0.012% of BF16 values differ by one rounding step from the CPU chain). evc101 failed CUDA init (error 802) on about 1 in 6 process starts; the build retried. |
| Oct 16 09:01–09:11 | CL-0 smoke test and CL-A save/resume test (2 updates, save, resume at update 2 with the stream state, 1 more update; all finite), on the RoboTwin-10 benchmark cache. |
| Oct 16 15:56 / 16:01 | Slurm twins (2-GPU and 4-GPU per run, `chronoloop_robot_submit.sh`): the 2-GPU jobs started first, 935948 (CL-0) and 935950 (CL-A), both on evc103, and cancelled their 4-GPU twins 935949 and 935951 within seconds. |
| Oct 17 15:46–15:47 | Both jobs reached their 24 h limit. CL-A stopped cleanly at update 26,601 and saved. CL-0 was killed while writing its final checkpoint at update 31,760 and resumed later from update 31,500 (section 6). Each job queued a 24 h continuation. |
| Oct 17 15:46–16:02 | The continuations (950060, 950065) could not start: the `yshang` account had used 170,769 of its 180,000 GPU-minutes for October (section 6). They were cancelled. |
| Oct 17 17:43 | Both runs resumed in interactive job 930602 (evc101, 4 H100s, 8 CPUs): CL-0 on GPUs 0–1, CL-A on GPUs 2–3, each with 2 GPUs exactly as in its Slurm job, from the same code snapshot, with `--max-updates 35111` (end of epoch 5). One shared RAM copy of the cache (staged in 180 s). |
| Oct 17 20:36 | CL-0 reached update 35,111, saved `epoch_05.pt` and exited. Its weights were uploaded and verified at 21:16. |
| Oct 17 21:48 - Oct 18 00:11 | CL-0 RoboTwin evaluation, 500 episodes, 10 processes on GPUs 0-1 next to CL-A's training (section 8). CL-A slowed from 3.2 to about 3.6 s per update while it ran (CPU contention; 8 CPUs). |
| Oct 18 01:51 | CL-A reached update 35,111 and saved `epoch_05.pt`; the trainer exited at 01:52 and the RAM cache was removed. |
| Oct 18 01:53 - 03:36 | CL-A RoboTwin evaluation, 20 processes on 4 GPUs (episodes from 02:03; the shared T5 server took 10 min to import on a slow Lustre metadata server). CL-A's weights were uploaded and verified at 01:57. |

## 4. Training curves

Per-epoch means over all updates of the epoch (`metrics.jsonl`; for CL-0's duplicated updates 31,501-31,760 the
later record is used). CL-A's memory parameters at the last update of each epoch: λ = sigmoid(a), the retention of
the memory update (mean over channels, initialized at 0.90), and |tanh α|, the per-block, per-head gates on the
memory read in the video and action streams (initialized at 0, mean over blocks and heads).

| Epoch | Updates | CL-0 video | CL-0 action | CL-A video | CL-A action | CL-A grad norm | CL-A λ | CL-A \|tanh α\| video / action |
|---|---|---|---|---|---|---|---|---|
| 1 | 1-7,025 | 0.1014 | 0.1139 | 0.1014 | 0.1139 | 0.817 | 0.900 | 0.023 / 0.031 |
| 2 | 7,026-14,046 | 0.0756 | 0.0170 | 0.0756 | 0.0170 | 0.264 | 0.897 | 0.026 / 0.045 |
| 3 | 14,047-21,067 | 0.0691 | 0.0113 | 0.0690 | 0.0110 | 0.202 | 0.892 | 0.028 / 0.065 |
| 4 | 21,068-28,089 | 0.0651 | 0.0085 | 0.0648 | 0.0072 | 0.170 | 0.890 | 0.028 / 0.073 |
| 5 | 28,090-35,111 | 0.0623 | 0.0067 | 0.0618 | 0.0051 | 0.147 | 0.889 | 0.028 / 0.076 |

The two runs agree to four decimals for two epochs: the read gates start at zero, so CL-A starts as CL-0. The
action read gate opens slowly (|tanh α| 0.03 in epoch 1 to 0.08 in epoch 5), and from epoch 3 CL-A's action loss is
below CL-0's, 24% lower in epoch 5 (0.0051 vs 0.0067). The video loss differs by less than 1%. These are training
losses on the training windows, not held-out numbers.

## 5. Speed

Two H100s per run, 32 streams x 4 windows = 128 slots per update (88.1% valid). CL-0: 2.63-2.69 s per update,
peak 34.0 GB per GPU. CL-A: 3.13-3.33 s per update (sequential TBPTT over the 4 windows of a stream), peak 27.5 GB.
In the interactive job both shared 8 CPUs; CL-A ran at 3.2 s per update alone and about 3.6 s while CL-0's evaluation
used the other two GPUs. RoboTwin evaluation: about 11 GB of GPU memory per process plus 12 GB for the shared T5;
episodes are CPU-bound (SAPIEN physics, CuRobo, video encoding), 1-1.5 min for a success and 3.5-5 min for a failure
that runs to the step limit when 10 processes share 6 CPUs.

## 6. Problems found

**CL-0's first job was killed mid-checkpoint (fixed).** Job 935948 lost 260 updates (31,501–31,760, about
12 minutes); the previous checkpoint (update 31,500) was intact because saves write a temporary file and rename it.
Two causes:
1. The trainer counted its 24 h budget (`--time-limit-hours`) from after process-group init. That start took
   5 minutes on evc103 (slow Lustre imports), so its stop-and-save, meant for 15 minutes before the walltime, fell at
   10 minutes before, together with Slurm's warning signal (`--signal=B:USR1@600`).
2. The batch script traps USR1, and a trapped signal makes bash's `wait` return at once (status 138). The script then
   exited, and Slurm killed the trainer while it was writing `latest.pt.tmp`.

Fix (commit with this report; not yet exercised in a real job): `train_chronoloop_robot.py --deadline-unix` takes
the allocation end as an absolute time, and `chronoloop_robot_job.sbatch` passes it and keeps waiting after USR1
until the trainer exits (tested with a mock trainer: exit statuses 0 and 3 are returned after the signal). The new
interactive launcher passes the deadline too when the code snapshot supports it.

**Account GPU budget.** The `yshang` account has `GrpTRESMins gres/gpu=180000` (3,000 GPU-hours per month,
`PriorityUsageResetPeriod=MONTHLY`, no decay). On the afternoon of October 17 it had used 170,769 GPU-minutes (an221229
55,122; two other members 115,647), and running jobs held 8,324 more for their remaining walltime. A continuation
asks for 2 GPUs x 24 h = 2,880 GPU-minutes; with about 900 left it pends with reason `AssocGrpGRESMinutes` until the
reset on November 1, whatever start time `squeue --start` shows. A job that ends early returns its unused
reservation. Finishing epochs 6–10 needs about 3,200 (CL-0) + 3,800 (CL-A) GPU-minutes on 2 GPUs each.

**evc101 CUDA error 802.** Unchanged: about 1 in 6 process starts on evc101 fail CUDA init (NVRM assertions); every
launcher retries. The interactive start on October 17 did not hit it.

## 7. Files, checkpoints and commands

Hugging Face, https://huggingface.co/anhdao69/ChronoLoop_Robotwin5:
`<run>/epoch_05/{policy.pt,config.json}` (weights only, FP32), `<run>/{manifest.json,metrics.jsonl,timing.json}`,
`data/{dataset_stats.json,data_manifest.json}` (normalization and data protocol),
`eval/robotwin5hard_epoch05_results.json` (per-seed RoboTwin outcomes of both models), `README.md`.
Uploads are verified file by file against the local size and SHA-256 (`<run>/hf_upload.json`).

Cluster, `/groups/yshang/an221229/checkpoints/ChronoLoop/robot/`:
- `runs/robotwin5hard/<run>/checkpoints/epoch_05.pt`: the uploaded weights.
- `runs/robotwin5hard/<run>/latest.pt`: resume checkpoint at update 35,111, with optimizer, scheduler and stream state.
- Logs: `job_<id>.log` (Slurm) and `interactive_930602.log`.
- `robotwin5hard/`: schedule, data manifest and statistics, and the full latent cache (136 GB).
- `code/54b7c805c3bb`: the frozen code every run used.

`metrics.jsonl` has one record per update. CL-0's file also holds updates 31,501–31,760 twice: once from the killed
job and once after the resume. Keep the later record of each update; `timing.json` and the checkpoints are unaffected.

Code (branch `chrono`):
- `scripts/chronoloop_robot_interactive.sh` (new): runs one robot run on a GPU subset of an interactive allocation.
  It takes the same run lock as the Slurm jobs, retries error 802, shares the RAM cache copy, and stops before the
  allocation ends.
- `scripts/chronoloop_robot_job.sbatch` and `train_chronoloop_robot.py`: the walltime fix of section 6.
- `scripts/chronoloop_hf_upload.py`: the per-checkpoint `config.json` now records the dataset and the robot loader.

```bash
cd /lustre/fs1/home/an221229/code/LoopWAM_chrono
CODE=/groups/yshang/an221229/checkpoints/ChronoLoop/code/54b7c805c3bb
# what ran on October 17 (inside interactive job 930602)
setsid nohup bash scripts/chronoloop_robot_interactive.sh $CODE robotwin5hard CL-0 0,1 --max-updates 35111 &
setsid nohup bash scripts/chronoloop_robot_interactive.sh $CODE robotwin5hard CL-A 2,3 --max-updates 35111 &
# upload
python scripts/chronoloop_hf_upload.py --repo anhdao69/ChronoLoop_Robotwin5 --epochs 5 --require-all \
  --run-dir /groups/yshang/an221229/checkpoints/ChronoLoop/robot/runs/robotwin5hard/CL-A_mem16-learned-loopwrite_a4 \
  --folder CL-A_mem16-learned-loopwrite_a4
```

**Continuing to 10 epochs.** Both runs resume from `latest.pt` with no change of schedule: run the same launcher
without `--max-updates`, or submit `bash scripts/chronoloop_robot_submit.sh --dataset robotwin5hard --gpus 2 CL-0 CL-A`
once the account has budget. Commit first: the submit script freezes HEAD as a new code snapshot, which includes the
walltime fix. The run's resume contract does not include the code revision. The remaining 35,109 updates take about
26.3 h (CL-0) and 31.4 h (CL-A) on 2 H100s.

## 8. RoboTwin 2.0 evaluation

### 8.1 Protocol

The official RoboTwin 2.0 harness (`script/eval_policy.py`, with FastWAM's vendored `eval_policy.py` and
`_base_task.py`), aloha-agilex, D435 cameras (320x240), CuRobo expert planner:

- **Settings.** `demo_clean` (Easy: plain table and background) and `demo_randomized` (Hard: random background
  textures, cluttered table, random lighting, table height +-3 cm). Training saw both (50 clean + 500 randomized
  demonstrations per task).
- **Episodes.** Up to 50 per task and setting, LiLa-WAM's count (arXiv 2608.03701, Tables 1 and 8). Seed 0: the
  harness tries scene seeds 100000, 100001, ..., keeps a seed only if the expert planner solves it, then runs the
  policy on that scene. A seed is the same scene for every model, but the expert check is not exactly repeatable:
  CuRobo solved some borderline seeds in one run and not in the other (e.g. place_can_basket/clean seed 100002:
  accepted for CL-0, rejected for CL-A), so the two runs' seed lists differ by a few seeds. Models are compared on the
  seeds both ran ("matched scenes").
- **Instruction.** One sentence per episode from the task's **unseen** templates (the default of LiLa-WAM's
  `eval.sh` and FastWAM's `sim_robotwin.yaml`, following Motus). Upstream leaves this draw unseeded; our harness copy
  seeds Python's `random` with the scene seed before it, so the instruction is also identical across models.
- **Harness changes** (`third_party/RoboTwin/script/eval_policy.py` on this branch, 10 lines; the evaluation tree
  `/groups/yshang/an221229/code/RoboTwin_eval/RoboTwin` uses the same file):
  the seeded instruction draw above; an optional `st_seed` to start a job at a given accepted seed; and an optional
  `instruction_pool`, because the draw picks from up to `eval_num_episodes` generated candidates. With these, a
  50-episode job can run as two processes: episodes 0-24 from seed 100000, and episodes 25-49 from the seed of CL-0's
  26th episode, with pool 50. Checked on CL-0 scan_object/clean episode 2 (seed 100014): same instruction and same
  outcome as in the sequential run. CL-A's jobs ran split this way to fit the allocation. Because the expert check is
  not exactly repeatable, the first half can end a seed or two before or after where the second half starts. A seed
  run twice counts once, at its first episode.
- **Termination.** Success, or the task's step limit: hanging_mug 900, blocks_ranking_size 1200, scan_object 500,
  place_can_basket 700, handover_mic 600.
- **Policy.** Training input protocol exactly (`experiments/robotwin/chronoloop_policy`): three cameras packed into
  384x320 by `preprocess_frames`, the 14-D joint state z-scored with the training statistics, the instruction in
  `DEFAULT_PROMPT` through the training T5 (bit-identical to the training text cache, checked on cached prompts).
  32 actions per query (10 Euler steps, no CFG), de-normalized, the first 24 executed as absolute joint targets, then
  replan (the training stride). Noise fixed per episode. CL-A's memory is reset at episode start and advances once
  per query, as in training.

**What LiLa-WAM reports.** Main results (Table 1, per task in Table 8): one model trained on 50 clean + 500 randomized
demonstrations of each of the 50 tasks, evaluated in both settings, 50 rollouts per task. The ablations (Tables 3-6)
report one "average success rate over the 10 RoboTwin 2.0 tasks" without naming the setting. Their ablation data
(`data-200-10-taskcond`) mixes clean and randomized demonstrations, and their `eval.sh` defaults to
`demo_clean` with unseen instructions, so the ablation numbers are most likely Clean. The paper does not say so.
"Clean -> clean" and "clean -> randomized" are RoboTwin 2.0's single-task protocol (train on 50 clean demonstrations,
test Easy and Hard), which LiLa-WAM does not use.

### 8.2 Results

All 1,000 episodes finished; no process failed. Per-seed outcomes for both models:
`plans/performance/ChronoLoop_Robotwin5hard_Epoch5_results.json` (written by `chronoloop_robotwin_eval_report.py`).
Episode logs and one video per episode are in `/groups/yshang/an221229/checkpoints/ChronoLoop/robot/eval/robotwin5hard/`
(`CL-0_no-memory_a4_epoch05`, `CL-A_mem16-learned-loopwrite_a4_epoch05`).

| Setting | Task | Matched scenes | CL-0 (matched) | CL-A (matched) | CL-0 (all 50) | CL-A (all 50) |
|---|---|---:|---:|---:|---:|---:|
| Clean | hanging_mug | 47 | 9/47 (19%) | 13/47 (28%) | 12/50 (24%) | 15/50 (30%) |
| Clean | blocks_ranking_size | 46 | 23/46 (50%) | 23/46 (50%) | 24/50 (48%) | 23/50 (46%) |
| Clean | scan_object | 46 | 28/46 (61%) | 27/46 (59%) | 30/50 (60%) | 29/50 (58%) |
| Clean | place_can_basket | 37 | 19/37 (51%) | 20/37 (54%) | 22/50 (44%) | 27/48 (56%) |
| Clean | handover_mic | 44 | 44/44 (100%) | 43/44 (98%) | 50/50 (100%) | 49/50 (98%) |
| Clean | **mean of 5 tasks** | 220 | **56.3%** | **57.6%** | **55.2%** | **57.6%** |
| Randomized | hanging_mug | 41 | 10/41 (24%) | 11/41 (27%) | 15/50 (30%) | 14/49 (29%) |
| Randomized | blocks_ranking_size | 49 | 18/49 (37%) | 18/49 (37%) | 18/50 (36%) | 18/50 (36%) |
| Randomized | scan_object | 45 | 20/45 (44%) | 24/45 (53%) | 22/50 (44%) | 25/50 (50%) |
| Randomized | place_can_basket | 37 | 16/37 (43%) | 21/37 (57%) | 21/50 (42%) | 25/49 (51%) |
| Randomized | handover_mic | 48 | 47/48 (98%) | 48/48 (100%) | 49/50 (98%) | 50/50 (100%) |
| Randomized | **mean of 5 tasks** | 220 | **49.3%** | **54.7%** | **50.0%** | **53.1%** |

"All 50" counts every distinct seed a model ran. CL-A ran 50 episodes in every job, but in three split jobs the
first half's expert check rejected one or two seeds that CL-0's had accepted, so the first half ran past the second
half's start seed. Seeds run by both halves count once, so those cells have 48 or 49 scenes. "Mean of 5 tasks" is the mean of the per-task rates, as LiLa-WAM reports it.

Matched scenes by outcome:

| Setting | Task | Both succeed | Only CL-0 | Only CL-A | Neither |
|---|---|---:|---:|---:|---:|
| Clean | hanging_mug | 3 | 6 | 10 | 28 |
| Clean | blocks_ranking_size | 12 | 11 | 11 | 12 |
| Clean | scan_object | 22 | 6 | 5 | 13 |
| Clean | place_can_basket | 12 | 7 | 8 | 10 |
| Clean | handover_mic | 43 | 1 | 0 | 0 |
| Randomized | hanging_mug | 5 | 5 | 6 | 25 |
| Randomized | blocks_ranking_size | 11 | 7 | 7 | 24 |
| Randomized | scan_object | 18 | 2 | 6 | 19 |
| Randomized | place_can_basket | 13 | 3 | 8 | 13 |
| Randomized | handover_mic | 47 | 0 | 1 | 0 |
| | **total** | 186 | 48 | 62 | 144 |

Most outcomes on the middle-difficulty tasks change between the two models on the same scene (blocks_ranking_size
Clean: 22 of 46 scenes flip, evenly in both directions). At epoch 5, which model succeeds on a given scene depends
largely on sampling (flow noise and the closed loop), not only on the scene. Exact McNemar tests on the discordant
scenes: Clean 31 vs 34 (p = 0.80), Randomized 17 vs 28 (p = 0.14), both settings 48 vs 62 (p = 0.22). The Randomized
gap is the more consistent one (CL-A ahead or tied on every task), but 50 episodes per cell cannot separate it from
noise.

Wall time: CL-0's evaluation took 2.4 h (10 processes on GPUs 0-1, sharing 8 CPUs with CL-A's training); CL-A's took
1.6 h of episodes (20 split processes on 4 GPUs; jobs took 0.9-1.55 h). Both are CPU-bound.

### 8.3 Environment and commands

RoboTwin runs inside the training environment (`/groups/yshang/an221229/envs/FastWAM`, Python 3.10, torch 2.7.1),
with its packages installed beside it rather than in a separate conda environment:
- `/groups/yshang/an221229/code/RoboTwin_eval/RoboTwin`: upstream RoboTwin (bf44be5) with its assets, plus the
  vendored `script/eval_policy.py` and `envs/_base_task.py` from `third_party/RoboTwin`, and
  `policy/chronoloop_policy` linked to `experiments/robotwin/chronoloop_policy`.
- `/groups/yshang/an221229/envs/robotwin_overlay` (`uv pip install --target`): sapien 3.0.0b1 and mplib 0.2.1 (both
  patched as in RoboTwin's `_install.sh`), open3d and its dependencies, gymnasium, warp-lang 1.4.2, yourdfpy,
  numpy-quaternion, scikit-image, setuptools_scm, a `curobo` link and an `ffmpeg` binary (imageio-ffmpeg 7.0.2).
  Packages that would shadow the environment's own copies (setuptools, packaging, typing_extensions) were removed.
- `/groups/yshang/an221229/envs/robotwin_overlay_np1`: numpy 1.26.4, placed first on `PYTHONPATH` for evaluation
  only. Under numpy 2.2, mplib 0.2.1's `fcl.Box` segfaults while loading the robot.
- CuRobo v0.7.7, built in place in `/groups/yshang/an221229/code/RoboTwin_eval/curobo` with GCC 11.2
  (`CC/CXX/CUDAHOSTCXX=/apps/gcc/gcc-11.2.0/bin/...`, `CUDA_HOME=/apps/cuda/cuda-12.6.0`, `TORCH_CUDA_ARCH_LIST=9.0`,
  `python setup.py build_ext --inplace`). The system GCC 8 is too old for torch extensions. At run time,
  `gcc-11.2.0/lib64` goes on `LD_LIBRARY_PATH`. Newer warp-lang wheels (>= 1.5) need glibc 2.29; the node has 2.28.

```bash
cd /lustre/fs1/home/an221229/code/LoopWAM_chrono && source scripts/chrono_env.sh
EVAL=/groups/yshang/an221229/checkpoints/ChronoLoop/robot/eval/robotwin5hard
RUNS=/groups/yshang/an221229/checkpoints/ChronoLoop/robot/runs/robotwin5hard
# CL-0: 10 jobs (5 tasks x 2 settings) x 50 episodes, 5 processes per GPU
python scripts/evaluate_chronoloop_robotwin.py --checkpoint $RUNS/CL-0_no-memory_a4/checkpoints/epoch_05.pt \
  --output-dir $EVAL/CL-0_no-memory_a4_epoch05 --gpus 0,1 --per-gpu 5 --episodes 50
# CL-A: the same episodes, each job split in two at CL-0's 26th accepted seed
python scripts/evaluate_chronoloop_robotwin.py --checkpoint $RUNS/CL-A_mem16-learned-loopwrite_a4/checkpoints/epoch_05.pt \
  --output-dir $EVAL/CL-A_mem16-learned-loopwrite_a4_epoch05 --split-from $EVAL/CL-0_no-memory_a4_epoch05 --parts 2 \
  --gpus 0,1,2,3 --per-gpu 5 --t5-gpu 0
python scripts/chronoloop_robotwin_eval_report.py CL-0=$EVAL/CL-0_no-memory_a4_epoch05 \
  CL-A=$EVAL/CL-A_mem16-learned-loopwrite_a4_epoch05 --json $EVAL/robotwin5hard_epoch05_results.json
```

Each job directory holds `eval.log` (per-episode success and seed, and the instruction as a `chronoloop_episode`
line), one video per episode (`episodeK_randomized-..._success-....mp4`, 640x480 at 10 fps: head camera top left, left
and right wrist cameras below) and
`_result_*.txt`. Each evaluation directory has `manifest.json` (protocol and code revision) and `summary.json`.

## 9. Caveats

- **Mid-schedule checkpoints.** Epoch 5 is the midpoint of a 10-epoch cosine schedule (LR still 53% of peak), not an
  annealed 5-epoch model. Both runs continue from `latest.pt` (section 7).
- **Sampling error.** 50 episodes per task and setting give a binomial standard error of up to 7 points per cell;
  the mean over 5 tasks has about 3 points. One training seed per run and one evaluation seed sequence (seed 0).
  CL-A vs CL-0 differences of a few points are within noise; the matched-scene comparison removes scene variation,
  not policy sampling noise.
- **Not comparable to LiLa-WAM's absolute numbers.** These models are trained on 5 tasks only (550 demonstrations
  each); LiLa-WAM and the baselines in its Table 8 are trained on all 50 tasks. Only unseen instructions were tested.
- **Harness.** Three small changes to the vendored harness (section 8.1), and the expert check is not repeatable
  across runs, so each model's 50 scenes per task are not exactly the same set; the matched columns use the
  common scenes.
- **CPU load does not change outcomes.** The simulation advances by steps, and the policy is queried synchronously,
  so the heavy CPU sharing changed wall time only. (It may have contributed to the expert check's variability.)
