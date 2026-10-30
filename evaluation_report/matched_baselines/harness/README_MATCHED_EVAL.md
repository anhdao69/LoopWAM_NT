# Matched LIBERO evaluation of released WAM baselines and LoopWAM all-suite exports

Released baselines and the LoopWAM all-suite exports (Hugging Face `anhdao69/LoopWAM_NT`, `libero-all-suites/*`)
evaluated with the exact rollout protocol of the LoopWAM campaign
(`lightwam_20261009_matched`, see `/data/anh/code/LoopWAM_NT/evaluation_report/final`).

## Layout

| Path | Content |
|---|---|
| `Light-WAM/` | Light-WAM @ `db7d05e` (native evaluator `experiments/libero/eval_libero_single.py`, sha256 `c7f5c738…`) |
| `FasterWAM/` | FasterWAM @ `8366781` (`third_party/LIBERO` → `../LIBERO`) |
| `LiLa-WAM/` | LiLa-WAM @ `339860f` (RoboTwin-only release; no LIBERO checkpoint/config/evaluator) |
| `LoopWAM-chrono/` | LoopWAM_NT `origin/chrono` @ `277c635` (git worktree); model/dataset files byte-identical to the dense training revision `8e84591` |
| `LoopWAM-init-variants/` | LoopWAM_NT `origin/v0-init-variants` @ `4ebf38a` (git worktree; `chrono` + init-variant builders, the training revision of `init_variants/*`); checkpoint loading/inference identical to `LoopWAM-chrono/` |
| `LoopWAM-partition/` | LoopWAM_NT `origin/partition_grid_20261026` @ `a584631` (git worktree); the eight training code files recorded in the partition exports' manifests are byte-identical (trained at `96dc3f9`) |
| `LIBERO/` | LIBERO @ `8f1084e` (benchmark tasks, BDDL files, official initial states) |
| `.venvs/lightwam-libero` | Light-WAM environment from its lock (Python 3.10.21, torch 2.7.1+cu128, mujoco 3.3.2, robosuite 1.4.0) |
| `FasterWAM/.venvs/libero` | FasterWAM environment from `environments/libero/uv.lock` |
| `checkpoints/` | Released weights (SHA-256 verified, see `logs/download_*.log`) |
| `data/libero_mujoco3.3.2/` | FastWAM LIBERO dataset `yuanty/LIBERO-fastwam` (LeRobot v2.1, SHA-256 verified); linked into both repos as `data/libero_mujoco3.3.2` |
| `text_cache/<family>/` | Prompt embeddings from each family's own `encode_prompt` + provenance and parity records (`text_cache/loopwam/`: LoopWAM native cache format, shared by all LoopWAM variants) |
| `matched_eval/` | Matched runner, gates, offline check and report code |
| `matched_eval_next/`, `matched_eval_partition/` | Staged supersets of `common.py`/`families.py` (Ka sweep; partition exports) used by queued runs until no running run hashes the installed files |
| `runs/<family>_matched/` | Episode/group JSON, videos, manifest, summary |

## Checkpoints

| Model | Suite | File | SHA-256 |
|---|---|---|---|
| Light-WAM | Spatial | `lightwam_libero_spatial/.../step_055000.pt` | `421e954b…` |
| Light-WAM | Object | `lightwam_libero_object/.../step_012500.pt` | `57a83d30…` |
| Light-WAM | Goal | `lightwam_libero_goal/.../step_060000.pt` | `0c40ca81…` |
| Light-WAM | Long | `lightwam_libero_10/.../step_080000.pt` | `93c816ed…` (same as campaign Light-WAM Long) |
| FasterWAM | all four | `fasterwam_release/libero/step_021700.pt` | `01be4efc…` |
| LoopWAM v0 video4/action4 (`loopwam_v0_full44`) | all four | `loopwam_hf/v0-video4-action4/policy.pt` | `1c659a7c…` (= campaign "LoopWAM full 4/4"; validation only) |
| LoopWAM Dense-S12 (`loopwam_dense_s12`) | all four | `loopwam_hf/dense-s12/policy.pt` | `a94f1839…` (584,536,135 policy parameters) |
| LoopWAM Dense-S30 (`loopwam_dense_s30`) | all four | `loopwam_hf/dense-s30/policy.pt` | `9726eec3…` (1,416,114,247 policy parameters) |
| LoopWAM v0 partition (0, 6, 6) (`loopwam_partition_p0c6s6`) | all four | `loopwam_hf/partition-grid-centered-20261026/p0c6s6_k4/policy.pt` | `9d6a754a…` (584,536,135 policy parameters) |
| LoopWAM v0 partition (1, 6, 5) (`loopwam_partition_p1c6s5`) | all four | `loopwam_hf/partition-grid-centered-20261026/p1c6s5_k4/policy.pt` | `26992d6c…` (584,536,135 policy parameters) |
| LoopWAM v0 init core16_21 (`loopwam_init_core16_21`) | all four | `loopwam_hf/init_variants/core16_21/policy.pt` | `7810a434…` (584,536,135 policy parameters) |

LoopWAM exports: step 21,700, 10 epochs on all 1,712 LIBERO training demos of the four suites, global batch 128,
seed 42; normalization `dataset_stats.json` `d5ad351e…` (identical for all three); Wan2.1 VAE `38071ab5…`.

Test-time action depth of the v0 export (`loopwam_v0_ka3`, `loopwam_v0_ka5`, `loopwam_v0_ka6`; same weights
`1c659a7c…`, video loops 4, NFE 10): the action stream runs Ka core iterations over the 4-loop video KV cache, with
the repository's late alignment (iteration r reads video core loop `max(1, 4 − Ka + r)`: Ka 3 → 2,3,4; Ka 5 →
1,1,2,3,4; Ka 6 → 1,1,1,2,3,4). The checkpoint was trained at Ka = 4, so these are test-time depth changes; Ka 5
and 6 exceed the trained loop capacity, which the native code refuses (`create_loopwam` and `LoopMoT.action_loops`),
so `LoopWAM.set_action_loops` sets them explicitly. `matched_eval/run_v0_ka_sweep.sh` queues Ka 3 → 5 → 6 behind
the dense runs.

Centered-partition v0 exports (HF `libero-all-suites/partition-grid-centered-20261026`, revision `bd59888`,
`checkpoints/loopwam_hf/download_partition.sh`): same data, normalization, prompt caches, VAE and initialization
artifact as the v0 export; the twelve blocks per expert are split (prelude, core, coda) = (0, 6, 6) and (1, 6, 5)
instead of (3, 6, 3), 4 loops in both experts, aligned KV, donor blocks centered in Wan (`stage_centered`); both
execute 30 block applications per pass like v0. Only the partition-aware code loads them
(`format_version loopwam-s-partition-v1`), so `loopwam_partition_*` families run from `LoopWAM-partition/`
(`launch.sh` routes them; `families.LoopWAM` refuses code imported from another checkout or revision). Its
inference path differs from `LoopWAM-chrono/` only in the partition depths; the regression control
`loopwam_partition_code_v0_full44` (v0 export through the partition code) must reproduce the chrono v0 gate's
42 episodes: every success identical, and steps, queries and action digests identical except on episodes where
the chrono code itself varies between runs. Reruns of campaign group `libero_10:9:42` with identical chrono code
and weights on GPU0 (`logs/gates/chrono_rerun_libero_10_9_42/`, `logs/gates/chrono_rerun_variation.json`) gave
different action digests on episodes 3–7 and 9 (once also 352 vs 353 policy steps; success never changed), while
episodes 0–2 and 8 were bit-identical in every run, so episode-level bit-identity is not reproducible on this host
for every long-horizon episode. The partition-code control matched the chrono gate on 2/2 matched and 35/40
campaign episodes, with all 40 successes agreeing; its 5 differing episodes (libero_10:9:42 episodes 3–7) all lie
in that same-code variation. The init-variant control (below) also matched 2/2 and 35/40 with 40/40 successes; it
differed on episodes 3, 4, 6, 7 and 8. Its checkpoint-load and inference path is byte-identical to chrono and its
worker fingerprint equals the chrono gate's, so it is itself a same-code rerun, and episode 8 was added to the
variation file with that provenance.

Initialization variant `core16_21` (HF `libero-all-suites/init_variants/core16_21`, revision `9257a7e`,
`checkpoints/loopwam_hf/download_init_variants.sh`): the v0 model, data and training with core blocks initialized
from Wan blocks 15–20 instead of 9–14 (initialization artifact `3eb17dcb…`); default partition (3, 6, 3), format
`loopwam-s-v1`. It runs from `LoopWAM-init-variants/` (its training revision; all eight recorded training code
files byte-identical), with the regression control `loopwam_init_code_v0_full44`.

`matched_eval/run_partition_queue.sh` (service `wam-queue-partition`) ran these gates once Dense-S12 had
finished (all passed 2026-10-27 19:07).

On 2026-10-27 19:26 (GPUs freed up; user request to use them) the sequential services were replaced by
`matched_eval/run_parallel_queue.sh` (service `wam-queue-parallel`, log `logs/run_parallel_queue.log`): Ka3 (its
runner interrupted with SIGINT; 83 complete groups kept, resumed with 3 workers), Ka5, Ka6, p0c6s6, p1c6s5 and
core16_21 run as separate units `wam-matched-<family>`, at most 3 at a time with 3 workers each on GPUs 0 and 1, a
new run starting only with ≥ 16 GB RAM available. Worker count is not part of the protocol (each group is seeded
on its own and taken from a shared queue), and each run keeps its staged code dir, so the source hashes in its
manifest are unchanged. At the end the queue installs `matched_eval_partition` (the superset) into `matched_eval` and
writes the report to `reports/matched_baselines`.

Added 2026-10-27 (user request): centered-partition exports p5c6s1_k4 (5, 6, 1) and p6c6s0_k4 (6, 6, 0), same HF
folder at revision `906fcb1` (`checkpoints/loopwam_hf/download_partition_p5_p6.sh`; policy SHA-256 and all file blob
ids verified). Their training manifests differ from p0c6s6 only in `block_partition`, the stage-centered donor
indices (p5c6s1: 0–4 | 12–17 | 29; p6c6s0: 0–5 | 12–17 | none) and run bookkeeping (output path, setup time, Slurm
id), so they run on the same `LoopWAM-partition` code (families `loopwam_partition_p5c6s1`/`_p6c6s0`), gated by
`matched_eval/run_gates_p5_p6.sh`, then appended to the parallel queue; the final report covers twelve runs.

Added 2026-10-28 (user request): initialization variant `core_mean` (HF `libero-all-suites/init_variants/core_mean`,
revision `05d793d`, `checkpoints/loopwam_hf/download_init_core_mean.sh`; policy SHA-256 `7613e262…` and all file blob
ids verified): the v0 model, data and training with core block i initialized as the element-wise mean of Wan blocks
3+i, 9+i, 15+i, 21+i (initialization artifact `506bcecf…`); default partition (3, 6, 3), same training revision
`4ebf38a` (all eight recorded training code files byte-identical to `LoopWAM-init-variants/`), so it runs as family
`loopwam_init_core_mean` on the init-variant code already covered by `loopwam_init_code_v0_full44`. Its entries live
in the staged dir `matched_eval_core_mean` (= `matched_eval_partition` + three `common.py` entries), because the
running partition runs hash `matched_eval_partition`. Gate: `matched_eval/run_gates_core_mean.sh` (unit
`wam-gates-core-mean`); it is the ninth queued run, the queue now installs `matched_eval_core_mean` at the end and the
final report covers thirteen runs.

2026-10-28 03:10: other users' jobs grew and the host reached 0.8 GB available with swap full (thrashing). Ka6 was
paused again (runner SIGINT, 40 complete groups kept; 58 in-progress episodes rerun on resume) and requeued, and the
scheduler now starts a run that would be the second one running only with >= 24 GB available (`SECOND_RAM_GB`; the
first still needs 16 GB), so two runs overlap only when the other jobs leave room. p1c6s5, started by the old
scheduler instance during that pause, was stopped before its first episode and requeued after Ka6.

2026-10-28 03:41 (user request: "just use 1 run for me and make all the run in sequence"): `SLOTS=1`. Ka6 was paused
(runner SIGINT, 47 complete groups kept) while p0c6s6 continued; the queue then runs Ka6, p1c6s5, core16_21, p5c6s1,
p6c6s0 and core_mean one after another, each with 3 workers on GPUs 0 and 1.

Added 2026-10-28 (user request), HF revision `ac02c3f` (`checkpoints/loopwam_hf/download_extra_20261028.sh`; policy
SHA-256 and every file blob id verified), staged in `matched_eval_extra` (= `matched_eval_core_mean` + entries; gates
`matched_eval/run_gates_extra.sh`, unit `wam-gates-extra`, all passed 05:50):
* `loopwam_partition_p1c9s2_k3` (1, 9, 2; 3 video/action loops) and `loopwam_partition_p4c3s5_k7` (4, 3, 5; 7 loops):
  their manifests differ from p0c6s6's only in partition, loop counts, donor indices and run bookkeeping (same
  training revision, code-file hashes, data and text caches), so they run on `LoopWAM-partition`. Loop counts come
  from the checkpoint; `LOOPWAM_TRAINED_LOOPS` makes the build refuse any other value. Gate builds: action core
  schedule [1, 2, 3] and [1 … 7].
* `loopwam_init_no_video_pretraining`: v0 4/4 with every Wan-derived tensor from Wan2.1's from-scratch initialization
  (artifact `69f284e9…`), trained at LoopWAM_NT `56e3328` plus a training-only patch (init builder, cached-latent
  loader). Its inference files equal `LoopWAM-chrono`'s except `loop_mot.py`, whose difference is a training-only
  activation-checkpointing option (`self.training and torch.is_grad_enabled()`), so it runs on the init-variant code
  (`LoopWAM-init-variants`, same inference path). Its 40 training T5 caches were regenerated on its cluster (same
  prompts and valid-token counts, different bytes; documented in its `fairness.json`); the shared evaluation cache
  is itself a regeneration not byte-equal to any export's training cache (`bytes_equal_training: 0`), so for this
  family only (`LOOPWAM_REGENERATED_TEXT_CACHE`) the build checks prompt-file names and valid tokens instead of the
  v0 training bytes, and records `training_text_cache` in its build report. Data manifest otherwise equal to v0's
  apart from cluster paths.
The queue runs them last (one at a time) and the final report covers sixteen runs.

2026-10-28 19:12 (user request): `core_mean` and `no_video_pretraining` were removed from the queue before either
started (checkpoints, passed gates and staged code entries are kept). The queue now ends with p5c6s1, p6c6s0,
p1c9s2_k3 and p4c3s5_k7 after core16_21, and the final report covers fourteen runs.

2026-10-28 21:24–23:18: another user's `hold_gpu.py` held ~90 GB of GPU1 and 44 GB of GPU0, so p5c6s1's three
workers could not get GPU room (`run_until_done.sh` waits for 8 GB per worker). By user decision p5c6s1 was restarted,
before its first episode, with 1 worker on GPU0 only (worker count is not part of the protocol).
On 2026-10-29 10:09 its worker hung on `libero_10_seed44_task5` episode 10 (main thread at 100% CPU with no syscalls,
page faults or I/O for 40 min, no GPU activity; ptrace/perf are not permitted on this host, so the frame is unknown).
The runner was interrupted at 10:52 and `run_until_done.sh` resumed: the group reran from its start with the same
seed, reproduced episodes 0–9 (same successes and steps) and finished episode 10 normally (700 steps, failure), so the
hang was not reproducible.

**Host reboot and GPU1 failure (2026-10-29).** p5c6s1 finished at 11:43 (verified, 5590/6000). The host rebooted at
19:25 and killed p6c6s0 (95 complete groups, 4764 episodes) and the queue; transient units do not survive a reboot.
Crash damage was limited to the unfinished group `libero_10_seed42_task5` (six zero-byte episode files, rewritten when
the group reran from episode 0) and a truncated last line in `events.jsonl` (newline appended so later events start on
their own line); all 95 group files and every other episode file parse. Since the reboot GPU1 is in an error state
(nvidia-smi: `ERR!`, "GPU requires reset"; CUDA: "No CUDA GPUs are available"; resetting it needs root). On the user's
request to maximize throughput ("maximize the running for me", "using gpu0 also") the queue runs two models at once
with two workers each on GPU0 (~4.7 GB per worker next to other users' ~69 GB), units get `OOMScoreAdjust=500` (a host
OOM kills our workers before other users' processes), and `wam-ram-guard` (`matched_eval/ram_guard.sh`) pauses the
newest run when MemAvailable stays below 5 GB. p6c6s0 resumed at 23:47 and p1c9s2_k3 started at 23:52; their build
reports match the gates (checkpoint, partition, loops, NFE/CFG/settling/replan, strict loads).
At 00:18 on 2026-10-30, after another user's 34 GB job on GPU0 ended, p4c3s5_k7 started as a third parallel run
(2 workers; a further 2-worker run now needs 20 GB RAM available, the same margin as the 24 GB rule for 3-worker runs);
its build reports also match its gate (7 loops, schedule 1..7).

**Completion (2026-10-30).** p6c6s0 finished at 05:52 (5416/6000) and p1c9s2_k3 at 15:29 (5507/6000). The RAM guard
paused p4c3s5_k7 twice (05:24 and 15:59, MemAvailable 4 GB and 2 GB); `run_until_done.sh` resumed it each time. Its
third attempt then hit a CUDA OOM (another user's job filled GPU0 after our model was built) and in its fourth the host
OOM killer took one of our workers (our units carry `OOMScoreAdjust=500`). `run_until_done.sh` therefore gained
`MIN_RAM_GB` (wait for that much available RAM before each attempt; default 10) and p4c3s5_k7 was relaunched at 16:26
with `MIN_RAM_GB=16` (scheduler stopped first, then the unit). It finished at 17:09 (5445/6000) and the queue wrote
`reports/matched_baselines` (fourteen runs) at 17:11. Interrupted groups always reran from episode 0 with the same
seed. The host rebooted again at 18:01, after every run had finished; GPU1 is healthy again after that reboot.
`matched_eval/verify_run.py` (driver `matched_eval/verify_all.sh`) then re-read all fourteen runs: full 6000-episode
grid, 120 group files against the episode files, manifest, every build report and the summary. All fourteen pass;
outputs in `reports/verification/`.

Pretrained assets: Wan2.1-T2V-1.3B DiT/VAE/T5 (T5 `7cace0da…`, identical to the campaign encoder),
DiffSynth converted Wan2.2 VAE (`0e913a2c…`) and T5 (`d92de679…`) safetensors for FasterWAM.

## Protocol (identical for every model)

* Suites `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`; tasks 0–9; seeds 42, 43, 44;
  official initial states 0–49 → 6,000 episodes per model.
* Per task/seed group: `set_global_seed(seed)`, `get_libero_env(task, 256, seed)` (env.seed), environment
  reused for the 50 episodes; each episode `reset()` + `set_init_state(state[i])`.
* Light-WAM's unchanged `run_single_episode`: 30 settling actions `[0,0,0,0,0,0,-1]`, caps 400/400/400/700 policy
  steps, 32-action chunks, replanning after 10 executed actions, success = environment `done`.
* Every action query receives the same base seed (no per-episode or per-query reseeding).
* Policy = family's unchanged native `_predict_action_chunk` and native config (Light-WAM: training-run config,
  10 steps, direct action decoder, Torch resize; FasterWAM: `one_pass_future_cache`, 10 steps, shift 5, PIL resize,
  BF16). Two 256×256 views rotated 180°, 224×448 concatenation, 8-D proprio, `binarize_gripper=true`.
* LoopWAM (no native matched callback is released): the campaign's written LoopWAM rule, implemented with the native
  LoopWAM pieces (`families.LoopWAM`): strict `create_loopwam(checkpoint_path, vae_path)` (FP32 policy, BF16 VAE, depth
  and loops from the checkpoint, action shift 1), native `ObservationAdapter` (Torch antialiased resize, [-1, 1], 8-D
  proprio with the released training statistics), `infer_action` under BF16 autocast with 10 steps, CFG 1 and the
  group's base seed, gripper `sign(-(2g-1))`, no clipping. Prompt context: Wan T5 (`d92de679…` safetensors, tensor-for-
  tensor identical to `7cace0da…`), batch one, BF16, length 128, whitespace cleaning (`prepare_text_loopwam.py`); all 40
  prompts are training prompts with identical valid-token counts (bytes differ from the unreleased training caches).
* Rendering: `MUJOCO_GL=egl` for all models (FasterWAM's launcher defaults to OSMesa; EGL matches the campaign).

## Memory-placement deviations and the gates that verify them

The host is shared (RAM often < 10 GB free), so model construction differs from the launchers only in placement:

1. Parameters are allocated directly on the GPU (`accelerate.init_on_device(..., include_buffers=False)`) and
   checkpoints are memory-mapped. Only random initial values change; `LoadTracker` proves every parameter and
   buffer is overwritten by a pretrained/checkpoint load and no checkpoint tensor is dropped, and `gates.py`
   builds the worker model with two different init seeds and requires all tensors to be bit-identical.
2. Rollout workers do not instantiate the frozen T5. `prepare_text.py` encodes the 40 prompts with each family's own
   `encode_prompt` (Light-WAM: exactly its native pre-encoding step), and on a real settled observation of every task
   requires the native online-prompt action chunk to equal the cached-prompt chunk bit for bit.
3. FasterWAM is driven by Light-WAM's rollout loop (as in the campaign); `gates.py` requires identical actions,
   steps and outcome versus FasterWAM's own `run_single_episode`.
4. LoopWAM: every parameter and buffer belongs to a strictly loaded module (`mot`, `proprio_encoder`, VAE), the
   checkpoint passes the native final-training contract (`validate_checkpoint`, ten complete epochs), two builds
   are bit-identical and repeated rollouts identical. The adapter is validated on `v0-video4-action4` against the
   campaign's per-episode "LoopWAM full 4/4" records (`gates.py --campaign-model`, `runs/loopwam_v0_full44_diag`).

## Commands

```bash
cd /data/anh/code/wam_baselines
CUDA_VISIBLE_DEVICES=1 matched_eval/launch.sh fasterwam $PWD/matched_eval/prepare_text.py --family fasterwam
CUDA_VISIBLE_DEVICES=1 matched_eval/launch.sh fasterwam $PWD/matched_eval/gates.py --family fasterwam --suite libero_10
matched_eval/launch.sh fasterwam $PWD/matched_eval/run_matched.py --family fasterwam \
    --output $PWD/runs/fasterwam_matched --workers 3 --gpus 1
matched_eval/launch.sh lightwam $PWD/matched_eval/run_matched.py --family lightwam \
    --output $PWD/runs/lightwam_matched --workers 2 --gpus 0
CUDA_VISIBLE_DEVICES=1 matched_eval/launch.sh loopwam $PWD/matched_eval/prepare_text_loopwam.py \
    --training-manifest $PWD/checkpoints/loopwam_hf/dense-s12/training_manifest.json
CUDA_VISIBLE_DEVICES=0 matched_eval/launch.sh loopwam $PWD/matched_eval/gates.py --family loopwam_v0_full44 \
    --suite libero_spatial --campaign-model "LoopWAM full 4/4" \
    --campaign-groups libero_spatial:0:42 libero_object:3:43 libero_goal:6:44 libero_10:9:42
matched_eval/launch.sh loopwam $PWD/matched_eval/run_matched.py --family loopwam_dense_s12 \
    --output $PWD/runs/loopwam_dense_s12_matched --workers 2 --gpus 0
```

Runs are resumable: completed task/seed groups are re-verified and kept; a partial group restarts at episode 0.
To stop a run, send SIGINT to the runner (`kill -INT <run_matched.py pid>`): its cleanup terminates the workers.
SIGTERM skips that cleanup and leaves orphaned workers writing into the run directory.

### Launching on this shared host

Launch long runs as their own systemd user services, never from an interactive/agent terminal:

```bash
systemd-run --user --unit=wam-matched-<family> -p OOMPolicy=continue -p WorkingDirectory=$PWD \
    $PWD/matched_eval/run_until_done.sh <family> <workers> <gpu> [gpu...]
journalctl --user -u wam-matched-<family> -f      # service log; run log: logs/run_<family>_matched.log
```

Processes started from a tmux pane live in that pane's systemd scope. When the kernel OOM killer hits any of
them, systemd stops the whole scope (this killed two runs and the agent session on 2026-10-26/27). A separate
service with `OOMPolicy=continue` loses at most the killed run, and `run_until_done.sh` resumes it after RAM recovers.
