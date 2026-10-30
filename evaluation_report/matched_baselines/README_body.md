## 6. Protocol (identical for every model)

- Suites `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`; tasks 0-9; seeds 42, 43, 44; official
  initial states 0-49.
- Per task/seed group: `set_global_seed(seed)`, one environment `get_libero_env(task, 256, seed)` (env.seed) reused
  for its 50 episodes in order; each episode `reset()` + `set_init_state(state[i])`.
- Light-WAM's unchanged `run_single_episode` (sha256 `c7f5c738…`) drives every model: 30 settling actions
  `[0,0,0,0,0,0,-1]` (done ignored), then at most 400 (Spatial/Object/Goal) or 700 (Long) policy steps, 32-action
  chunks with 10 executed per query, success = the environment `done` flag.
- Every action query receives the group's base seed (no per-episode or per-query reseeding).
- Policies run their own native inference: FasterWAM `one_pass_future_cache`, 10 steps, shift 5, BF16; Light-WAM its
  training-run config, 10 steps; LoopWAM the campaign's written rule with the native pieces (strict
  `create_loopwam`, FP32 policy, BF16 VAE, `infer_action` under BF16 autocast, 10 steps, CFG 1, action shift 1,
  gripper `sign(-(2g-1))`). The prompt is encoded once per task with each family's own encoder (cached actions
  verified bit-identical to online prompt encoding).
- Rendering `MUJOCO_GL=egl`. Groups are claimed from a shared queue and written atomically, so the number of workers
  is not part of the protocol, and an interrupted group always reruns from episode 0.

The full specification, the model-construction gates (init independence, strict loading, repeat identity) and every
decision taken during the runs are in [`harness/README_MATCHED_EVAL.md`](harness/README_MATCHED_EVAL.md).

## 7. Setup and provenance

- **Hardware:** one host with 2x NVIDIA RTX PRO 6000 Blackwell (97.9 GB), shared with other users' jobs; runs used
  1-3 workers per model on GPU 0 and/or 1. Eager native inference (no CUDA graph, no compile).
- **Software:** Python 3.10.21, torch 2.7.1+cu128, mujoco 3.3.2, robosuite 1.4.0, LIBERO `8f1084e`.
- **Model code:** Light-WAM `db7d05e`, FasterWAM `8366781`; LoopWAM from this repository: `chrono` @ `277c635` (v0,
  Dense), `partition_grid_20261026` @ `a584631` (partition exports), `v0-init-variants` @ `4ebf38a` (`core16_21`, its
  training revision). The eight training code files recorded in each partition/init export's manifest are
  byte-identical to the checkout it runs on.
- **Checkpoints:** downloaded from Hugging Face at pinned revisions and checked against their LFS sha256 (and every
  other file against its git blob id); logs in [`evidence/download_logs`](evidence/download_logs). Partition exports:
  `bd59888` (p0c6s6, p1c6s5), `906fcb1` (p5c6s1, p6c6s0), `ac02c3f` (p1c9s2_k3, p4c3s5_k7); `core16_21`: `9257a7e`.
- **Per-run records:** [`runs/<model>/`](runs) holds each run's manifest (protocol, source hashes, code revisions),
  summary and first build report per suite.

## 8. Agreement with the H100 campaign

The harness was validated on the released v0 4/4 export (`1c659a7c…`, campaign "LoopWAM full 4/4") before any new
model was scored:

- Gate on 40 campaign episodes: 40/40 outcomes agree, 39/40 trajectories identical
  (`evidence/gates/gate_loopwam_v0_full44_libero_spatial.json`).
- A 200-episode diagnostic run (task 2, seed 43, all four suites): 200/200 outcomes agree with the campaign's
  per-episode records (193/200 successes in both).
- The partition-code and init-variant-code controls reproduced the v0 gate (all successes identical); the only
  differences lie in episodes where the same code varies between reruns on this host (next section).

So the local results and the H100 results above are directly comparable at the success-rate level.

## 9. Caveats and run history

- **Episode-level nondeterminism.** Rerunning campaign group `libero_10:9:42` with identical code and weights on this
  host changed the action digest (sometimes the step count) of episodes among 3-9 of that group from run to run, while
  success never changed (`evidence/gates/chrono_rerun_variation.json`). Success rates are stable; per-episode trajectories are not
  guaranteed bit-identical across reruns.
- **Seed effect.** Seed 44 is the lowest of the three seeds for 11 of the 14 models here (Section 2); all six
  four-suite models with per-episode records in `../final/episodes.csv` decline from seed 42 to 43 to 44. Every query
  of a task/seed group uses the same base seed, so a seed effect is shared by all 50 episodes of a group.
- **Timing is not comparable across models.** The GPUs were shared with other users' jobs and up to 9 of our workers
  ran together, so the mean seconds per query in `data/` (0.12-0.63 s) reflect contention: the five v0 4/4-schedule
  exports (partition 0/6/6, 1/6/5, 5/6/1, 6/6/0 and `core16_21`) share one architecture yet range from 0.20 to
  0.38 s.
  Use `../branch_reports/elastic/plans/performance/LoopWAM_All_Models_Latency_20261030.md` for latency.
- **Interruptions never changed a completed result.** A worker hang (p5c6s1, 2026-10-29), two host reboots
  (2026-10-29 19:25 during p6c6s0, 2026-10-30 18:01 after all runs had finished), a GPU that needed a reset after the
  first reboot, and memory-pressure pauses / one host OOM kill of our own worker (p4c3s5_k7) only interrupted
  unfinished groups, which reran from episode 0 with the same seed; completed groups are written atomically and
  re-verified (Section 5). Details and timestamps: [`evidence/run_parallel_queue.log`](evidence/run_parallel_queue.log)
  and `harness/README_MATCHED_EVAL.md`.
- **Removed from the queue by request (2026-10-28, before they started):** `core_mean` and `no_video_pretraining`;
  both have H100 matched results on branch `v0-init-variants` (Section 1).

## 10. Files

| Path | Content |
|---|---|
| `data/summary_table.md` | Table written by the report script at the end of the queue (includes the campaign rows it compares with) |
| `data/suite_results.csv`, `data/task_results.csv` | Per suite (Wilson 95% interval, per seed) and per task |
| `data/paired_results.csv` | McNemar paired tests per suite against the campaign's LoopWAM full 4/4 |
| `data/episodes.csv`, `data/results.json` | Every scored episode (steps, queries, action digest, checkpoint sha256) and the full results |
| `verification/` | `verify_run.py` output per model (`.txt`) and as JSON |
| `runs/<model>/` | Manifest, summary and build reports per model |
| `evidence/` | Model-construction gates, checkpoint download logs, the queue log |
| `harness/` | Evaluation code (`matched_eval_source.tar.gz`) and its README |
| `build_readme.py`, `README_body.md` | Build this README from the files above and `../branch_reports` |
