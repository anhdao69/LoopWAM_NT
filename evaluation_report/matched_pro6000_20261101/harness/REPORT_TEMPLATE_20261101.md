# Matched LIBERO evaluation — batch of 2026-11-01 (2x RTX PRO 6000)

Seven conditions requested on 2026-11-01, each scored on the full matched protocol `lightwam_20261009_matched`
(4 suites x 10 tasks x official initial states 0-49 x evaluation seeds 42/43/44 = **6,000 episodes per condition**),
the same protocol as every row of [`../matched_baselines`](../matched_baselines).

{{PENDING}}

## 1. Results of this batch

{{BATCH}}

The 95% intervals are cluster-bootstrap percentile intervals (cluster = suite, task and initial state with its three
seeds; 10,000 resamples), as declared in the analysis plan before any scored episode
([`ANALYSIS_PLAN_20261101.md`](ANALYSIS_PLAN_20261101.md), sha256 `245439595c058403…`).

{{FINDINGS}}

## 2. Contrasts

All contrasts are secondary and unadjusted: the plan declares no primary hypothesis for this batch. The difference is
the mean of per-cluster differences; the interval comes from 10,000 cluster-bootstrap resamples and p from a two-sided
paired sign-flip permutation test (10,000 flips). A nonsignificant difference is not evidence of equivalence.
Names: "Z10" is the base v0 4/4 checkpoint without distillation; "base v0 KaN" its local NFE-10 runs at action depth N;
"campaign v0 a/b" the campaign's NFE-10 records of the v0 exports with video depth a and action depth b.

- **Same host:** this machine with the same regenerated prompt cache (the base-v0 K_a references are the local runs of
  2026-10-27/28 in `../matched_baselines`).
- **Same hardware (campaign):** the campaign records in [`../final`](../final) were scored on another machine of the same
  hardware model (RTX PRO 6000 Blackwell Max-Q, Threadripper 9960X; NVIDIA driver 580.178.04 then, 580.105.08 here). The plan listed
  these contrasts as cross-host; they are labelled separately here because gate G4 reproduced the campaign's v0 4/4
  trajectories 100/100 on this machine.
- **Cross host (H100):** the elastic branch's H100 evaluation, which also read the training prompt-cache bytes rather
  than this host's regenerated cache. Treat these as indicative only.

{{CONTRASTS}}

## 3. Reference rows used by the contrasts

{{REFS}}

## 4. How the batch was run

**Weights** (Hugging Face `anhdao69/LoopWAM_NT`, verified by sha256 before use):

| Condition | File | sha256 |
|---|---|---|
| v0 4/2 | `libero-all-suites/v0-video4-action2/policy.pt` | `33476d5808965252eaa3f5e996b5f5f75cc0542314f825a0fc07fbdec2f0c9d4` |
| E0 [k]x10 | `Elastic/E0/epoch_10.pt` | `5569c47fa6c91e136a611d2b819a053c94f6b2956a7f602bbed4db23c69f60e8` |
| base v0 4/4 (Z10) | `libero-all-suites/v0-video4-action4/policy.pt` (unmodified) | `1c659a7c2f6970c3ea5972a8fba4abd9db35857c83a67a053c0ee35555fb602b` |

**Evaluators.**

- **v0 4/2:** the local matched harness used for the 14 local rows of `../matched_baselines` (variant
  `matched_eval_v042`, family `loopwam_v0_v4a2`, LoopWAM `chrono` model code, eager native `infer_action`, NFE 10, CFG 1).
- **The six other conditions:** the elastic matched runner `scripts/evaluate_rt_matched.py` of `LoopWAM-elastic`, local
  branch `eval/matched-pro6000-20261101` (commits `5f359cb`..`93beb6d` on top of `6ec1401`; a local branch, not pushed: its four commits are in `harness/elastic_runner_patches/`). CUDA-graph
  backend, one model build per worker, groups claimed from a shared queue with heartbeats. "Z10" is the unmodified v0 4/4
  checkpoint run through `TransportInference(zero_shot_kind=T0)` (no step conditioning, no transport); E0 runs its own
  elastic schedule. K_a 3, 5 and 6 are test-time depths that E0 was not trained on (the plan: 3 untrained,
  5 and 6 deeper than trained).
- Protocol details shared by both evaluators: per task/seed group `set_global_seed(seed)` and one environment
  (`env.seed(seed)`) reused for its 50 episodes; 30 settling steps; at most 400 policy steps (700 for Long); 32-action
  chunks, 10 executed per query; every query gets the group's base seed; success is the environment `done` flag;
  `MUJOCO_GL=egl`, 256-pixel cameras. A group record is written only when all 50 episodes finish, so worker count and
  interruptions are not part of the protocol.

**Gates passed before scoring** (outputs under `wam_baselines/logs/elastic_gates`, `logs/gates` and
`runs/elastic_gate_repro_20261101`):

- **G1, graph = eager:** the CUDA-graph backend is bit-exact to eager inference on this host (E0: 80 queries over 5
  schedules; Z10: 112 queries over 7 schedules).
- **G2, zero-shot parent = native:** Z10 through the elastic runner is bit-exact to native LoopWAM v0 inference
  (112 queries, 16 per schedule, all seven schedules).
- **G3, depth definition:** the base-v0 K_a definition used by the elastic runner is the one used by the local
  `loopwam_v0_ka3/5/6` runs.
- **G4, reproduction:** Z10 [4]x10 vs the campaign's v0 4/4, and Z10 K_a 5 vs the local `loopwam_v0_ka5` run, were
  trajectory-identical on 100/100 episodes. E0 records vs the H100 records agreed on 94/100 and 88/100 outcomes, with no
  systematic deficit (different GPU, and the regenerated prompt cache).
- **v0 4/2 harness gate:** passed; the asymmetric-export control (`loopwam_v0_v4a1_control` vs the campaign's 4/1)
  matched 40/40 outcomes and 38/40 trajectories.

**Integrity of the scored records:**

{{INTEGRITY}}

The report script (`matched_eval_elastic/report_20261101.py`) checks every group record (checkpoint sha256, schedule,
50 episodes in order, episode keys consistent with the group, no duplicates), the run manifest against the plan
(backend, clean source tree, EGL, zero-shot kind, test-time flag, prompt cache, E0 depths, every protocol field), and,
once written, the runner's own `summary.json` against the records.

**Interruptions** (no scored record was affected: a group record is written only when all 50 episodes finish, and an
interrupted group reruns from episode 0 with the same seed):

- v0 4/2: interrupted five times (times UTC on 2026-11-01) — a CUDA MPS fault at 06:49 when one of our elastic lanes
  was stopped (fixed by a graceful SIGTERM handler, commit `93beb6d`); a restart by hand at 07:11 to go from 4 to 3
  workers (host RAM); a RAM-guard pause at 09:18; then "device busy" at 09:22 and a CUDA OOM at 09:24 while other
  users' jobs filled the GPUs. Its `events.jsonl` holds the three runtime errors; none touched a written record.
  306 episode runs (301 distinct episodes) from interrupted groups were discarded and rerun. Every one had been logged
  before it was discarded: all 306 had the same outcome as the scored rerun, 303 also the same step count and action
  digest, and 3 (Long, seed 42, task 8) the same step count but different actions — the episode-level nondeterminism
  described in `../matched_baselines`.
- Elastic lanes (lane0 started by hand, l1-l18 by the scheduler): 13 were paused by our RAM guard when host RAM fell
  below its threshold (lane0, l2, l3, l5, l7, l8, l9, l10, l11, l13, l14, l15, l16) and three were stopped by hand (l1
  before the SIGTERM fix had a CUDA context, l4 after a CUDA OOM at build, l12 at 00:29 UTC on 2026-11-02 when host RAM
  fell to 2-3 GB). Claims held by a stopped lane were retaken by the next lane to start.

## 5. Caveats

- **Shared host.** Other users' training and inference jobs shared both GPUs and most of the 62 GB of host RAM
  throughout, so host RAM, not GPU, limited concurrency. A running worker holds about 3 GB of host RAM (about 1.8 GB of
  it the MuJoCo/EGL renderer), about 4 GB while it builds its model. The v0 4/2 harness ran 4 workers, then 3
  (06:20-10:54 UTC on 2026-11-01). From the elastic group records, 1-4 groups were in progress at once: 3 for 13.0 h,
  4 for 3.8 h, 1-2 for 4.8 h and none for 1.5 h (model builds, pauses and restarts). CUDA MPS shared each GPU between our
  workers. The elastic runner completed 22-63 groups per clock hour from 11:00 UTC on 2026-11-01; throughput is a
  property of this shared host, not model latency. The batch ran from 06:16 UTC on 2026-11-01 (analysis plan) to
  06:20 UTC on 2026-11-02 (last group).
- **Prompt cache.** The six elastic conditions read a prompt cache regenerated on this host (names and valid tokens
  checked against the training manifest; bytes differ from the training cache). The same cache was used for the local
  base-v0 K_a references, so same-host contrasts are unaffected; the H100 rows read the training bytes.
- **Two backends.** v0 4/2 and the local references ran eagerly; the six elastic conditions ran with CUDA graphs (gate G1).
- **Code changes during the batch** were host-memory and shutdown fixes only (`malloc_trim` after model build, a
  synchronize-then-exit SIGTERM handler, recording the code head per group), all committed before the first production
  group: every elastic group record carries `93beb6d`. Scheduler and RAM-guard thresholds were tuned during the run;
  they decide only when workers start or stop.
- **Seeds and suites.** Per-seed spread is shown in Section 1; Long carries most of the differences between models.

## 6. Project status at the start of this batch

Read from `evaluation_report/` at `5a249fc` (pulled from GitHub on 2026-11-01 05:54 UTC; still the latest commit when
re-fetched at 00:33 UTC on 2026-11-02). Re-fetched on 2026-11-03: the only commits since are this report's
own sync and the new branch in the last bullet.

- **Matched LIBERO table** (`../matched_baselines`, updated 2026-10-30): 88 models on this protocol. FasterWAM 98.72 (public
  release) and FastWAM 96.63 lead; the five ChronoLoop runs at action depth 4 score 95.90-96.18 (95.70 and 95.33 at
  depth 1), Light-WAM 95.77. Best other LoopWAM rows: elastic F0-PB [4,4] 95.88, F0-PB [4]x5 95.75, E1 [4]x5 95.65,
  Dense-S30 95.63; the base v0 4/4 scores 94.92. The lowest rows are Dense-S12 89.98, base v0 at K_a 3 (89.47) and
  K_a 6 (89.20), and two init variants (85.65, 82.20). Rows from other branches were copied, not re-verified: the remote
  campaign scored FastWAM, the campaign v0 exports, ChronoLoop and part of Residual Transport on its RTX PRO 6000 host,
  and the rest were scored on H100 (the table had labelled them all H100; corrected 2026-11-03). The 14 local rows
  were each re-verified.
  The report listed v0 4/2 as "not evaluated"; this batch fills that gap. LiLa-WAM has no matched result (its only
  public weights are for RoboTwin).
- **RoboTwin v0** (`../robotwin_v0`, completed 2026-10-31): 91.73% over all 50 tasks and 10,000 episodes (91.60% clean,
  91.86% randomized), one seed.
- **Robustness** (`../robustness`, LIBERO-Plus / LIBERO-Pro, seven ChronoLoop checkpoints, seed 42): Plus 45.8-48.4%,
  Pro about 7% (position perturbations almost always fail, task perturbations 13-15%).
- `../dense_full` records that Dense-S12/S30 were removed from an earlier queue on 2026-10-27; both were later scored on
  the matched harness (89.98 and 95.63).
- **New branch since the batch started:** `robotwin10_training_20261101` (commits of 2026-11-02 and 2026-11-03) holds the
  code to train seven variants (v0 4/4, 4/2, 4/1, 2/2, 1/4, Dense-12 and Dense-30) on the LiLa-WAM 10-task RoboTwin
  subset with the full-RoboTwin v0 recipe (5 epochs, seed 42), with a Slurm dispatcher and Hugging Face publishing.
  It has no evaluation results yet.

## 7. Files

- `ANALYSIS_PLAN_20261101.md` — written before any scored episode.
- `results.json`, `results.md` — output of `report_20261101.py` (all rates, intervals and contrasts above).
- `harness/` — the report script, this report's builder and template, the episode exporter, the scheduler
  (`autoscale.sh`), RAM guard (`ram_guard.sh`) and lane wrapper (`run_until_done.sh`); the elastic runner patches
  (`elastic_runner_patches/`, `git format-patch 6ec1401..93beb6d`); the v0 4/2 harness source and `verify_run.py`
  (`matched_eval_v042_source.tar.gz`).
- Per-episode records stay on the evaluation host under `wam_baselines/runs/loopwam_v0_v4a2_matched` and
  `wam_baselines/runs/elastic_matched_20261101/final_rollout`; `data/episodes_*.csv` export every episode (suite, task,
  seed, initial state, success, policy steps, queries, action digest).
