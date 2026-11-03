# Matched LIBERO evaluation — batch of 2026-11-01 (2x RTX PRO 6000)

Seven conditions requested on 2026-11-01, each scored on the full matched protocol `lightwam_20261009_matched`
(4 suites x 10 tasks x official initial states 0-49 x evaluation seeds 42/43/44 = **6,000 episodes per condition**),
the same protocol as every row of [`../matched_baselines`](../matched_baselines).

## 1. Results of this batch

| Condition | Success % (95% CI) | Spatial | Object | Goal | Long | Seeds 42 / 43 / 44 | Evaluated on |
|---|---|---:|---:|---:|---:|---|---|
| v0 4/2 (video 4 / action 2 export, NFE 10) | **94.53** (93.82–95.22) | 96.93 | 98.80 | 95.33 | 87.07 | 96.25 / 94.30 / 93.05 | RTX PRO 6000 (this batch) |
| E0 epoch 10, K_a 3, NFE 10 | **94.15** (93.38–94.90) | 96.80 | 99.07 | 95.53 | 85.20 | 94.45 / 94.15 / 93.85 | RTX PRO 6000 (this batch) |
| E0 epoch 10, K_a 5, NFE 10 | **95.03** (94.32–95.73) | 96.27 | 98.93 | 96.80 | 88.13 | 96.10 / 95.45 / 93.55 | RTX PRO 6000 (this batch) |
| E0 epoch 10, K_a 6, NFE 10 | **94.78** (94.05–95.48) | 96.53 | 99.07 | 96.60 | 86.93 | 95.55 / 95.25 / 93.55 | RTX PRO 6000 (this batch) |
| base v0 4/4, NFE 2 [4,4], no distillation | **95.78** (95.15–96.40) | 97.40 | 98.73 | 96.67 | 90.33 | 96.00 / 95.50 / 95.85 | RTX PRO 6000 (this batch) |
| base v0 4/4, NFE 5 [4]x5, no distillation | **95.73** (95.07–96.35) | 96.73 | 98.80 | 96.60 | 90.80 | 95.65 / 96.20 / 95.35 | RTX PRO 6000 (this batch) |
| base v0 4/4 at [1,1] (K_a 1, NFE 2), no distillation | **11.92** (10.75–13.10) | 18.53 | 22.87 | 6.27 | 0.00 | 13.25 / 11.00 / 11.50 | RTX PRO 6000 (this batch) |

The 95% intervals are cluster-bootstrap percentile intervals (cluster = suite, task and initial state with its three
seeds; 10,000 resamples), as declared in the analysis plan before any scored episode
([`ANALYSIS_PLAN_20261101.md`](ANALYSIS_PLAN_20261101.md), sha256 `245439595c058403…`).

**Findings** (all contrasts secondary and unadjusted; Section 2 has the intervals):

1. **v0 4/2** (94.53%) is statistically indistinguishable from the campaign's v0 4/4 (-0.38 pp, p 0.36) and 4/1
   (+0.53 pp, p 0.22), and better than 2/2 (+2.08 pp) and 1/4 (+2.85 pp, p 0.0001 for both), mostly on Long.
2. **E0 at test-time action depths** K_a 3, 5 and 6 (NFE 10) scores 94.15%, 95.03% and 94.78% (94.87% at its trained
   K_a 4 on H100). The undistilled base v0 at the same depths scores 89.47%, 94.60% and 89.20% on this host, so
   E0 - base v0 is +4.68 pp (p 0.0001), +0.43 pp (p 0.30) and +5.58 pp (p 0.0001). Against E0's own K_a 4 (H100,
   cross-host) only K_a 3 hints at a drop: -0.72 pp (p 0.044), almost all on Long (-3.5 pp).
3. **Base v0 4/4 with fewer denoising steps, no distillation:** NFE 2 ([4,4], 95.78%) and NFE 5 ([4]x5, 95.73%) both
   score above the same model at NFE 10 (campaign 94.92%): +0.87 pp (p 0.012) and +0.82 pp (p 0.0095), mostly on Long
   (+2.5 and +2.9 pp); [4,4] and [4]x5 do not differ (+0.05 pp, p 0.91). They are also not below E0 at the same
   schedules on H100 ([4,4] +0.82 pp, p 0.021; [4]x5 +0.28 pp, p 0.38; cross-host). On this benchmark, at the trained
   depth 4, cutting the denoising steps from 10 to 2 or 5 cost no success without distillation.
4. **Base v0 at [1,1]** (one action loop, NFE 2, no distillation) collapses to 11.92%, with 0.00% on Long, against
   94.33% for E0 at [1,1] on H100 (-82.42 pp). This is the model's own behaviour, not an evaluator fault: the elastic
   runner is bit-exact to native LoopWAM inference at this schedule (16/16 gate queries), every failed episode ran to
   the step cap, all 6,000 action sequences are distinct, and K_a 1 is the native code's own definition (one action pass
   reading video core loop 4, the same rule as the K_a 3/5/6 sweep). E0's advantage over the base model appears
   when the action depth differs from the base model's trained depth 4 (at K_a 1, 3 and 6, but not at 5; E0 was trained
   at K_a 1, 2 and 4), not when only the number of denoising steps is reduced.

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

| Contrast (secondary, unadjusted) | Kind | Difference, pp (95% CI) | p | Discordant clusters | Spatial | Object | Goal | Long |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E0 [3]x10 - base v0 Ka3 | same host | +4.68 (+3.77 to +5.62) | 0.0001 | 464/2000 | +1.27 | +6.47 | +2.60 | +8.40 |
| E0 [5]x10 - base v0 Ka5 | same host | +0.43 (-0.33 to +1.22) | 0.2958 | 294/2000 | +1.80 | +0.07 | +0.60 | -0.73 |
| E0 [6]x10 - base v0 Ka6 | same host | +5.58 (+4.58 to +6.60) | 0.0001 | 470/2000 | +8.40 | +2.13 | +2.87 | +8.93 |
| Z10 [4,4] - Z10 [4]x5 | same host | +0.05 (-0.53 to +0.63) | 0.9078 | 227/2000 | +0.67 | -0.07 | +0.07 | -0.47 |
| Z10 [4,4] - Z10 [1,1] | same host | +83.87 (+82.57 to +85.12) | 0.0001 | 1877/2000 | +78.87 | +75.87 | +90.40 | +90.33 |
| Z10 [4]x5 - Z10 [1,1] | same host | +83.82 (+82.52 to +85.10) | 0.0001 | 1873/2000 | +78.20 | +75.93 | +90.33 | +90.80 |
| Z10 [4,4] - campaign v0 4/4 | same hardware (campaign) | +0.87 (+0.18 to +1.53) | 0.0116 | 278/2000 | +0.47 | -0.00 | +0.53 | +2.47 |
| Z10 [4]x5 - campaign v0 4/4 | same hardware (campaign) | +0.82 (+0.22 to +1.42) | 0.0095 | 246/2000 | -0.20 | +0.07 | +0.47 | +2.93 |
| Z10 [1,1] - campaign v0 4/4 | same hardware (campaign) | -83.00 (-84.32 to -81.67) | 0.0001 | 1878/2000 | -78.40 | -75.87 | -89.87 | -87.87 |
| v0 4/2 - campaign v0 4/4 | same hardware (campaign) | -0.38 (-1.15 to +0.42) | 0.3558 | 338/2000 | +0.00 | +0.07 | -0.80 | -0.80 |
| v0 4/2 - campaign v0 4/1 | same hardware (campaign) | +0.53 (-0.28 to +1.35) | 0.2224 | 380/2000 | +0.67 | +0.73 | -0.07 | +0.80 |
| v0 4/2 - campaign v0 2/2 | same hardware (campaign) | +2.08 (+1.20 to +2.95) | 0.0001 | 425/2000 | +0.00 | +2.27 | +1.07 | +5.00 |
| v0 4/2 - campaign v0 1/4 | same hardware (campaign) | +2.85 (+1.92 to +3.77) | 0.0001 | 444/2000 | +0.93 | +1.00 | +0.60 | +8.87 |
| Z10 [4,4] - E0 [4,4] (H100) | cross host (H100) | +0.82 (+0.15 to +1.50) | 0.0211 | 264/2000 | +0.40 | -0.07 | +0.33 | +2.60 |
| Z10 [4]x5 - E0 [4]x5 (H100) | cross host (H100) | +0.28 (-0.30 to +0.88) | 0.3809 | 227/2000 | -0.07 | -0.47 | +0.27 | +1.40 |
| Z10 [1,1] - E0 [1,1] (H100) | cross host (H100) | -82.42 (-83.77 to -81.07) | 0.0001 | 1867/2000 | -77.93 | -76.07 | -89.00 | -86.67 |
| E0 [3]x10 - E0 [4]x10 (H100) | cross host (H100) | -0.72 (-1.38 to -0.05) | 0.0440 | 302/2000 | +0.27 | +0.13 | +0.27 | -3.53 |
| E0 [5]x10 - E0 [4]x10 (H100) | cross host (H100) | +0.17 (-0.47 to +0.80) | 0.6505 | 277/2000 | -0.27 | +0.00 | +1.53 | -0.60 |
| E0 [6]x10 - E0 [4]x10 (H100) | cross host (H100) | -0.08 (-0.77 to +0.60) | 0.8460 | 296/2000 | -0.00 | +0.13 | +1.33 | -1.80 |

## 3. Reference rows used by the contrasts

| Condition | Success % (95% CI) | Spatial | Object | Goal | Long | Seeds 42 / 43 / 44 | Evaluated on |
|---|---|---:|---:|---:|---:|---|---|
| base v0 Ka3 | **89.47** (88.43–90.47) | 95.53 | 92.60 | 92.93 | 76.80 | 92.00 / 89.35 / 87.05 | RTX PRO 6000 (2026-10-27/28) |
| base v0 Ka5 | **94.60** (93.88–95.32) | 94.47 | 98.87 | 96.20 | 88.87 | 94.90 / 94.50 / 94.40 | RTX PRO 6000 (2026-10-27/28) |
| base v0 Ka6 | **89.20** (88.18–90.23) | 88.13 | 96.93 | 93.73 | 78.00 | 87.95 / 90.80 / 88.85 | RTX PRO 6000 (2026-10-27/28) |
| campaign v0 4/4 | **94.92** (94.20–95.62) | 96.93 | 98.73 | 96.13 | 87.87 | 95.55 / 94.95 / 94.25 | RTX PRO 6000 (campaign) |
| campaign v0 4/1 | **94.00** (93.28–94.70) | 96.27 | 98.07 | 95.40 | 86.27 | 94.35 / 94.20 / 93.45 | RTX PRO 6000 (campaign) |
| campaign v0 2/2 | **92.45** (91.62–93.27) | 96.93 | 96.53 | 94.27 | 82.07 | 93.20 / 92.40 / 91.75 | RTX PRO 6000 (campaign) |
| campaign v0 1/4 | **91.68** (90.78–92.57) | 96.00 | 97.80 | 94.73 | 78.20 | 92.95 / 92.70 / 89.40 | RTX PRO 6000 (campaign) |
| E0 [4]x10 (H100) | **94.87** (94.17–95.55) | 96.53 | 98.93 | 95.27 | 88.73 | 95.70 / 95.10 / 93.80 | H100 (elastic evaluation) |
| E0 [4,4] (H100) | **94.97** (94.27–95.65) | 97.00 | 98.80 | 96.33 | 87.73 | 95.80 / 96.10 / 93.00 | H100 (elastic evaluation) |
| E0 [4]x5 (H100) | **95.45** (94.73–96.12) | 96.80 | 99.27 | 96.33 | 89.40 | 95.35 / 95.95 / 95.05 | H100 (elastic evaluation) |
| E0 [1,1] (H100) | **94.33** (93.58–95.07) | 96.47 | 98.93 | 95.27 | 86.67 | 95.25 / 94.05 / 93.70 | H100 (elastic evaluation) |

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

| Condition | Episodes | Code head in every group record | Gripper exactly 0.5 | Runner summary.json |
|---|---:|---|---:|---|
| v0 4/2 | 6000 | matched harness (`matched_eval/verify_run.py` passed: 6000 episode files, 120 groups, 18 builds, 0 issues) | n/a | matches |
| E0 [3]x10 | 6000 | 93beb6d (manifest 93beb6d) | 0 | matches |
| E0 [5]x10 | 6000 | 93beb6d (manifest 93beb6d) | 0 | matches |
| E0 [6]x10 | 6000 | 93beb6d (manifest 93beb6d) | 0 | matches |
| Z10 [4,4] | 6000 | 93beb6d (manifest 93beb6d) | 0 | matches |
| Z10 [4]x5 | 6000 | 93beb6d (manifest 93beb6d) | 0 | matches |
| Z10 [1,1] | 6000 | 93beb6d (manifest 93beb6d) | 0 | matches |

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
