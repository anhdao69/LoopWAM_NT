# Residual Transport: H100 evaluation of the missing conditions (2026-10-13)

**Status: IN PROGRESS (interim snapshot, 2026-10-13 05:25 EDT).** 4 of 14 conditions are complete; the rest are running or queued. Rows without numbers have not been evaluated yet. A blank never means zero.

[Results tables](Residual_Transport_Results.md) · [Remote campaign report](../../evaluation_report/campaign/report.html) · [Evidence](../residual_transport_evidence/h100_eval/)

## Scope

The remote campaign (`evaluation_report/campaign`, RTX PRO 6000) completed these RT conditions:

| Run | Schedules completed remotely |
|---|---|
| RT-A | S1, S2, S3 |
| RT-B4 | S1 |

RT-B4 S2 stopped at 2,505/6,000 episodes, and the remote eval queue was stopped. RT-A S4 failed with a missing manifest. The other RT conditions were cancelled.

This work evaluates the 14 missing conditions on the H100 cluster:

| Run | Schedules |
|---|---|
| RT-A | S4 |
| RT-T1ft | S1, S2 |
| RT-Pa | S1, S2 |
| RT-TF | S1, S2 |
| RT-A2 | S1, S2, S5, S6, S7, S8 |
| RT-B2a | parent2 |

RT-B4 S2 is **not** re-run here. The remote branch snapshot contains only its partial result.

## Weights

Weights come from Hugging Face `anhdao69/ResidualTrans` at revision `318ced3b8bfe` (the latest revision, unchanged since 2026-10-12). They are stored in `/groups/yshang/an221229/checkpoints/LoopWAM_NT/<run>/epoch_10.pt`.

- All 66 hub files passed the size and LFS-SHA256 check.
- Each `epoch_10.pt` SHA256 equals the hash the remote campaign recorded for that run:

| Run | SHA256 prefix |
|---|---|
| RT-A | `15ba0de1` |
| RT-T1ft | `efb19b69` |
| RT-Pa | `17f089ae` |
| RT-TF | `c6cdba96` |
| RT-A2 | `25ad8ca4` |
| RT-B2a | `25175895` |

- The runner rechecks these hashes before every launch.
- It also validates the epoch-10 training contract (`validate_transport_checkpoint`), the Wan VAE hash (`38071ab5…`) and the normalization hash (`d5ad351e…`).

## Matched protocol

The original campaign runner (`campaign_runner_fast.py`) lives only on the remote server. `scripts/evaluate_rt_matched.py` reimplements the protocol documented in `evaluation_report/audit_setup.py` and the campaign manifests.

**Grid**
- Four suites × 10 tasks × official initial states 0–49 × evaluation seeds 42/43/44.
- That is 6,000 episodes per condition and 1,500 per suite.

**Seeding and environment lifecycle**
- One fresh `OffScreenRenderEnv` per task/seed group, created with `env.seed(base seed)` and reused for its 50 initial states in order.
- The global Python/NumPy/Torch seed is set once per group (`set_global_seed`, rank 0).
- Every action query uses the base seed (fixed noise). Episodes are never reseeded.

**Rollout**
- 30 settling actions `[0,0,0,0,0,0,-1]`; their `done` flag is ignored, as in the Light-WAM loop.
- Policy steps are capped at 400 for Spatial/Object/Goal and 700 for Long.
- Each query predicts 32 actions; 10 are executed before replanning.
- Success is the environment `done` flag.

**Policy path**
- Unchanged `ObservationAdapter` (two 224 views concatenated horizontally, 180° rotation, canonical normalization, gripper threshold 0.5).
- Unchanged `TransportInference` with an FP32 model and BF16 autocast.

**Speed (does not change results)**
- CUDA-graph replay, checked bit-exact against eager on 24 real-observation queries (`graph_preflight_RT-A_S4_S2.json`). It is about 2.8× faster per query (65 ms vs 181 ms).
- 16 persistent workers on 4 H100s.
- Rollout videos are disabled for the scored run. Videos never affect outcomes; the remote campaign saved them only as evidence.

**Environment**
- torch 2.7.1+cu128, mujoco 3.3.2, robosuite 1.4.0, LIBERO revision `8f1084e3` — the same as the remote simulator revision.
- MuJoCo renders through EGL.

## Parity check

Before scoring, I re-ran RT-A S2 on H100 for 400 episodes (Long and Spatial, tasks 0–3, seed 42). I compared each episode with the remote campaign's record (`parity_RT-A_S2_vs_remote.txt`):

| | Success |
|---|---|
| H100 | 97.0% (388/400) |
| Remote | 98.0% (392/400) |

- 95.5% of episodes have the same outcome.
- 7 episodes succeed only on H100 and 11 only remotely. With 18 discordant pairs, the 1-point gap is not significant (McNemar p≈0.5).
- Exact step counts match in only 14% of episodes. Different GPU architectures change BF16 rounding, so trajectories diverge, but aggregate behaviour agrees.

**Caveat for comparisons that span the two servers:** read differences of about 1 point with care. Comparisons within H100 runs have no hardware confound.

Text embeddings: the H100 cache has the same 40 prompts and valid-token counts as the training manifest. Its file hashes differ because `torch.save` archives embed a random serialization ID, so bitwise tensor identity can't be checked without the remote files.

## Results so far (epoch 10, fixed noise, success %)

| Run | Schedule | Action calls | Source | Spatial | Object | Goal | Long | Four-suite mean | Seed std |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| RT-A | S1 | 192 | remote | 96.40 | 98.80 | 95.33 | 87.33 | 94.47 | 0.50 |
| RT-A | S2 | 138 | remote | 96.53 | 98.33 | 95.40 | 86.20 | 94.12 | 1.33 |
| RT-A | S3 | 168 | remote | 96.47 | 98.73 | 95.20 | 85.80 | 94.05 | 1.31 |
| RT-B4 | S1 | 192 | remote | 95.53 | 98.87 | 95.67 | 87.20 | 94.32 | 1.23 |
| **RT-A** | **S4** | 156 | H100 | 95.40 | 98.40 | 95.00 | 86.67 | **93.87** | 1.28 |
| **RT-T1ft** | **S1** | 192 | H100 | 96.67 | 98.47 | 94.93 | 88.20 | **94.57** | 1.27 |
| **RT-T1ft** | **S2** | 138 | H100 | 96.73 | 98.87 | 95.67 | 85.47 | **94.18** | 0.50 |
| **RT-Pa** | **S1** | 192 | H100 | 96.73 | 98.40 | 94.80 | 85.87 | **93.95** | 0.87 |
| RT-Pa | S2 | 138 | H100 | running | | | | | |
| RT-TF | S1, S2 | | H100 | queued | | | | | |
| RT-A2 | S1, S2, S5, S6, S7, S8 | | H100 | queued | | | | | |
| RT-B2a | parent2 | 60 | H100 | queued | | | | | |

Seed std is the sample standard deviation of the three per-seed four-suite means.

## Paired comparisons so far

Paired differences are matched by (suite, task, initial state, seed). The 95% interval is a cluster bootstrap over the 2,000 (suite, task, initial state) clusters, with 10,000 resamples. Full output: `paired_comparisons.json`.

| Decision | Comparison | Difference (pp) | 95% CI | Rule | Verdict |
|---|---|---:|---|---|---|
| Gate 1 / Phase-1 R2 | RT-A S1 − RT-B4 S1 (both remote) | +0.15 | [−0.45, +0.78] | ≥ +1 (Gate 1); ≥ +2 with CI > 0 (R2) | **Fail** at S1; S2 unavailable (RT-B4 S2 partial) |
| Learned transport | RT-A S1 − RT-T1ft S1 (cross-hardware) | −0.10 | [−0.75, +0.55] | ≥ +1 | **Fail** |
| Learned transport | RT-A S2 − RT-T1ft S2 (cross-hardware) | −0.07 | [−0.72, +0.57] | ≥ +1 | **Fail** |
| Transport-only scope (S1, descriptive) | RT-Pa S1 − RT-A S1 (cross-hardware) | −0.52 | [−1.17, +0.13] | RT-Pa within 1 point of RT-A (rule is at S2) | — (S2 pending; RT-B5 not evaluated) |
| Schedule sweep | RT-A S4 − RT-A S2 (cross-hardware) | −0.25 | [−0.85, +0.35] | descriptive | — |

The remaining decision rows will be filled when their conditions finish.

## How the remaining work continues

- Interactive allocation 898977 (4 H100, 16 CPUs) ends 2026-10-13 05:55.
- Batch job **903898** (`scripts/run_rt_missing_eval.sbatch`: 4 H100, 32 CPUs, 1 day) is queued to continue.
- The runner writes each task/seed group atomically and claims groups with heartbeated lock files. A restart or a concurrent allocation never repeats or mixes work, and each group's identity (checkpoint hash, schedule, backend, protocol) is checked on resume.
- To resume anywhere: `bash scripts/run_rt_missing_eval.sh`.
- Progress: `python scripts/rt_eval_status.py`.
- Rebuild this evidence: `python scripts/build_rt_h100_report.py`.
- Outputs: `/groups/yshang/an221229/checkpoints/LoopWAM_NT/rt_eval_h100/final_rollout/<condition>/groups/*.json`.

Latency (p50, batch 1, CUDA graph, 500 replayed observations) will be measured on an idle H100 with `scripts/measure_rt_latency.py` after scoring finishes.
