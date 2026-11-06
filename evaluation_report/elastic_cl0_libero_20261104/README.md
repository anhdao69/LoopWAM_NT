# CL-0 elastic distillation (CL0-E0/E1/E2): training and matched LIBERO evaluation

**Status: complete** (training finished 2026-11-03, evaluation finished 2026-11-04 05:36 UTC on H100 jobs 5623/5624). Main report: [cl0_eval_report.md](eval/cl0_eval_report.md).

The elastic E0/E1/E2 recipes (T0 / T1 / T4 transport, FLAT9 schedules) with the memoryless ChronoLoop CL-0 (`anhdao69/ChronoLoop` `CL-0_no-memory_a4`, LoopWAM v0 4/4 fine-tuned ten more LIBERO epochs) as frozen teacher and student initializer; teacher re-wrapped to loopwam-s-v1, sha256 `1c57b8d57961fb70...`. Code: branch `elastic_cl0`.

## Evaluated checkpoints

| run | export | sha256 | update | export audit | evaluation gate (eager vs CUDA-graph policy queries) |
|---|---|---|---|---|---|
| CL0-E0 | epoch_10 (T0) | `1d47ff90a2795704...` | 21,700 (status `complete`) | passed | 144 queries, all bit-exact: True |
| CL0-E1 | epoch_10 (T1) | `d2f192e4ed62986f...` | 21,700 (status `complete`) | passed | 144 queries, all bit-exact: True |
| CL0-E2 | epoch_10 (T4) | `8e235d15cfa23410...` | 21,700 (status `complete`) | passed | 144 queries, all bit-exact: True |

Evaluation parity with the BASE10 campaign's E0 reference runner: 150/150 episodes trajectory-identical ([parity_e0.json](eval/validation/parity_e0.json)).

## Results

27 conditions (3 runs x 9 schedules), each 6,000 episodes (4 LIBERO suites x 10 tasks x 50 episodes x seeds 42/43/44): 162,000 episodes. Rows for the BASE10 E0/E1/E2 reference come from the earlier campaign ([reference_hf.tar.gz](eval/reference_hf.tar.gz)).

| run | flat_k1_n2 (primary) | best schedule |
|---|---|---|
| CL0-E0 | 95.18% (5,711/6,000) | flat_k4_n10: 96.22% |
| CL0-E1 | 95.25% (5,715/6,000) | flat_k4_n2: 96.13% |
| CL0-E2 | 95.55% (5,733/6,000) | flat_k2_n10: 96.12% |

Primary contrasts, quoted from the report (paired permutation test, Holm-adjusted within each family):

| contrast | family | difference pp (95% CI) | p (Holm) | LIBERO-Long pp |
|---|---|---|---|---|
| CL0-E0-E0 @ flat_k1_n2 | primary (T) | +0.85 (+0.15 to +1.55) | 0.0191 | +1.53 |
| CL0-E1-E1 @ flat_k1_n2 | primary (T) | +1.52 (+0.77 to +2.23) | 0.0003 | +3.93 |
| CL0-E2-E2 @ flat_k1_n2 | primary (T) | +1.48 (+0.77 to +2.18) | 0.0004 | +4.07 |
| CL0-E1-CL0-E0 @ flat_k1_n2 | primary (R) | +0.07 (-0.60 to +0.72) | 0.8853 | +0.73 |
| CL0-E2-CL0-E0 @ flat_k1_n2 | primary (R) | +0.37 (-0.25 to +0.98) | 0.7901 | +1.27 |
| CL0-E2-CL0-E1 @ flat_k1_n2 | primary (R) | +0.30 (-0.33 to +0.93) | 0.7901 | +0.53 |

All 27 conditions with per-suite and per-seed rates and the secondary contrasts are in the report.

**Independent check** ([independent_check.json](independent_check.json), script `../verify_h100_additions_20261106.py`): PASSED - 27 conditions x 6,000 episodes: unique episode keys; per-episode CSV == per-group rollout JSON (final_rollout.tar.gz); overall, per-suite and per-seed success rates == cl0_eval_report.md.

## Contents

- `eval/`: report, per-episode CSVs (`report_csv/`), gates, parity, setup validation, LIBERO config, lane/runner logs, `final_rollout.tar.gz` (raw per-group rollout JSON, 3,240 groups + manifests) and `reference_hf.tar.gz`.
- `training/`: per-run config, manifest, status, launches, schedule sequence, train log and `metrics.jsonl.gz`; preflight / equivalence / benchmark / smoke validation; init metadata; eval launch tools.
- Checkpoints are not published to Hugging Face (`*.upload_pending` markers in `training/runs/` record that).

Copied 2026-11-06 20:50 UTC (H100) / 20:52 UTC (ws) without modifying any source. `SOURCES.tsv` lists every stored file with its source host/path and the SHA-256 and size of the original bytes (also for `.gz` files and tar members). Not copied: model checkpoints (`*.pt`), rollout videos, copied EGL libraries and latent caches.
