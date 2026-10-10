# Residual Transport results — fill in after evaluation

[Implementation report](Residual_Transport_Implementation.md) · [Detailed CSV](Residual_Transport_Results.csv) · [Original protocol and decision rules](../residual_transport.md#6-results-to-fill)

All measured-result cells are intentionally blank. Action calls are theoretical action-block forward calls per query, computed as sum(6 + 6 × core loops). They are not measured latency. Historical scores in the proposal have not been copied into this matched evaluation table.

## How to fill

- Run Long first on the separate evaluation server. Use all four suites, 50 official initial states per task, evaluation seeds 42/43/44, and the same harness for all comparisons.
- Fixed noise is primary; fresh noise is secondary. Report success in percent (0–100). The four-suite mean is the arithmetic mean of the four suite success rates; keep per-suite results in the CSV.
- For latency use batch 1, the same GPU type, one CUDA graph per schedule, cached text K/V for every method, and 500 replayed observations. Measure raw observation to host action; record p50 in ms.
- Record checkpoint name/hash and report path. Evaluate retained epochs 8/9/10 separately; do not silently choose the best test result. Use a declared checkpoint selection rule. The CSV supplies separate rows for each checkpoint and an early update-2,000 row.
- The Markdown table is a summary: fill the checkpoint column before adding final scores. Long @ 2k is an early read only; never use it for a gate.
- The CSV defaults to fixed noise. Duplicate each relevant row with noise_mode=fresh for secondary measurements, and add per-seed rows if desired. Keep checkpoint/seed identifiers intact.
- Compute paired differences by (task, state, seed), with a cluster bootstrap over (task, state) and 95% confidence intervals. Equal latency means p50 within ±5%; otherwise use the interpolated RT-B1 line.
- Blank means not evaluated, never zero. Deferred rows document implemented recipes and do not imply submission.

## Run inventory

| Run | Transport / scope | Regime | Training schedules | Step conditioning | Submission |
|---|---|---|---|---|---|
| RT-A | T4 / pb | fu | S1, S2 | Off | Job 4847 |
| RT-B4 | T0 / pb | fu | S1, S2 | Off | Job 4846 |
| RT-T1ft | T1 / pb | fu | S1, S2 | Off | Job 4849 |
| RT-Pa | T4 / pa | fu | S1, S2 | Off | Job 4850 |
| RT-TF | T4 / pb | tf | S1, S2 | Off | Job 4851 |
| RT-A2 | T4 / pb | fu | S1, S2, S5, S6, S7, S8 | On | Job 4861 |
| RT-B2a | T0 / pc | fu | parent2 | On | Job 4859 |
| RT+B2 | T4 / pb | fu | S8, S7 | On | Gated; not submitted |
| RT-B4-2 | T0 / pb | fu | S1, S2, S5, S6, S7, S8 | On | Gated; not submitted |
| RT-T3 | T3 / pb | fu | S1, S2 | Off | Gated; not submitted |
| RT-T5 | T5 / pb | fu | S1, S2 | Off | Gated; not submitted |
| RT-T6 | T6 / pb | fu | S1, S2 | Off | Gated; not submitted |
| RT-Pc | T4 / pc | fu | S1, S2 | Off | Gated; not submitted |
| RT-NFE1 | T0 / pc | fu | parent1 | On | Gated; not submitted |
| RT-no-local | T4 / pb | fu | S1, S2 | Off | Gated; not submitted |
| RT-no-end | T4 / pb | fu | S1, S2 | Off | Gated; not submitted |
| RT-no-grip | T4 / pb | fu | S1, S2 | Off | Gated; not submitted |
| RT-no-anchor | T4 / pb | fu | S1, S2 | Off | Gated; not submitted |
| RT-A-seed43 | T4 / pb | fu | S1, S2 | Off | Gated; not submitted |
| RT-A-seed44 | T4 / pb | fu | S1, S2 | Off | Gated; not submitted |

## Evaluation-only baselines

| Run | Schedule | Action calls | Final checkpoint/hash | p50 ms | Long @ 2k % | Long fixed % | Long fresh % | Four-suite mean % | Report |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| RT-B0 | parent10 | 300 | parent /  | | N/A | | | | |
| RT-B0-20 | parent20 | 600 | parent /  | | N/A | | | | |
| RT-B1 | parent5 | 150 | parent /  | | N/A | | | | |
| RT-B1 | parent4 | 120 | parent /  | | N/A | | | | |
| RT-B1 | parent2 | 60 | parent /  | | N/A | | | | |
| RT-B1 | parent1 | 30 | parent /  | | N/A | | | | |
| RT-B3 | S1 | 192 | parent /  | | N/A | | | | |
| RT-B3 | S2 | 138 | parent /  | | N/A | | | | |
| RT-B5 | S1 | 192 | parent /  | | N/A | | | | |
| RT-B5 | S2 | 138 | parent /  | | N/A | | | | |
| RT-B6 | S1 | 192 | parent /  | | N/A | | | | |
| RT-B6 | S2 | 138 | parent /  | | N/A | | | | |

## Phase 1

| Run | Schedule | Action calls | Final checkpoint/hash | p50 ms | Long @ 2k % | Long fixed % | Long fresh % | Four-suite mean % | Report |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| RT-A | S1 | 192 |  | |  | | | | |
| RT-A | S2 | 138 |  | |  | | | | |
| RT-A | S3 | 168 |  | | N/A | | | | |
| RT-A | S4 | 156 |  | | N/A | | | | |
| RT-B4 | S1 | 192 |  | |  | | | | |
| RT-B4 | S2 | 138 |  | |  | | | | |
| RT-T1ft | S1 | 192 |  | |  | | | | |
| RT-T1ft | S2 | 138 |  | |  | | | | |
| RT-Pa | S1 | 192 |  | |  | | | | |
| RT-Pa | S2 | 138 |  | |  | | | | |
| RT-TF | S1 | 192 |  | |  | | | | |
| RT-TF | S2 | 138 |  | |  | | | | |

## Phase 2

| Run | Schedule | Action calls | Final checkpoint/hash | p50 ms | Long @ 2k % | Long fixed % | Long fresh % | Four-suite mean % | Report |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| RT-A2 | S1 | 192 |  | |  | | | | |
| RT-A2 | S2 | 138 |  | |  | | | | |
| RT-A2 | S5 | 102 |  | |  | | | | |
| RT-A2 | S6 | 78 |  | |  | | | | |
| RT-A2 | S7 | 48 |  | |  | | | | |
| RT-A2 | S8 | 42 |  | |  | | | | |
| RT-B2a | parent2 | 60 |  | |  | | | | |
| RT+B2 | S8 | 42 |  | |  | | | | |
| RT+B2 | S7 | 48 |  | |  | | | | |
| RT-B4-2 | S1 | 192 |  | |  | | | | |
| RT-B4-2 | S2 | 138 |  | |  | | | | |
| RT-B4-2 | S5 | 102 |  | |  | | | | |
| RT-B4-2 | S6 | 78 |  | |  | | | | |
| RT-B4-2 | S7 | 48 |  | |  | | | | |
| RT-B4-2 | S8 | 42 |  | |  | | | | |

## Deferred recipes

| Run | Schedule | Action calls | Final checkpoint/hash | p50 ms | Long @ 2k % | Long fixed % | Long fresh % | Four-suite mean % | Report |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| RT-T3 | S1 | 192 |  | |  | | | | |
| RT-T3 | S2 | 138 |  | |  | | | | |
| RT-T5 | S1 | 192 |  | |  | | | | |
| RT-T5 | S2 | 138 |  | |  | | | | |
| RT-T6 | S1 | 192 |  | |  | | | | |
| RT-T6 | S2 | 138 |  | |  | | | | |
| RT-Pc | S1 | 192 |  | |  | | | | |
| RT-Pc | S2 | 138 |  | |  | | | | |
| RT-NFE1 | parent1 | 30 |  | |  | | | | |
| RT-no-local | S1 | 192 |  | |  | | | | |
| RT-no-local | S2 | 138 |  | |  | | | | |
| RT-no-end | S1 | 192 |  | |  | | | | |
| RT-no-end | S2 | 138 |  | |  | | | | |
| RT-no-grip | S1 | 192 |  | |  | | | | |
| RT-no-grip | S2 | 138 |  | |  | | | | |
| RT-no-anchor | S1 | 192 |  | |  | | | | |
| RT-no-anchor | S2 | 138 |  | |  | | | | |
| RT-A-seed43 | S1 | 192 |  | |  | | | | |
| RT-A-seed43 | S2 | 138 |  | |  | | | | |
| RT-A-seed44 | S1 | 192 |  | |  | | | | |
| RT-A-seed44 | S2 | 138 |  | |  | | | | |

## Scientific decisions

Submission overrides are not evidence of a passed scientific gate. Apply these rules to matched final evaluations.

| Decision | Comparison | Rule | Difference, percentage points (95% CI) | Pass / fail | Evidence |
|---|---|---|---|---|---|
| Gate 0 | Parent NFE 4 vs NFE 10, Long fixed | If NFE 4 is within 1 point, follow the specification's stop rule after round 1 | | | |
| Gate 1 | RT-A minus RT-B4, S1 or S2 | At least +1 point at either schedule | | | |
| Phase-1 R1 | RT-A S2 vs RT-B1 at equal latency | At least +2 points with CI lower bound > 0, or equal success at at least 20% lower latency | | | |
| Phase-1 R2 | RT-A minus RT-B4, each S1/S2 | At least +2 points and CI lower bound > 0 | | | |
| Learned transport | RT-A minus RT-T1ft, each S1/S2 | At least +1 point | | | |
| Transport-only scope | RT-Pa minus RT-B5 at S2 | At least +2 points, and RT-Pa within 1 point of RT-A | | | |
| FU vs TF | RT-A minus RT-TF at S2 | At least +1 point | | | |
| Phase-2 retention | RT-A2 minus RT-A, each S1/S2 | Within 1 point | | | |
| Phase-2 R1 | RT-A2 S6/S7 vs RT-B2a and RT-B1 line | Same criterion as phase-1 R1 | | | |
| Phase-2 R2 | RT-A2 minus RT-B4-2 at S6 | At least +2 points and CI lower bound > 0 | | | |
| Phase-2 R3 | RT+B2 S8 minus RT-B2a parent2 | CI lower bound > -1.5 points, at lower latency | | | |

R-FM has no executable definition and is excluded from the 20-recipe inventory.
