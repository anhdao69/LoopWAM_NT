# Expanded Long KV follow-up

Original 300-episode decision remains **inconclusive**. No additional decision threshold was preregistered.

| Model | Successes/500 | SR | Wilson95 |
|---|---|---|---|
| concat | 404/500 | 80.80% | 77.12%–84.01% |
| aligned | 413/500 | 82.60% | 79.03%–85.67% |

Paired exact McNemar: `{"n": 500, "candidate_only": 60, "baseline_only": 69, "p_exact": 0.48135410525756583}`

Expanded seed42 grid includes the original first10states/task; do not pool these executions as independent evidence. No follow-up threshold was preregistered.
