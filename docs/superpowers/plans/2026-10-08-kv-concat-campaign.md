# All-loop action KV and continuous campaign implementation plan

**Goal:** Implement aligned/concat/mix safely, verify Long 4/1 fairness, then run
2Long+6full-suite fresh training/evaluation experiments in two four-run Slurm jobs.
**Spec:** User's detailed KV_concat prompt; fixed contract and decision rule in
`plans/performance/LoopWAM_v0_V4A1_KVconcat_20261008.md`.
**Architecture:** Keep aligned execution intact. New differentiable video-cache
training and mode-specific action slot sets share logic with cached inference.
Mode/depth are checkpoint contracts. Persistent rollout workers preserve per-
episode inference. Gated queues, cross-job Long barrier, automatic analysis.
**Tech stack:** PyTorch2.7,CUDA,H100,LIBERO,Slurm,Python.

## Tasks
- [ ] 1: Core modes and tiny numerical/gradient/mask tests, including generic aligned equivalence.
- [ ] 2: Strict checkpoint/contract/CLI propagation; mix no-decay group and exact parameter counts; legacy regression.
- [ ] 3: Fixed heldout concat attention diagnostics and mix weights, RNG-neutral; latency breakdown and compiled inference.
- [ ] 4: Four-run campaign with 42/43/44 evaluations, fairness/source gates, full-suite dense controls, and Long completion barrier.
- [ ] 5: Statistical report generator (Wilson, exact paired McNemar, two-proportion test, losses, diagnostics/latency plots); pre-register decision rule.
- [ ] 6: Archive CPU/GPU suites, benchmark each new mode/backends, cold/warm native smokes, simulator smokes, fairness checks. Diagnose any gate failure.
- [ ] 7: Pin source and launch verified fresh production in both existing replacement allocations; inspect finite initial updates, estimate full finish times.
- [ ] 8: Whole-branch review, implementation report/evidence sync and KV_concat push. Leave LoopWAM_NT unchanged.

## Review focus
No future-token leakage, no detached observation KV, identical aligned paths,
mode mismatch must fail before loading, mix logits must receive gradients with
no decay, diagnostics must preserve RNG, partial runs must never enter final
analysis, missing episodes must stop queues. Match every dataset/asset/hash
field against baseline, not only counts. No historical outputs overwritten.

## Rulings
- Three evaluation seeds now means42/43/44 for every model, superseding repeated42.
- Allocate Dense-S30 to the lighter full-loop pair and Dense-S12 to the heavier
  pair to balance total work with exactly four training runs per job.
- Existing Long controls may evaluate while implementation proceeds; they do
  not count as additional training runs and use pinned pre-change evaluator.
- Borderline expansion uses seed42,50states/task; retain 300-episode result and
  label any expanded analysis separately; no invented follow-up decision rule.
