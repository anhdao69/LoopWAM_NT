# ChronoLoop: results to fill

Fill this in from the evaluation server. Training and implementation details are in
[ChronoLoop_Implementation_and_Launch_20261010.md](ChronoLoop_Implementation_and_Launch_20261010.md).
The plan is [../ChronoLoop.md](../ChronoLoop.md).

## How to evaluate

Use the matched harness for every number below:
- 4 suites × 50 official initial states × evaluation seeds 42, 43, 44.
- 700 policy steps, 30 settling steps, replan every 10 actions, 10 denoising steps, CFG 1.
- Memory reset at episode start.
- **Fixed noise** (the same noise vector at every query) is primary; **fresh noise** is secondary.
- Run Long first; the other suites after.

```bash
# checkpoints: https://huggingface.co/anhdao69/ChronoLoop/tree/main/<folder>/epoch_XX/policy.pt
torchrun --standalone --nproc_per_node=4 scripts/evaluate_chronoloop_libero.py \
  --checkpoint <folder>/epoch_10/policy.pt --suite libero_10 --episodes-per-task 50 \
  --seed 42 --noise fixed --data-dir <dir with dataset_stats.json + data_manifest.json> \
  --output-dir eval/<run>/libero_10_e10_seed42_fixed
```

`--data-dir` needs normalization hash `d5ad351e…`, which is identical to the parent release.

Compare paired by (task, state, seed), with a cluster bootstrap over (task, state): 10,000 resamples,
95% CI. Unless a column says otherwise, report the final checkpoint (epoch 10).

## Runs and Hugging Face folders

| Run | Definition | Hugging Face folder | Memory params | Updates (10 ep) | Train layout | Train GPU-h |
|---|---|---|---|---|---|---|
| CL-0 | No memory (control), K_a = 4 | `CL-0_no-memory_a4` | 0 | 23,862 | 2 GPU interactive + continuation | |
| CL-A | Memory written by the core loops, K_a = 4 | `CL-A_mem16-learned-loopwrite_a4` | 31,008 | 23,862 | 2 GPU interactive + continuation | |
| CL-REG | 16 registers reset every query | `CL-REG_mem16-reset-registers_a4` | 31,008 | 23,862 | Slurm 2 or 4 GPU | |
| CL-W2 | Read-only memory, external 1-block updater | `CL-W2_mem16-learned-external-updater_a4` | 37,818,144 | 23,862 | Slurm 2 or 4 GPU | |
| CL-0@1 | CL-0 with K_a = 1 | `CL-0-at1_no-memory_a1` | 0 | 23,862 | Slurm 2 or 4 GPU | |
| CL-A@1 | CL-A with K_a = 1 | `CL-A-at1_mem16-learned-loopwrite_a1` | 31,008 | 23,862 | Slurm 2 or 4 GPU | |
| CL-FRAME | No memory; frame from query k−3 as an extra gated frame | `CL-FRAME_history-frame-k3_a4` | 288 | 23,862 | Slurm 2 or 4 GPU | |
| CL-ORACLE | Conditional on Gate 1 failing; not trained | — | | | | |

To fill **Train GPU-h**, sum `elapsed_training_seconds` × GPUs over each run's jobs; they are in
`timing.json` and the job logs.

## Per-run success (%), fixed noise unless stated

| Run | Long @ upd 2k (early) | Long e8 | Long e9 | Long e10 | Long e10 fresh | Spatial | Object | Goal | 4-suite avg | Task 6 | Task 8 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Parent v0 4/4 (reference) | — | | | 87.87 | | 96.93 | 98.73 | 96.13 | 94.92 | 62.7 | 60.0 |
| Light-WAM (reference) | — | | | 90.87 | | | | | | 82.0 | 75.3 |
| CL-0 | | | | | | | | | | | |
| **CL-A** | | | | | | | | | | | |
| CL-REG | | | | | | | | | | | |
| CL-W2 | | | | | | | | | | | |
| CL-0@1 | | | | | | | | | | | |
| CL-A@1 | | | | | | | | | | | |
| CL-FRAME | | | | | | | | | | | |
| CL-ORACLE (if run) | | | | | | | | | | | |
| CL-TF (eval only, CL-0 ckpt; not implemented) | — | | | | | | | | | | |

The Task 6 and Task 8 columns are Long success on those two tasks only, at e10 with fixed noise.
The early checkpoint, `update_002000.pt`, is kept on the training server only and is not uploaded.

## Per-seed detail (Long, e10, fixed noise)

| Run | Seed 42 | Seed 43 | Seed 44 | Mean | SD |
|---|---|---|---|---|---|
| CL-0 | | | | | |
| CL-A | | | | | |
| CL-REG | | | | | |
| CL-W2 | | | | | |
| CL-0@1 | | | | | |
| CL-A@1 | | | | | |
| CL-FRAME | | | | | |

## Questions (Long, e10, fixed noise)

| Question | Comparison | Δ (95% CI) | Pass rule | Pass? |
|---|---|---|---|---|
| Gate 1 | CL-A − CL-0 | | ≥ +2 → run round 3 | |
| Q1a: memory helps | CL-A − CL-0 | | ≥ +3 and CI lower bound > 0 | |
| Q1b: it is the carried state | CL-A − CL-REG | | ≥ +2 and CI lower bound > 0 | |
| Diagnostic (only if Gate 1 fails) | CL-ORACLE − CL-0 | | ≥ +2: revise the write; < +2: inconclusive | |
| Q2: loop writing matters | CL-A − CL-W2 | | ≥ +2 and CI lower bound > 0 | |
| Q3a: memory matters more at low depth | (CL-A@1 − CL-0@1) − (CL-A − CL-0) | | CI lower bound > 0 | |
| Q3b: 1 action loop is enough with memory | CL-A@1 − CL-0 | | CI lower bound > −1.5 | |
| Frame-stacking baseline | CL-A − CL-FRAME | | ≥ 0 | |

## Efficiency

Latency is batch 1, raw observation to action on the host, 500 replayed observations, H100.
Report p50 and state whether a CUDA graph was used. The ChronoLoop inference path in this branch is eager.

| Run | Latency p50 (ms) | Latency p95 (ms) | Peak inference memory (GB) | Policy params | Effective action depth | Train GPU-h |
|---|---|---|---|---|---|---|
| Parent v0 4/4 | 126 (reference) | | | 584,536,135 | 30 | 48.4 (parent, 10 ep from scratch) |
| Parent v0 4/1 | 70 (reference) | | | 584,536,135 | 12 | |
| CL-0 | | | | 584,536,135 | 30 | |
| CL-A | | | | 584,567,143 | 30 | |
| CL-A@1 | | | | 584,567,143 | 12 | |
| CL-W2 | | | | 622,354,279 | 30 | |
| CL-FRAME | | | | 584,536,423 | 30 | |

The evaluator logs per-query latency in each episode JSON (`query_latency_ms_p50`). That figure
includes VAE encoding, but it is not the plan's isolated 500-observation replay.

## Training sanity (already measured)

| Run | Update-1 loss video / action | Notes |
|---|---|---|
| CL-0 | 0.06120 / 0.01718 | 2-GPU vs 4-GPU agree within 1e-7 (video) |
| CL-A | 0.06120 / 0.01722 | equal to parent at init (gate closed) |
| CL-A@1, CL-0@1 | 0.0612 / 0.336 | removing 3 action loops from the 4/4 parent; falls to 0.236 by update 3 |

For each training run, `metrics.jsonl` logs loss, valid windows per suite, tanh(alpha) per block,
memory RMS and saturated fraction.
