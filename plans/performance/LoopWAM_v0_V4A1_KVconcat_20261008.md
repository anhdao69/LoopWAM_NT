# LoopWAM v0 4/1: all-loop observation KV experiment

## Pre-registration — October 8, 2026

Base: `8d74c8df2eca4d165626830d90c2cc9bc56412b6`, fetched from the
`LoopWAM_NT` branch of `anhdao69/LoopWAM_NT` (local remote alias `loopwam`).
Implementation branch: **KV_concat**. The base branch is not merged into or
rewritten. The previous full-suite launch is preserved as historical evidence.

This tests access to the video trajectory while holding action depth at 12 block
applications. Concat and mix use video/action loops 4/1 and 30/12 effective depth.
Concat has no new parameters; mix adds 6 layers × 12 heads × 4 logits = 288.
All-loop KV contains only clean observation tokens, after the existing RoPE.
Prelude/coda conditioning stays unchanged. Default `aligned` behavior and old
checkpoints must be preserved.

### Decision rule fixed before new concat/mix outcomes

- Pooled concat SR **≥89% over 300 episodes**: **access hypothesis supported**;
  the 4/1 deficit is mainly lack of access to the video loop trajectory.
- Pooled concat SR **≤84%**: **action-depth hypothesis supported**.
- Otherwise: **inconclusive**; evaluate concat and aligned 4/1 using 50 initial
  states per task (500 episodes) before deciding. Use evaluation seed 42 for
  this prespecified expanded grid, report separately, and retain the original
  three-seed result. No new post hoc threshold is introduced for that grid.

These are the user's operational decision thresholds, not a causal proof.
All models use one training seed (42); training-seed variance is unmeasured.

## Fixed contract

Concat/mix: Long 344/44 split, 92,678 training windows/epoch, 10 epochs,
7,250 updates, 926,780 real windows, global batch 128, fresh canonical Wan
initialization and optimizer. AdamW 1e-4, (0.9,0.95), eps1e-8, decay0.01,
5% warmup/cosine and clip1. FP32 weights/moments, BF16 compute. Prefer the
baseline DDP microbatch8 × accumulation8 on two H100s; record any necessary
layout deviation. Mix logits alone have zero weight decay.

Evaluation seeds **42/43/44**, each ten tasks × ten initial states, 700 policy
steps, 30 settling steps, chunk32/replan10/denoise10/CFG1. Checkpoint hashes,
all episode identities/outcomes and videos are retained. Baseline 4/1 seeds43/44
are required; repeat seed42 is an evaluator reproduction check and is not
additional independent evidence. Also evaluate the fresh 4/4 repeat at43/44.

## Expanded two-job campaign

The user superseded jobs4768/4769 shortly after verified production startup.
Those runs were cancelled solely to reorder this user's work; output directories
were preserved. Replacement allocations are **4770 (worker-0)** and **4771
(worker-1)**, two H100s,32CPUs,256GiB each. They wait for verified launch scripts.

Provisional balanced order (four training runs each):

1. 4770: concat Long → full 4/1 → full 3/3 → full Dense-S12.
2. 4771: mix Long → full 1/4 → full 2/2 → full Dense-S30.

Every model trains then evaluates before the next model. Both Long experiments
must complete before either job advances to the full suite phase. All full runs
use 1,712 available demos,277,713 windows/epoch,10epochs,21,700updates,global128,
and three evaluation seeds42/43/44 across all four suites (1,200 episodes/model).
Every training starts fresh. Previous short production weights are not resumed.

## Status

Implementation and verification are in progress. No concat/mix training results,
latency measurements, or final success rates are claimed. Measured tests,
throughput, fairness gates and launch estimates will be appended. The campaign
will generate the final statistical report from complete episode records.
