# ChronoLoop — Experiment Plan

Oct 10, 2026 · @Anh Dao

ChronoLoop gives LoopWAM a small recurrent memory that the looped core writes at every replanning query and reads at the next one. Every version fine-tunes the same full-suite v0 4/4 checkpoint on all four LIBERO suites. Training runs as a sweep on 4 H100s; each finished checkpoint is evaluated on a separate server.

## 1. Motivation

Two LIBERO-Long tasks cause most of LoopWAM's remaining failures, and both need the policy to know which subgoal is already done.

- **Where the gap is:** full-suite v0 4/4 scores 87.87 on Long vs 90.87 for Light-WAM (matched harness, 3 seeds × 50 states).
  - Task 6 (white mug on plate + pudding right of plate): 94/150.
  - Task 8 (both moka pots on the stove): 90/150.
  - Together: 116 of 182 Long failures. On the other 8 Long tasks LoopWAM already matches Light-WAM (94.50 vs 93.92).
- **Hypothesis:** the policy only sees the current frame at each query (replan every 10 steps, up to 70 queries per episode). After the first subgoal it can lose track of progress, re-grasp the placed object or stall. A state carried across queries should fix this.
- **Caveat:** Light-WAM is also stronger on these two tasks, so memory may not be the only fix. CL-ORACLE (privileged stage label) is a conditional diagnostic, not a guaranteed upper bound, since the model must still learn to read the label. It tests the value of "knowing the stage".
- **Why it must be loop-specific:** recurrent memory tokens already exist ([μVLA](https://arxiv.org/abs/2606.12497), [MemBodied](https://arxiv.org/abs/2609.28256), [DIM-WAM](https://arxiv.org/abs/2606.27677), [MemoryWAM](https://arxiv.org/abs/2606.20562)). The plan therefore tests three questions:
  1. **Q1:** does a carried memory help at all (vs no memory and vs the same tokens reset every query)?
  2. **Q2:** is writing the memory through the shared core loops better than an external updater with more parameters?
  3. **Q3:** does memory let the action expert run 1 loop instead of 4 (measured 70 ms vs 126 ms per query) without losing success rate?

Q1 alone is an incremental result; Q2 or Q3 is what makes it a LoopWAM contribution.

## 2. Method: default version CL-A

&#91;embedded content: ChronoLoop architecture · one replanning query\]

The memory tokens ride through pre and the shared core with the observation; their state after the last core loop is squashed, mixed into s, and fed back at the next query. The coda and the action expert only read it.

### 2.1 Insertion into the video expert

- **Where:** right after `video_expert.prepare` (tokens already patchified, `freqs` and `t_mod` built).
  - Training sequence: \[m (16) | obs (392) | fut (784)\].
  - Inference prefill: \[m (16) | obs (392)\].
- **Mask:** set `structured_attention_observation_tokens = 408`, so memory belongs to the clean prefix F. F reads F; future video reads all video; actions read F + actions. Memory never sees future frames or noisy actions, identically in training and prefill.
- **RoPE:** identity rotation (angle 0) for memory tokens.
  - Do not use `freqs[-1]`: index −1 wraps to the last position.
  - Do not shift video frame indices: that changes the video↔action relative RoPE the parent learned.
- **t\_mod:** clean-frame modulation (t = 0), same as obs. Text cross-attention unchanged.
- **Strip** the 16 memory tokens before `video_expert.post`. The video loss stays on future tokens only.

### 2.2 Read-out and update (rule U1)

h\_k = memory hidden states after the last core loop, before the coda. Read it out in both `forward_joint_exits` (training) and `prefill_video_cache_tensor` (inference).

```latex
\hat w_k = c \tanh\!\left(\mathrm{LN}(h_k)/c\right), \qquad s_k = \lambda \odot s_{k-1} + (1-\lambda) \odot \hat w_k, \qquad m_{k+1} = \gamma \odot s_k + e
```

| Symbol | Shape | Init |
| --- | --- | --- |
| c | scalar | 3 |
| λ = σ(a) | a ∈ ℝ^1536, per channel, shared by slots | a = 2.2 (λ ≈ 0.9) |
| s\_{−1} | 16 × 1536 | 0 at episode start |
| e | 16 × 1536 slot embedding | N(0, σ\_obs²) |
| γ | ℝ^1536 | RMS of obs tokens after `prepare` / c |
| LN | affine LayerNorm | γ = 1, β = 0 |

The only guarantee is |s| ≤ c element-wise. No stability claim beyond that.

### 2.3 Gate G-a (exact identity at init)

For every non-memory query token, per head h:

```latex
\mathrm{out}_h = \mathrm{Attn}_h(q, K_{base}, V_{base}) + \tanh(\alpha_h)\, \mathrm{Attn}_h(q, K_{mem}, V_{mem}), \qquad \alpha_h = 0 \text{ at init}
```

- Separate α per expert: one per head per physical block in the video expert (12 × 12) and in the action expert (12 × 12), 288 scalars in total. Core blocks share α across loops.
- Both terms are mask-free flash calls, in the style of `structured_mixed_attention`.
- Memory tokens as queries attend jointly over \[memory, obs\], no gate.
- At init the model output equals the parent exactly. This is a unit test.

### 2.4 Inference loop (per query k)

1. Prefill \[γ⊙s\_{k−1} + e | obs\_k\]. Cache obs K/V and memory K/V **separately** per virtual layer; read out h\_k.
2. Compute s\_k from h\_k.
3. Denoise actions reading both caches (memory through G-a).
4. Execute 10 actions.

Reset s = 0 at episode start; settling steps never touch s. Keep s as a static buffer so CUDA graphs still work.

**Functional state (required for activation checkpointing):** the model takes s\_{k−1} as an argument and returns (outputs, h\_k, s\_k). Never write memory into a module attribute or global buffer inside a checkpointed region: backward recomputes the forward, and mutated state would give wrong gradients. The CUDA-graph static buffer is updated only by the evaluator, outside the model.

**CL-A config:** U1 · W0 · R0 · S0 · M = 16 · G-a · neutral RoPE · TBPTT S = 4 · memory horizon = whole episode.

## 3. Shared training setup

Every run uses this setup; section 4 lists only what each run changes.

| Item | Setting |
| --- | --- |
| Init | Full-suite v0 4/4 checkpoint (87.87 Long). No from-scratch runs |
| Data | Full LIBERO, all four suites mixed: 1,712 demos, 277,713 windows; window definition as the parent (9 frames at stride 4, 32-action chunk) |
| Global batch | **128 windows per update** = 32 streams × 4 consecutive windows (TBPTT S = 4). Nominal: windows past a traversal end are masked; log the valid count |
| Budget | 6,000 updates = 768,000 window slots (≈ 2.8 passes), identical for every run; report the valid (unmasked) count |
| Loss | Parent v0 flow-matching loss (video + action, same weights and shifts, target v = ε − a), averaged over the valid windows of the update |
| Optimizer | AdamW β = (0.9, 0.95), grad clip 1.0, cosine schedule with 3% warmup |
| Learning rates | Backbone 3e-5, weight decay 0.01. Memory parameters (e, γ, a, LN, α, and the CL-W2 updater) 3e-4, weight decay 0 |
| Precision | FP32 master weights, BF16 compute; EMA exactly as the parent; per-block activation checkpointing on |
| Checkpoints | At update 2,000 (early read, Long only) and 6,000 (final), with `--retain-epochs-from`. Decisions use the final checkpoint only. Contents in 3.5 |

### 3.1 GPU layouts

Both layouts train with the same global batch, data order, learning rate and update count, so their results are interchangeable.

| Layout | Runs in parallel | GPUs per run | Streams per GPU | Windows per GPU per update |
| --- | --- | --- | --- | --- |
| **2 × 2** (recommended) | 2 | 2 | 16 | 64 |
| 1 × 4 | 1 | 4 | 8 | 32 |

- DDP. Each rank owns whole streams; a stream never spans GPUs. Rank r takes streams r, r + R, r + 2R, … of the shared schedule file.
- If a rank's streams do not fit in memory, process them in micro-batches of streams (for example 16 = 2 × 8) with gradient accumulation. Never change the global batch, learning rate or update count.
- 2 × 2 is recommended because each round trains the two runs of one comparison together, so every finished round answers a question.

### 3.2 Stream sampler (truncated BPTT, S = 4)

1. **Build the traversal list once.** Every (demo, phase u) pair with u ∈ {0, …, 9} appears exactly once per epoch: 1,712 × 10 = 17,120 traversals, shuffled with seed + epoch. Concatenate epochs until the budget is covered.
   - A traversal of (d, u) visits windows τ = u, u + 10, u + 20, … of demo d.
   - So every window appears exactly once per epoch, as with the parent's i.i.d. sampler.
2. **Keep 32 streams.** Each stream pops the next traversal from the shared list and holds (traversal id, demo, phase, next query index, detached state s).
3. **Each update,** each stream runs its next 4 windows in order. Gradient flows through s across those 4 windows.
4. **After the segment:** detach s, store it, and advance 4 windows.
5. **When a traversal ends inside a segment:** mask the remaining slots. At the next update the stream pops a new traversal with s = 0.

Notes:

- **Do not** draw demos with probability proportional to length while also walking each whole demo: exposure would grow with N\_d² and over-sample the long, mostly LIBERO-Long, demos.
- Stride 10 matches "replan every 10 actions" at eval, and the state at every window is accumulated from the demo start, as at eval.
- The list is generated once from a fixed seed and saved. **Every run reads the same file**, including CL-0 (which ignores s) and the K\_a = 1 runs; streams map to ranks as in 3.1.
- Log valid windows per update and per suite. Each suite's share must match its share of the parent's windows within 1 point.

### 3.3 Tests that must pass before round 1

- **Init equivalence:** with G-a and α = 0, outputs equal the parent; with memory disabled, bit-exact.
- **No leakage:** changing future-video or action tokens leaves h unchanged; changing obs\_{k+1} leaves s\_k unchanged.
- **Train ↔ inference:** on a 5-query sequence, s\_k and actions from the training path equal those from the stateful prefill path.
- **Bounded state:** |s| ≤ c after 1,000 random large updates.
- **Temporal credit, with the gate opened:**
  - Use a fixture with α set to 0.5. At α = 0 memory gets no gradient, so the test would pass trivially.
  - The loss at window 4 must send non-zero gradient into s at windows 1–3.
  - With a detach after every query, that gradient must be exactly 0. The shared core still gets gradient through the within-query path, which is expected.
  - No gradient may cross a segment boundary.
- **Sampler:** in one epoch every (demo, window) appears exactly once; suite shares match the parent's.
- **Recompute safety:** gradients with activation checkpointing equal gradients without it on a 4-window segment.
- **Layouts:** 2-GPU and 4-GPU layouts give the same first-update loss and gradient norm within numerical tolerance.
- **Resume:** killing and resuming a run from a checkpoint reproduces the next 10 updates.
- **CUDA graph:** eager vs graph match over 70 consecutive queries.
- **Checkpoint guard:** refuse to load or evaluate when memory flags differ from the saved config.

### 3.4 Log every update

- Video and action loss.
- RMS(s) and the saturated fraction (|ŵ| > 0.95c).
- Mean λ.
- tanh(α) per block. Flag the run if it is still ≈ 0 after 1,000 updates.

### 3.5 Checkpoint contents

Saving only the model and optimizer is not enough, because streams carry state across updates. Save and restore:

- model, optimizer, EMA, LR scheduler, update count;
- sampler: traversal-list file hash, epoch, list cursor, and for every stream its stream id, traversal id, demo id, phase, next query index and detached s;
- RNG states (Python, NumPy, torch CPU and CUDA) for every rank;
- config hash of the memory flags, used by the checkpoint guard.

## 4. Runs (6 core + 1 conditional + 1 optional)

Six core runs answer whether the idea works (Q1), whether it is loop-specific (Q2) and whether it buys latency (Q3). CL-ORACLE runs only if Q1 fails, to diagnose why. Each run differs from its comparison partner in exactly one thing.

| Run | Definition | Compared with | Answers |
| --- | --- | --- | --- |
| CL-0 | No memory tokens; same traversal file, budget and K\_a = 4 | — | Control for everything |
| **CL-A** | Section 2 exactly (K\_a = 4) | CL-0 | Q1: does carried memory help? |
| CL-REG | 16 memory tokens with s = 0 at every query (plain registers, rule U3) | CL-A | Q1: is it the carried state, or just extra tokens? |
| CL-W2 | External updater: 1 Wan block (1536-d, init from the first core block), query = γ⊙s + e, keys = \[itself; obs tokens after the last core loop\], output = ŵ. Inside the backbone memory is read-only | CL-A | Q2: writing through the loops vs a separate updater with more parameters |
| CL-0@1 | CL-0 with action loops K\_a = 1 (late alignment as the 4/1 model) | CL-0 | Q3: cost of cutting action depth without memory |
| CL-A@1 | CL-A with K\_a = 1 | CL-0@1 and CL-0 | Q3: does memory replace action depth? |
| CL-ORACLE (conditional) | Run **only if Gate 1 fails**. s replaced by an embedding of the privileged stage label: number of satisfied BDDL goal conjuncts + which ones. Same 16 tokens and gate. Labels from sim replay of the demos (train) and sim predicates (eval) | CL-0 | Diagnostic. Positive: stage information helps and the write is the problem. Null: inconclusive, because the model must still learn to read the label |
| CL-FRAME (optional) | No memory; the observation from query k − 3 (30 steps earlier) is added as a second clean frame in F (+392 tokens; repeat the first frame when k < 3) | CL-A | Reviewer baseline: frame stacking |

**Eval-only, no training:** CL-TF. On the CL-0 checkpoint, F also attends the obs K/V of the previous 3 queries (FIFO). Answers whether training-free KV memory is enough.

**Deferred until Q1 passes:** update rule, TBPTT length, capacity, gate type, read path, horizon and init ablations. They tune the design but cannot decide whether the idea works. Seeds 43/44 also come only after Q1–Q3 are read.

## 5. Run plan on 4 × H100

### 5.1 Rounds (2 × 2 layout, recommended)

Only rounds 3 and 3′ wait for an eval; the other rounds start as soon as GPUs free up.

| Round | GPUs 0–1 | GPUs 2–3 | Starts when | Answers |
| --- | --- | --- | --- | --- |
| 1 | CL-0 | CL-A | Tests in 3.3 pass | First read of Q1 |
| 2 | CL-REG | CL-W2 | Round 1 training finished (no eval wait) | Q1 complete; Q2 |
| 3 | CL-0@1 | CL-A@1 | **Round 1 final eval is back and Gate 1 passes** | Q3 |
| 3′ | CL-ORACLE | free (give to RT) | **Round 1 final eval is back and Gate 1 fails** | Why Q1 failed |
| 4 | CL-FRAME (optional) | free (give to RT) | Round 3 training finished (no eval wait) | Frame-stacking baseline |

**1 × 4 layout:** CL-0 → CL-A → CL-REG → CL-W2 → Gate 1. If it passes: CL-0@1 → CL-A@1 → CL-FRAME. If it fails: CL-ORACLE.

### 5.2 Gate 1 (Long, fixed noise, final checkpoints)

- **CL-A − CL-0 ≥ +2:** pass; start round 3.
- **CL-A − CL-0 < +2:** start round 3′ (CL-ORACLE) and read round 2. Then:
  - **CL-ORACLE − CL-0 ≥ +2:** stage information helps but CL-A does not learn it. Revise the write (for example an auxiliary progress loss or a selective gate) before spending more runs.
  - **CL-ORACLE − CL-0 < +2:** inconclusive, because the model may also fail to read the label. Pause ChronoLoop. Do not claim that stage memory is useless.
- While a gate waits, give the free GPUs to Residual Transport rounds.
- **Early read:** evaluate each run's update-2,000 checkpoint on Long only for a quick look. Never use it for a decision.

### 5.3 Launch flags

Common to every run:

```bash
torchrun --nproc_per_node 2 scripts/train_chronoloop.py \
  --init <full-suite v0 4/4 checkpoint> --data libero_all \
  --global-batch 128 --streams 32 --tbptt 4 --updates 6000 \
  --schedule-file <shared stream schedule> --save-at 2000,6000 \
  --run-name <Run> <run flags below>
```

Use `CUDA_VISIBLE_DEVICES=0,1` and `2,3` for the two runs of a round, or `--nproc_per_node 4` for the 1 × 4 layout.

| Run | --memory-tokens | --mem-source | --mem-write | --action-loops | --history-frame |
| --- | --- | --- | --- | --- | --- |
| CL-0 | 0 | — | — | 4 | 0 |
| CL-A | 16 | learned | loop | 4 | 0 |
| CL-REG | 16 | reset | loop | 4 | 0 |
| CL-ORACLE | 16 | oracle | — | 4 | 0 |
| CL-0@1 | 0 | — | — | 1 | 0 |
| CL-A@1 | 16 | learned | loop | 1 | 0 |
| CL-W2 | 16 | learned | external | 4 | 0 |
| CL-FRAME | 0 | — | — | 4 | 3 |

Flag names are a proposal for the agent; what matters is that each run differs from its partner only in the listed column.

## 6. Results to fill

### 6.1 Eval protocol

- Matched harness: 4 suites × 50 official states × eval seeds 42/43/44, same step caps as the existing matched runs.
- **Fixed noise** (same noise vector every query) is primary; **fresh noise** (new noise per query) is secondary.
- Comparisons are paired by (task, state, seed), with a cluster bootstrap over (task, state), 10,000 resamples, 95% CI.
- Task 6 and task 8 columns: Long success on those two tasks only, fixed noise.
- Run Long first; the other three suites after.

### 6.2 Per-run results

| Run | Long @ 2k (early) | Long fixed | Long fresh | Spatial | Object | Goal | 4-suite avg | Task 6 | Task 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Parent v0 4/4 (reference) | — | 87.87 |  | 96.93 | 98.73 | 96.13 | 94.92 | 62.7 | 60.0 |
| Light-WAM (reference) | — | 90.87 |  |  |  |  |  | 82.0 | 75.3 |
| CL-0 |  |  |  |  |  |  |  |  |  |
| **CL-A** |  |  |  |  |  |  |  |  |  |
| CL-REG |  |  |  |  |  |  |  |  |  |
| CL-ORACLE |  |  |  |  |  |  |  |  |  |
| CL-0@1 |  |  |  |  |  |  |  |  |  |
| CL-A@1 |  |  |  |  |  |  |  |  |  |
| CL-W2 |  |  |  |  |  |  |  |  |  |
| CL-FRAME |  |  |  |  |  |  |  |  |  |
| CL-TF (eval only) | — |  |  |  |  |  |  |  |  |

### 6.3 Questions (Long, fixed noise)

| Question | Comparison | Δ (95% CI) | Pass rule | Pass? |
| --- | --- | --- | --- | --- |
| Q1a: memory helps | CL-A − CL-0 |  | ≥ +3 and CI lower bound > 0 |  |
| Q1b: it is the carried state | CL-A − CL-REG |  | ≥ +2 and CI lower bound > 0 |  |
| Diagnostic (only if Gate 1 fails) | CL-ORACLE − CL-0 |  | ≥ +2: stage information helps, so revise the write; < +2: inconclusive |  |
| Q2: loop writing matters | CL-A − CL-W2 |  | ≥ +2 and CI lower bound > 0 |  |
| Q3a: memory matters more at low depth | (CL-A@1 − CL-0@1) − (CL-A − CL-0) |  | CI lower bound > 0 |  |
| Q3b: 1 action loop is enough with memory | CL-A@1 − CL-0 |  | CI lower bound > −1.5 |  |
| Frame-stacking baseline | CL-A − CL-FRAME |  | ≥ 0 |  |

**Latency for Q3** (batch 1, CUDA graph, raw observation to action on the host, 500 replayed observations, p50): CL-0 \_\_\_\_ ms · CL-A \_\_\_\_ ms · CL-A@1 \_\_\_\_ ms. Parent references: 126 ms (4/4) and 70 ms (4/1).

**What the outcome means:** Q1 alone is an incremental memory result. Q1 + Q2 or Q1 + Q3 is a LoopWAM contribution worth seeds 43/44 and the deferred ablations.

