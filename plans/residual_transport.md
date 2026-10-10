# Residual Transport — Experiment Plan

Oct 10, 2026 · @Anh Dao

Residual Transport (RT) warm-starts LoopWAM's action core across denoising steps: every step after the first starts from the previous step's core residual and runs 1–2 loops instead of 4. Every trained version fine-tunes the full-suite v0 4/4 checkpoint on all four LIBERO suites. Training runs as a sweep on 4 H100s; each finished checkpoint is evaluated on a separate server.

## 1. Motivation

Action denoising is most of LoopWAM's per-query compute, and its looped core restarts from scratch at every denoising step even though consecutive steps see almost the same input.

- **Where compute goes:** one query = 30 video block calls (prefill, once) + action calls. At 4/4 with NFE 10 that is 30 + 10 × (3 pre + 6 × 4 core + 3 coda) = 330 calls. Measured CUDA-graph latency: 126 ms at 4/4 (300 action calls) and 70 ms at 4/1 (120 action calls).
- **Idea:** the core's residual at step j is a good starting point for step j + 1. Starting there, later steps should need 1–2 core loops instead of 4. Example: schedule \[4, 1 × 9\] needs 138 action calls instead of 300.
- **Why only a looped model can do it:** a shared core can restart mid-depth and run a few correcting iterations. Feature caching ([TaylorSeer](https://arxiv.org/abs/2503.06923), DeepCache, FORA) instead skips compute with no correction, and a dense network has no layer to re-run.
- **What RT must beat:** the cheapest alternatives.
  - Fewer denoising steps: 4/4 at NFE 4 uses 120 action calls, NFE 2 uses 60.
  - Step distillation to NFE 2 or 1.
  - Fine-tuning the same model for fewer loops **without** transport (B4). If RT does not beat B4, the gain is fine-tuning, not transport.
- **Core tension:** the straighter the flow trajectories, the better the identity warm start works, but also the better few-step sampling works. RT only wins where trajectories bend, such as multi-modal action choices.

**Questions:**

1. **R1:** at equal latency, does RT reach higher success than NFE reduction, step distillation and depth-cut fine-tuning? Or equal success while at least 20% faster?
2. **R2:** does the gain come from transport? RT must beat B4 by at least 2 points at the same schedule.
3. **R3:** does RT stack with step distillation (NFE 2 student plus transport)?

**Two phases.** Phase 1 tests R1 and R2 at NFE 10 only, where every denoising step has the same length, so one input never has two targets. Phase 2 adds step-size conditioning and tests the low-NFE regime and R3; it runs only if R2 passes.

## 2. Method

RT changes only the **starting state** of the action core at each denoising step. Video prefill, pre, coda and the Euler update are unchanged.

&#91;embedded content: Residual transport · two consecutive denoising steps\]

The first step runs the full 4 loops from a cold start; every later step receives the previous residual through T and runs 1–2 correcting loops.

### 2.1 Current denoising loop (parent)

- N = 10 Euler steps with shift 1: τ\_j = 1 − j/N, Δτ = −1/N, z\_0 = ε \~ N(0, I) of shape 32 × 7 (the noisy action chunk; the core states p, h, r below are 32 × 512).
- Each step:
  1. `action_expert.prepare(z_j, τ_j)`.
  2. Pre (3 blocks, reads video pre slots) → p\_j.
  3. Core × K\_a: iteration r reads video loop max(1, K\_v − K\_a + r) (late alignment).
  4. Coda (3 blocks, reads the video coda slot), then `post` → v\_j.
- Update z\_{j+1} = z\_j + Δτ · v\_j; actions = z\_N; execute the first 10 of 32.
- The core does not re-inject its input: h^(0) = p, h^(r) = Core(h^(r−1)). Replacing h^(0) is all a warm start needs; no block changes.

### 2.2 Transport rule

```latex
h_{j+1}^{(0)} = p_{j+1} + g \odot T\left(r_j;\ \tau_j,\ \tau_{j+1},\ \Delta z_j\right), \qquad r_j = h_j^{(k_j)} - p_j
```

- Δz\_j = z\_{j+1} − z\_j = Δτ · v\_j, mapped to tokens by the frozen action embedding.
- p\_{j+1} is always added back: it is the only path for the new input z\_{j+1} into the core.
- **Transport is on only when step j + 1 runs fewer than 4 loops.** A 4-loop step starts cold, so schedule \[4\] × 10 has the parent's computation graph. It equals the parent only while LoRA = 0; training changes the function.
- With k loops, iteration r reads video loop max(1, K\_v − k + r): k = 1 reads only the deepest video loop, like the 4/1 model. The coda always reads the video coda slot.
- **Shared engineering fix for every version:** if the text-context K/V of the action cross-attention is recomputed at every step (as `action_conditioning_kv` was on the KV\_concat branch; check the main branch), cache it once per query.

### 2.3 Transport variants (axis T)

| ID | Definition | Init |
| --- | --- | --- |
| T0 | No transport: h^(0) = p\_{j+1} (cold start) | — |
| T1 | Identity: g = 1, T = r\_j | — |
| T2 | First-order forecast: T = r\_j + ρ(r\_j − r\_{j−1}), ρ = (τ\_{j+1} − τ\_j)/(τ\_j − τ\_{j−1}); T1 at j = 0 | — |
| T3 | Learned per-channel gate: g = σ(MLP(\[φ(τ\_j); φ(τ\_{j+1})\])), T = r\_j; φ = sinusoidal embedding | Bias so g ≈ 0.95 |
| T4 | T3 + low-rank time FiLM: T = r\_j + B((A r\_j) ⊙ (1 + f(φ(τ\_j), φ(τ\_{j+1})))), A: 512 → 32, B: 32 → 512 | B = 0, so T4 = T3 |
| T5 | T4 with f also taking the per-token embedding of Δz\_j | B = 0, so T5 = T3 |
| T6 | One cross-attention layer: queries from p\_{j+1}, keys/values from r\_j, added to T1 | Output projection 0, so T6 = T1 |

### 2.4 Depth schedules (axis Σ)

k\_j = number of core loops at step j. Action calls per query = Σ\_j (6 + 6 k\_j).

| Schedule | k per step | NFE | Action calls | Phase |
| --- | --- | --- | --- | --- |
| Parent 4/4 | \[4\] × 10 | 10 | 300 | Reference |
| Σ1 | \[4, 2 × 9\] | 10 | 192 | 1: train + eval |
| Σ3 | \[4, 3, 2, 2, 2, 1 × 5\] | 10 | 168 | 1: eval only |
| Σ4 | \[4, 4, 1 × 8\] | 10 | 156 | 1: eval only |
| Parent at NFE 5 | \[4\] × 5 | 5 | 150 | Reference |
| Σ2 | \[4, 1 × 9\] | 10 | 138 | 1: train + eval |
| Parent at NFE 4 | \[4\] × 4 | 4 | 120 | Reference |
| Σ5 | \[4, 2, 2, 2, 2\] | 5 | 102 | 2 |
| Σ6 | \[4, 1, 1, 1, 1\] | 5 | 78 | 2 |
| Parent at NFE 2 | \[4, 4\] | 2 | 60 | Reference |
| Σ7 | \[4, 2\] | 2 | 48 | 2 |
| Σ8 | \[4, 1\] | 2 | 42 | 2 |
| Parent at NFE 1 | \[4\] | 1 | 30 | Reference |

- **Phase 1** trains and evaluates only NFE-10 schedules. Every step covers Δτ = 0.1, so an input (z, τ) never has two different targets.
- **Phase 2** adds Σ5–Σ8 (NFE 5 or 2), which need step-size conditioning (2.5). NFE 5 replaces NFE 4 so that every step boundary lies on the parent's NFE-10 grid and the teacher is always the deployed parent.
- Latency is measured, not predicted (section 6).

### 2.5 Step-size conditioning (phase 2 only)

- **Why:** with several NFEs in one model, the same input gets different targets. The cold first step (z = ε, τ = 1, 4 loops) must output the average velocity over \[1, 0.9\] at NFE 10, \[1, 0.8\] at NFE 5 and \[1, 0.5\] at NFE 2. The same happens later, for example at τ = 0.5 in NFE 5 vs NFE 2.
  - Without knowing the step length, the network can only learn a compromise.
  - Worse, T3–T5 see τ\_{j+1} while the cold control RT-B4 does not, so RT could win just by knowing the step length. That would confound R2.
- **How:** add an embedding of the step length d\_j = τ\_j − τ\_{j+1} to the action expert's time modulation in every block, for **every** phase-2 run. Use a sinusoidal embedding + MLP with a zero-initialized last layer, so at init the model equals the parent.
- **Convention:** d = 0 means instantaneous velocity, as in [shortcut models](https://arxiv.org/abs/2410.12557). The anchor loss uses d = 0, so it no longer conflicts with step distillation.

## 3. Shared training setup

Every trained run uses this setup; section 4 lists only what each run changes.

| Item | Setting |
| --- | --- |
| Init | Full-suite v0 4/4 checkpoint. The video expert is frozen in every run |
| Teacher | The same checkpoint, frozen, on its deployed NFE-10 grid (step 0.1). No NFE-20 teacher |
| Data | Full LIBERO, all four suites, i.i.d. windows as the parent (277,713 windows); same seed and window order for every run |
| Per window | One video prefill without gradient and one noise ε, both shared by teacher and student |
| Global batch / budget | 128 windows per update; 6,000 updates = 768,000 windows (≈ 2.8 passes) |
| Phase 1 schedules | One per batch, drawn uniformly from {Σ1, Σ2} (both NFE 10) |
| Phase 2 schedules | Per run (section 4), with step-size conditioning on |
| Optimizer | AdamW β = (0.9, 0.95), grad clip 1.0, cosine schedule with 3% warmup; BF16 compute, FP32 master weights |
| Checkpoints | At update 2,000 (early read) and 6,000 (final). Save model, optimizer, EMA, LR scheduler, sampler cursor and RNG states of every rank, so a resumed run continues identically |

### 3.1 Targets

The target for a step from τ\_j to τ\_{j+1} is the teacher's average velocity over that step, starting from the student's own state:

```latex
\bar u_j = \frac{\Phi^{T}_{\tau_j \to \tau_{j+1}}(z_j) - z_j}{\tau_{j+1} - \tau_j}
```

- **Phase 1 (NFE 10):** Φᵀ is one parent Euler step, so ū\_j = vᵀ(z\_j, τ\_j). That is one parent call at the student's state, exactly the parent's own step from there.
- **Phase 2:** Φᵀ is d\_j / 0.1 parent Euler steps of 0.1 from z\_j (2 steps at NFE 5, 5 steps at NFE 2). Every step boundary lies on the 0.1 grid.
- **No gradient through the teacher.** Run every teacher call inside `torch.no_grad()` and detach the target. Frozen teacher weights alone do **not** stop gradient flowing into z\_j, which depends on the student.
- **Teacher cost per window:**
  - 10 parent passes for the local targets, whatever the schedule;
  - 10 more for the endpoint reference (the parent's NFE-10 rollout from the same ε), computed once and reused.
- **Measure before choosing a layout:** in the first 20 updates, log teacher passes, student forward/backward block calls, peak memory and seconds per update.

### 3.2 Training regimes (axis R)

| ID | Where the student's z\_j comes from | Target |
| --- | --- | --- |
| **R-FU** (default) | The student runs its own schedule from ε | ū\_j at the student's state |
| R-TF | The parent's own NFE-10 trajectory from ε (phase 1 only) | ū\_j at the parent's state |

**Gradient truncation.** Use one helper that detaches, at the same step boundary, every tensor the next step reads: z\_j, r\_j, r\_{j−1} (used by T2) and any cached transport input.

- z is detached at every step except the last 3, so the endpoint loss reaches 3 steps back.
- The transported residual chain is cut at the same boundary.

### 3.3 Loss

```latex
L = \frac{1}{N}\sum_j \lVert v_j - \bar u_j \rVert^2 + 0.5\, L_{end} + 0.1\, L_{grip} + 0.5\, L_{anchor}
```

- **L\_end:** MSE, in normalized action space, between the student's first 10 actions (the executed part) and the first 10 actions of the parent's NFE-10 endpoint from the same ε.
- **L\_grip:** computed in **command space**, on the gripper dimension of the first 10 actions.
  - Un-normalize both actions with the dataset statistics.
  - The target is the sign of the parent's gripper command, with the same threshold the deployed policy uses (0 for LIBERO's {−1, +1} command).
  - Loss = softplus(−κ · sign(gᵀ) · ĝ), with κ = 5.
- **L\_anchor:** the parent's action flow-matching loss on the full-depth path (K = 4, cold start, random τ, real data; step length d = 0 when step-size conditioning is on).
  - The video loss drops out, because the video expert is frozen and video tokens never read action tokens.
  - It keeps \[4\] × 10 close to the parent, not identical.
  - Not used with scope P-a.

### 3.4 Parameter scopes (axis P)

| ID | Trained | Peak LR | Weight decay |
| --- | --- | --- | --- |
| P-a | Transport only (< 1M) | 1e-3 | 0 |
| **P-b** (default) | Transport + LoRA r = 16, α = 16 on q, k, v, o and FFN of the 6 action core blocks (+ step-size embedding in phase 2) | Transport and embedding 1e-3, LoRA 2e-4 | 0 |
| P-c | Transport + the whole action expert (\~105M) | Transport 1e-3, expert 3e-5 | 0.01 on the expert |

### 3.5 Tests that must pass before round 1

- &#91;4\] × 10 with T0 and LoRA = 0 gives **bit-exact** parent actions for the same seed.
- The cached text K/V gives the same output as recomputing it.
- Action-call counts measured by hooks match the table in 2.4.
- With k loops, iteration r reads video slot max(1, K\_v − k + r).
- p\_j + r\_j = h\_j^(k\_j) exactly.
- At init: T4/T5 equal T3 (B = 0); T6 equals T1.
- A 4-loop step always starts cold, even with transport on.
- **Teacher stop-gradient:** with the student's z\_j requiring grad, the target has no `grad_fn`, and no gradient reaches z\_j through the teacher.
- **Truncation:** no gradient reaches any tensor older than the boundary, including r\_{j−1}.
- **Targets:** at NFE 10, ū\_j equals one parent velocity call; at d = 0.5, Φᵀ equals 5 parent Euler steps.
- **Gripper conversion:** on 1,000 dataset actions, the command-space sign matches the deployed policy's gripper command.
- **Step-size embedding** (phase 2): zero-init leaves outputs bit-exact with the parent.
- LoRA: merging into the weights gives the same output; LoRA = 0 is bit-exact with the parent.
- **Layouts:** 2-GPU and 4-GPU first-update loss and gradient norm match within tolerance.
- **Resume:** resuming from a checkpoint reproduces the next 10 updates.
- CUDA graph per schedule matches eager over 50 queries.

## 4. Runs

Phase 1 answers R1 and R2 at NFE 10 with four core runs. Phase 2 (three core runs) starts only if R2 passes. Each trained run differs from its comparison partner in exactly one thing.

### 4.1 Eval-only baselines (no training, run on the eval server now)

| ID | Definition | Schedules to evaluate |
| --- | --- | --- |
| RT-B0 | Parent 4/4 | \[4\] × 10 |
| RT-B0-20 | Parent at NFE 20, for information: does sampling more finely help in closed loop? | \[4\] × 20 |
| RT-B1 | Parent with fewer denoising steps | \[4\] × 5, \[4\] × 4, \[4, 4\], \[4\] |
| RT-B3 | Parent, depth cut with cold start (T0) | Σ1, Σ2 |
| RT-B5 | Parent with identity transport (T1), zero-shot | Σ1, Σ2 |
| RT-B6 | Parent with forecast transport (T2), zero-shot | Σ1, Σ2 |

### 4.2 Phase 1 runs (NFE 10, schedule mixture {Σ1, Σ2})

| Run | Definition | Compared with | Answers |
| --- | --- | --- | --- |
| **RT-A** | T4 · R-FU · P-b | The RT-B1 line at equal latency | R1 at NFE 10: does RT beat simply taking fewer steps? |
| RT-B4 | RT-A with transport **off** (T0) | RT-A at the same schedule | R2: is the gain from transport or from fine-tuning for fewer loops? |
| RT-T1ft | RT-A with T1 fixed; only LoRA is trained | RT-A | Is a learned transport needed, or is identity warm start + fine-tuning enough? |
| RT-Pa | RT-A with scope P-a: transport only, backbone frozen, no anchor | RT-A and RT-B5 | Can a module under 1M parameters make warm start work on its own? |
| RT-TF (optional) | RT-A with regime R-TF | RT-A | Is the on-policy unroll needed? |

At NFE 10 every step has length 0.1, so a transport that sees τ\_{j+1} learns nothing that RT-B4 lacks. R2 is a clean comparison here.

### 4.3 Phase 2 runs (only if Gate 1 passes; step-size conditioning on in every run)

| Run | Definition | Compared with | Answers |
| --- | --- | --- | --- |
| **RT-A2** | RT-A + step-size conditioning, mixture {Σ1, Σ2, Σ5, Σ6, Σ7, Σ8} | The RT-B1 line and RT-B2a | R1 at low NFE |
| RT-B2a | Step distillation to NFE 2 at full depth: fixed schedule \[4, 4\], transport off, R-FU, scope P-c, anchor at d = 0 | RT-A2 at Σ6/Σ7 | The strongest few-step competitor |
| RT+B2 | RT recipe initialized from the final RT-B2a checkpoint, schedules {\[4, 1\], \[4, 2\]} | RT-B2a | R3: do cutting depth and cutting steps stack? |
| RT-B4-2 (optional) | RT-A2 with transport off | RT-A2 | R2 at low NFE |

The teacher is always the parent on its NFE-10 grid, including for RT+B2. RT-A2 must also be evaluated at Σ1 and Σ2 to check that step-size conditioning and the wider mixture did not hurt NFE 10.

**Deferred until both phases are read:**

- Transport forms T3, T5 and T6.
- The R-FM regime (it needs its own target and endpoint definitions).
- Loss-term ablations.
- Scope P-c for RT.
- An NFE-1 student.
- Seeds 43/44.

## 5. Run plan on 4 × H100

### 5.1 GPU layouts

Both layouts train with the same global batch, data order, learning rate and update count, so their results are interchangeable. Only the action expert, transport and LoRA get gradients; the video expert runs prefill without gradient.

| Layout | Runs in parallel | GPUs per run | Windows per GPU per update |
| --- | --- | --- | --- |
| **2 × 2** (recommended) | 2 | 2 | 64 |
| 1 × 4 | 1 | 4 | 32 |

- DDP with the shared i.i.d. sampler seed.
- If a rank's windows do not fit with the teacher and student unrolls, use micro-batches with gradient accumulation. Never change the global batch, learning rate or update count.
- Confirm the layout with the first-20-update measurements from 3.1.

### 5.2 Rounds (2 × 2 layout)

| Round | GPUs 0–1 | GPUs 2–3 | Starts when | Answers |
| --- | --- | --- | --- | --- |
| 0 | — | — | Now: eval-only baselines (4.1) on the eval server | Gate 0 |
| 1 | RT-A | RT-B4 | Tests in 3.5 pass | R2; R1 at NFE 10 |
| 2 | RT-T1ft | RT-Pa | Round 1 training finished **and** Gate 0 passed | Learned vs identity transport; tiny module |
| 3 | RT-TF (optional) | free (give to ChronoLoop) | Round 2 training finished | On-policy vs teacher-forced |
| 4 (phase 2) | RT-A2 | RT-B2a | **Gate 1 passed** (round 1 final evals) | R1 at low NFE |
| 5 (phase 2) | RT+B2 | RT-B4-2 (optional) | RT-B2a training finished (no eval wait) | R3; R2 at low NFE |

**1 × 4 layout:**

1. RT-A → RT-B4.
2. Gate 0 → RT-T1ft → RT-Pa → RT-TF.
3. Gate 1 → RT-A2 → RT-B2a → RT+B2 → RT-B4-2.

### 5.3 Gates

- **Gate 0** (eval-only baselines, Long, fixed noise): if parent NFE 4 is within 1 point of NFE 10, fewer steps already matches at 120 action calls, and Σ2 (138) cannot win R1. Run round 1 anyway (it still answers R2), then stop.
- **Gate 1** (round 1 final evals, Long, fixed noise, at Σ1 and Σ2):
  - RT-A − RT-B4 ≥ +1 at Σ1 or Σ2 → phase 2 may start.
  - Otherwise transport adds nothing beyond fine-tuning: stop RT after round 2 and report depth-cut distillation as the result.
- **Early read:** evaluate each run's update-2,000 checkpoint on Long at its main schedules for a quick look. Never use it for a decision.

### 5.4 Launch flags

Common to every run:

```bash
torchrun --nproc_per_node 2 scripts/train_transport.py \
  --teacher <full-suite v0 4/4 checkpoint> --teacher-grid 0.1 \
  --data libero_all --freeze-video --global-batch 128 --updates 6000 \
  --save-at 2000,6000 --run-name <Run> <run flags below>
```

Use `CUDA_VISIBLE_DEVICES=0,1` and `2,3` for the two runs of a round, or `--nproc_per_node 4` for the 1 × 4 layout. mix6 = {Σ1, Σ2, Σ5, Σ6, Σ7, Σ8}.

| Run | --init | --transport | --regime | --scope | --schedules | --step-cond | --anchor |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RT-A | parent | T4 | fu | pb | Σ1, Σ2 | off | on |
| RT-B4 | parent | T0 | fu | pb | Σ1, Σ2 | off | on |
| RT-T1ft | parent | T1 | fu | pb | Σ1, Σ2 | off | on |
| RT-Pa | parent | T4 | fu | pa | Σ1, Σ2 | off | off |
| RT-TF | parent | T4 | tf | pb | Σ1, Σ2 | off | on |
| RT-A2 | parent | T4 | fu | pb | mix6 | on | on (d = 0) |
| RT-B2a | parent | T0 | fu | pc | \[4, 4\] | on | on (d = 0) |
| RT+B2 | RT-B2a final | T4 | fu | pb | \[4, 1\], \[4, 2\] | on | on (d = 0) |
| RT-B4-2 | parent | T0 | fu | pb | mix6 | on | on (d = 0) |

Flag names are a proposal for the agent; what matters is that each run differs from its partner only in the listed column.

## 6. Results to fill

### 6.1 Eval protocol

- **Success rate:** matched harness, 4 suites × 50 official states × eval seeds 42/43/44. **Fixed noise** is primary, **fresh noise** secondary. Run Long first.
- **Comparisons:** paired by (task, state, seed), cluster bootstrap over (task, state), 95% CI.
- **Latency:** batch 1, one CUDA graph per schedule, raw observation to action on the host, 500 replayed observations, p50. Same GPU type for every row, with the text K/V cache fix applied to all.
- **Equal latency** means p50 within ±5%. Otherwise compare against the RT-B1 line interpolated at RT's latency.

### 6.2 Phase 1 (NFE 10)

| Run | Schedule | Action calls | p50 (ms) | Long @ 2k (early) | Long fixed | Long fresh | 4-suite avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RT-B0 | \[4\] × 10 | 300 | 126 (re-measure) | — | 87.87 |  | 94.92 |
| RT-B0-20 | \[4\] × 20 | 600 |  | — |  |  |  |
| RT-B1 | \[4\] × 5 | 150 |  | — |  |  |  |
| RT-B1 | \[4\] × 4 | 120 |  | — |  |  |  |
| RT-B1 | \[4, 4\] | 60 |  | — |  |  |  |
| RT-B1 | \[4\] | 30 |  | — |  |  |  |
| RT-B3 | Σ1 | 192 |  | — |  |  |  |
| RT-B3 | Σ2 | 138 |  | — |  |  |  |
| RT-B5 | Σ1 | 192 |  | — |  |  |  |
| RT-B5 | Σ2 | 138 |  | — |  |  |  |
| RT-B6 | Σ1 | 192 |  | — |  |  |  |
| RT-B6 | Σ2 | 138 |  | — |  |  |  |
| RT-B4 | Σ1 | 192 |  |  |  |  |  |
| RT-B4 | Σ2 | 138 |  |  |  |  |  |
| **RT-A** | Σ1 | 192 |  |  |  |  |  |
| **RT-A** | Σ3 (eval only) | 168 |  | — |  |  |  |
| **RT-A** | Σ4 (eval only) | 156 |  | — |  |  |  |
| **RT-A** | Σ2 | 138 |  |  |  |  |  |
| RT-T1ft | Σ1 | 192 |  |  |  |  |  |
| RT-T1ft | Σ2 | 138 |  |  |  |  |  |
| RT-Pa | Σ1 | 192 |  |  |  |  |  |
| RT-Pa | Σ2 | 138 |  |  |  |  |  |
| RT-TF | Σ1 | 192 |  |  |  |  |  |
| RT-TF | Σ2 | 138 |  |  |  |  |  |

### 6.3 Phase 2 (step-size conditioning)

| Run | Schedule | Action calls | p50 (ms) | Long @ 2k (early) | Long fixed | Long fresh | 4-suite avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **RT-A2** | Σ1 | 192 |  |  |  |  |  |
| **RT-A2** | Σ2 | 138 |  |  |  |  |  |
| **RT-A2** | Σ5 | 102 |  |  |  |  |  |
| **RT-A2** | Σ6 | 78 |  |  |  |  |  |
| **RT-A2** | Σ7 | 48 |  |  |  |  |  |
| **RT-A2** | Σ8 | 42 |  |  |  |  |  |
| RT-B2a | \[4, 4\] | 60 |  |  |  |  |  |
| RT+B2 | \[4, 2\] | 48 |  |  |  |  |  |
| RT+B2 | \[4, 1\] | 42 |  |  |  |  |  |
| RT-B4-2 | Σ5 | 102 |  |  |  |  |  |
| RT-B4-2 | Σ6 | 78 |  |  |  |  |  |
| RT-B4-2 | Σ7 | 48 |  |  |  |  |  |
| RT-B4-2 | Σ8 | 42 |  |  |  |  |  |

### 6.4 Questions (Long, fixed noise)

| Phase | Question | Comparison | Δ (95% CI) | Pass rule | Pass? |
| --- | --- | --- | --- | --- | --- |
| 1 | R1: beats fewer steps | RT-A at Σ2 vs the RT-B1 line at equal latency |  | ≥ +2 and CI lower bound > 0, or equal success at ≥ 20% lower latency |  |
| 1 | R2: gain is from transport | RT-A − RT-B4, at Σ1 and at Σ2 |  | ≥ +2 and CI lower bound > 0 at the same schedule |  |
| 1 | Learned transport matters | RT-A − RT-T1ft, at Σ1 and at Σ2 |  | ≥ +1 |  |
| 1 | Tiny module is enough | RT-Pa − RT-B5 at Σ2 |  | ≥ +2, and RT-Pa within 1 point of RT-A |  |
| 1 | On-policy unroll matters | RT-A − RT-TF at Σ2 |  | ≥ +1 |  |
| 2 | No regression at NFE 10 | RT-A2 − RT-A, at Σ1 and at Σ2 |  | Within 1 point |  |
| 2 | R1 at low NFE | RT-A2 at Σ6 or Σ7 vs RT-B2a and the RT-B1 line |  | Same rule as phase-1 R1 |  |
| 2 | R2 at low NFE | RT-A2 − RT-B4-2 at Σ6 |  | ≥ +2 and CI lower bound > 0 |  |
| 2 | R3: steps and depth stack | RT+B2 at \[4, 1\] − RT-B2a at \[4, 4\] |  | CI lower bound > −1.5, at lower latency |  |

**What the outcome means:**

- **Phase-1 R1 + R2:** this is a paper result.
- **R1 without R2:** depth-cut distillation, not transport, is the useful part. It fits better as one section of an elastic-WAM paper.
- **Phase 2:** decides whether the result extends to the few-step regime.
