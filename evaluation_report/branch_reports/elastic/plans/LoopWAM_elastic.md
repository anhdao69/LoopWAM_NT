# Elastic LoopWAM from v0: Technical and Experimental Plan

**Status:** Research design and implementation specification. The new elastic experiments have not been implemented, trained, or evaluated in this document. Existing repository behavior, proposed changes, and derived arithmetic are distinguished explicitly.

**Goal:** Starting from the existing v0 LoopWAM policy, train one checkpoint that supports nine action-depth/denoising-step budgets, then determine whether residual transport adds value after the whole action expert is allowed to adapt.

**Architecture:** Keep the four-loop video expert fixed. Fine-tune the student action expert across one-, two-, and four-loop execution and two-, five-, and ten-step sampling. Compare no transport, identity residual reuse, and learned T4 transport under matched training conditions.

**Tech stack:** Existing LoopWAM/FastWAM PyTorch code, flow matching, AdamW, BF16 computation with FP32 weights, DDP when using multiple GPUs, and the pinned LIBERO evaluation harness. No new model library or simulator is required by this design.

**Source specification:** `LoopWAM_Elastic_RT_Final_Run_Plan.md` and `LoopWAM_Elastic_RT_Run_Matrix.json`, supplied in this conversation. This document expands their five-new-run design; it does not replace it with a wider initial sweep.

**Reviewed snapshots:** v0 at `56e33285a82c6abe075565b3d60c26a5ba7a148f`; residual transport at `944a07f1e6f3d6ca8b96034370c5fc88e0cf9ac2`. These are pinned source references, not a claim that every repository branch remains unchanged.

---

## 0. Scope and terminology

“From v0” means initialize from a robot-trained v0 checkpoint, not retrain from random weights. Call the original full-LIBERO, 10-epoch, video-four/action-four checkpoint **BASE10**. BASE10 is both the student initializer and the frozen teacher for the initial campaign. Its checkpoint SHA-256 in the supplied specification is `4ffe1b14f35484ea582189ba0f06c5a41ac199e1fa733b491719727105e5e5cc`; validate the actual artifact, normalization, and VAE identity before use. [P1, P2]

There are three separate iteration axes:

| Symbol | Meaning | What changes? |
|---|---|---|
| q | Robot replanning query | The robot has executed actions and observes the environment again |
| j | Denoising/flow-sampling step within one query | The noisy action chunk z changes |
| r | Core loop inside one action-network evaluation | Hidden features h are refined by repeated shared blocks |

`K_v/K_a = 4/1` means four video core loops and one action core loop. It does not mean four denoising steps or a one-action prediction horizon. `N` denotes NFE, the number of action-network evaluations in the Euler sampler. All initial experiments predict 32 actions and execute the first 10 before replanning. [R1, R3, P1]

**Fixed in the initial sweep:** video depth, v0 partition, donor initialization, data, teacher, objective, conditioning, action horizon, and evaluation protocol. No ChronoLoop memory, new KV-fusion mechanism, new video backbone, adaptive halting, arbitrary loop schedules, or transport-rank search is introduced.

**Not promised:** monotonic success as computation increases, accuracy at untrained depths, a fixed inference speedup, or matching the simulated targets in the ideal paper.

---

## 1. The starting model: what v0 computes

### 1.1 Two experts

The policy uses a video expert and an action expert. The video expert processes observation features and, during base training, future video targets. The action expert predicts an action flow velocity conditioned on observation features, language, and current proprioception. Their weights are separate; sharing occurs across repeated depth inside each expert, not between the two experts. [R1–R3]

| Component | Video expert | Action expert |
|---|---:|---:|
| Hidden dimension | 1536 | 512 |
| FFN dimension | 6144 | 2048 |
| Mixed-attention heads | 12 | 12 |
| Head dimension | 128 | 128 |
| Physical blocks | 12 | 12 |
| Prelude/core/coda | 3/6/3 | 3/6/3 |
| Base core loops | 4 | 4 |
| Effective block applications | 30 | 30 |

The action expert projects its 512-dimensional hidden features into the common 1536-dimensional attention space. The original initialization selects Wan2.1-T2V-1.3B donor blocks with zero-based indices `[0,1,2,9,10,11,12,13,14,27,28,29]`. Do not mix the centered-donor or partition experiments into this campaign. [R2]

The declared base policy contains 584,536,135 parameters, excluding frozen VAE/text-encoder components. Parameter count is not peak GPU memory. The action expert itself contains 105,277,959 parameters. [R7]

### 1.2 Looped depth

An ordinary 30-block model stores 30 distinct block parameter sets. v0 stores:

```
Prelude: P1 P2 P3
Core:    C1 C2 C3 C4 C5 C6    reused four times
Coda:    D1 D2 D3
```

For either expert, effective depth at K loops is:

`D(K) = 3 + 6K + 3 = 6 + 6K`.

Thus K=1,2,4 gives 12,18,30 block applications, respectively. The core weights are tied; the input hidden state changes each time. Repeating weights reduces stored parameters relative to a dense model of the same effective depth, but does not remove the repeated block computation. [R1]

For an action evaluation at denoising step j:

- Encode noisy actions, then run the prelude to obtain p_j.
- Apply the shared six-block core K_a times.
- Run the coda and action output head to obtain velocity v_j.

Do not interpret this as a proven fixed-point solver. Moreover, although core weights are shared, each repetition reads a different virtual video-cache slot; the effective conditioned mapping is not necessarily identical across repetitions.

### 1.3 Observation/future/action masking

During base training, define F=current observation, U=noisy future video, and A=noisy actions:

| Query group | Reads F | Reads U | Reads A |
|---|---|---|---|
| F | Yes | No | No |
| U | Yes | Yes | No |
| A | Yes | No | Yes |

Actions never consume ground-truth future video. The observation stream does not depend on actions. Therefore the observation computation can be performed once and reused over the action denoising steps. Future-video prediction remains a base-training objective, not a requirement to generate video during control. [R3, R8]

### 1.4 Observation caching and late alignment

For K_v=4 the video expert executes 30 virtual blocks, producing separate K/V cache entries for repeated uses of the same physical block. Weight sharing does not mean these cache entries are identical.

An action core iteration r in a K_a-loop evaluation reads video core loop:

`video_loop = 4 - K_a + r`, with r starting at 1.

Consequently:

| Action depth | Video core loops read |
|---|---|
| K_a=1 | 4 |
| K_a=2 | 3,4 |
| K_a=4 | 1,2,3,4 |

Prelude and coda use their corresponding video slots. A shallow action expert must still read the late/deep visual features. Do not implement depth reduction by simply truncating the first K_a video-aligned iterations. [R1, R4]

---

## 2. Flow matching and why v0 is not already elastic

### 2.1 Base training is not a ten-step unroll

For a clean normalized action chunk a and Gaussian noise epsilon, the action flow path is:

`z_tau = (1-tau) a + tau epsilon`, with target `u* = epsilon - a`.

Base v0 training samples noise levels and predicts instantaneous flow velocities. It uses the configured four-loop depth and final-exit supervision. It does not ordinarily simulate all ten deployed Euler steps in each training example. Calling it “trained at NFE10” is imprecise: ten steps describe its reference deployment sampler. [R3]

The action part of the base objective is a timestep-weighted, padding-masked velocity error. The video part trains future-latent flow prediction, with the current frame kept clean. Both experts are trainable during base training; VAE/text preprocessing remains frozen.

### 2.2 Inference samples actions by Euler integration

Within a robot query:

1. Encode the current observations and construct the four-loop video cache.
2. Sample z_0 = epsilon, shape `[B,32,7]`.
3. Set `tau_j = 1 - j/N`.
4. Predict v_j using the action expert.
5. Update `z_{j+1} = z_j + (tau_{j+1}-tau_j) v_j`.
6. After N steps, denormalize and execute the first ten actions.

The Euler time increment is negative: `tau_{j+1}-tau_j = -1/N`. The positive step length used for conditioning is `d = 1/N`. These signs are intentionally different. [R4, R5]

### 2.3 Two cuts create two different distribution shifts

Reducing K_a changes the hidden representation presented to the coda and action head. A coda trained on four-loop states is not guaranteed to decode one-loop states well.

Reducing N makes each numerical integration step longer. A model trained to predict instantaneous velocity is not guaranteed to reproduce several small teacher steps in one large step.

The first problem is depth adaptation. The second is finite-step/trajectory adaptation. Elastic training must address both; merely changing inference flags addresses neither.

---

## 3. Motivation and the exact elasticity hypothesis

### 3.1 Spend computation where it is not repeatedly paid

A useful bookkeeping model is:

`T_query ≈ T_VAE + T_video(K_v=4) + sum_j T_action(k_j) + T_transport + T_other`.

At the v0 reference point, video executes 30 blocks once, while actions execute 30 blocks on each of ten denoising steps: 300 action-block calls.

At flat K_a=1,N=2, actions execute 12 blocks twice: 24 calls. That is 12.5 times fewer action-block calls, not a measured 12.5-times end-to-end speedup. VAE, observation encoding, attention sizes, runtime kernels, and transport still contribute cost.

The architectural hypothesis is: a deep cached visual representation may permit a shallower repeatedly evaluated action expert, provided that action expert is trained for the shallow regime. This is the hypothesis to test, not a guaranteed property of every WAM.

### 3.2 One checkpoint, nine operating points

Define:

`G_flat9 = {(K_a,N): K_a in {1,2,4}, N in {2,5,10}}`.

“Flat” means the same action-loop count is used at every denoising step of a query:

| Name | K_a | NFE | Loops per denoising step | Action calls |
|---|---:|---:|---|---:|
| flat_k1_n2 | 1 | 2 | [1,1] | 24 |
| flat_k1_n5 | 1 | 5 | [1,1,1,1,1] | 60 |
| flat_k1_n10 | 1 | 10 | [1]×10 | 120 |
| flat_k2_n2 | 2 | 2 | [2,2] | 36 |
| flat_k2_n5 | 2 | 5 | [2]×5 | 90 |
| flat_k2_n10 | 2 | 10 | [2]×10 | 180 |
| flat_k4_n2 | 4 | 2 | [4,4] | 60 |
| flat_k4_n5 | 4 | 5 | [4]×5 | 150 |
| flat_k4_n10 | 4 | 10 | [4]×10 | 300 |

Every row uses four video loops. Counts follow `sum_j(6+6k_j)`. Equal counts do not guarantee equal FLOPs or latency; for example [4,4] and [1]×5 distribute their work differently.

One training run produces one set of action weights supporting this finite grid. It is not nine independently optimized networks. It is also not an automatic halting policy: the deployment system chooses a supported setting at a query boundary. Arbitrary mixtures of depths inside a query and depths outside the grid are not established capabilities.

### 3.3 Why full-action fine-tuning is primary

The main comparison uses P-c: train the whole student action expert, including prelude, shared core, coda, action input/output projections, and its conditioning layers. Keep the teacher, video expert, VAE, text encoder, and shared proprioception projection frozen.

P-b instead trains rank-16 LoRA only in the six core blocks. That remains a valid adapter-constrained question, but should not determine whether transport helps a fully adaptable elastic model. The primary P-c design avoids unnecessarily restricting the no-transport baseline. It does not assert that a frozen coda has already been proven to cause failure. [R4, R5, P1]

---

## 4. Elastic training without transport: the E0 method

### 4.1 Student and teacher construction

Load BASE10. Freeze its video expert and action teacher. Deep-copy its action expert into the student and make the complete student action expert trainable. Add a common step-size conditioning module. Keep initial action weights identical across matched methods.

The teacher remains the original reference policy throughout fine-tuning; it is not EMA-updated and is not the student at its current full depth.

This is self-distillation in the broad sense of distilling the robot policy into an elastic version of itself. Although base v0 originated from video-pretrained Wan weights without an external robot-policy teacher, the elastic stage does use a robot-trained teacher.

### 4.2 Add step length to the time modulation

The original action expert receives native timestep `t = 1000 tau`. Preserve this convention. Add:

`m_student(tau,d) = m_base(1000 tau) + StepCondition(d)`.

The existing StepCondition uses a 32-dimensional sinusoidal embedding, Linear(32,128), SiLU, and Linear(128,6×512), reshaped to six modulation channels. Its final layer is zero-initialized. [R4, R5]

At initialization the module contributes zero, but this does not make shallow paths equivalent to the four-loop parent. It only preserves the original modulation before learning.

Use d=0 as the instantaneous-flow anchor convention. Use d=0.5,0.2,0.1 for deployed NFE2,5,10 respectively. After training, StepCondition(0) need not be zero: zero initialization is not an algebraic constraint maintained by optimization.

No new loop-index or loop-budget embedding is added in this initial design. K_a selects the number of shared-core applications and late-aligned video slots. Adding explicit loop conditioning would be a separate architectural experiment.

### 4.3 Why the target depends on d

An instantaneous ODE velocity depends on state and time, not on the numerical solver step length. However, a distilled student is asked to predict an average velocity that reproduces a finite segment of the teacher trajectory. That finite-interval target depends on d.

At tau=1, predicting the teacher's displacement to tau=0.9 is a different task from predicting its displacement to tau=0.5. The additional conditioning disambiguates these targets.

### 4.4 One sampled configuration per optimizer update

For E0/E1/E2, generate randomized blocks containing all nine configurations exactly once. One configuration is held fixed across the entire global batch and every gradient-accumulation microbatch. Every DDP rank uses that same configuration. Reshuffle after nine optimizer updates.

All configurations update the same student weights. They are not all evaluated as student rollouts on every update. There is an additional full-depth anchor forward on each training example.

Over 21,700 updates, each configuration receives 2,411 update assignments and one receives one extra. Log actual valid windows as well, since the final partial batch of each epoch has 81 real windows rather than 128. Equal update assignments are not equal GPU time and are only approximately equal real-window exposure.

Use the same schedule sequence, data order, and per-window noise for E0/E1/E2. Keep RNG streams for schedule sampling, data order, noise, and new-module initialization separate. [P1]

### 4.5 Local targets start at the student's own state

At step j, the student is at z_j and will move from tau_j to tau_{j+1}. Define `Phi_T(z_j; tau_j -> tau_{j+1})` as the frozen teacher's deployed Euler map over that interval on its fixed 0.1 grid.

The target is:

`u_bar_j = stopgrad((Phi_T(stopgrad(z_j)) - stopgrad(z_j)) / (tau_{j+1}-tau_j))`.

The teacher takes:

| Student NFE | Positive d | Teacher microsteps per student step |
|---|---:|---:|
| 10 | 0.1 | 1 |
| 5 | 0.2 | 2 |
| 2 | 0.5 | 5 |

These N values divide ten, so all macrostep boundaries lie on the reference grid. At NFE10 the implementation returns one teacher velocity directly rather than subtracting nearby states. [R4]

The teacher map is the parent policy's numerical behavior, not exact physical truth or an exact continuous-time ODE solution. Do not change to a finer NFE20 teacher during this experiment.

Student states differ across methods after training diverges. Therefore local targets must be recomputed at each method's own states. Sharing initial noise and teacher weights is fair; forcing all methods to use the same precomputed local states is a different, teacher-forced training regime.

### 4.6 A separate teacher endpoint

From the same initial noise, run the teacher's complete ten-step trajectory once to obtain a reference final chunk z_T.

This target is distinct from local correction targets. Local targets ask how to move from the student's possibly imperfect current state. The endpoint target asks whether the final executed actions remain close to the original teacher output from the shared initial noise.

In the existing implementation, local targets require ten teacher action evaluations in total, regardless of student N, and the endpoint reference requires ten more. These are teacher forward evaluations, not backward passes. The student also performs its selected rollout and a full-depth anchor forward. Distillation is therefore not as cheap as the student deployment path. [R5]

---

## 5. Exact losses and gradient boundaries

Use the existing objective:

`L = L_local + 0.5 L_end + 0.1 L_grip + 0.5 L_anchor`.

### 5.1 Local loss

Average the padding-masked action-velocity MSE over the N student steps:

`L_local = (1/N) sum_j masked_MSE(v_student_j, u_bar_j)`.

The 1/N factor prevents a ten-step configuration receiving five times the local-loss weight merely because it has more sampled steps than a two-step configuration. It does not equalize gradient magnitudes or training compute.

### 5.2 Endpoint loss

Match the first ten normalized action vectors of the student and teacher endpoints:

`L_end = mean((z_student[:10] - z_teacher[:10])^2)`.

These are the actions the policy executes before replanning. This loss compares teacher predictions, not future ground-truth video.

### 5.3 Gripper loss

Use the repository's exact normalization and deployed command conversion. In the reviewed code:

`raw = (a_norm[...,6] - offset[6]) / scale[6]`

`g(a_norm) = 1 - 2 raw`.

Positive command closes and negative opens under this convention. The target sign is +1 when teacher command is nonnegative and -1 otherwise. The loss is:

`L_grip = mean softplus(-5 * teacher_sign * student_command)`

over the first ten actions. Do not infer gripper semantics from the sign of the normalized action value. [R4]

### 5.4 Flow anchor

Independently sample real clean actions, anchor noise, and an anchor time. Predict instantaneous velocity with a cold-start four-loop student and d=0:

`L_anchor = w(tau) * masked_MSE(v_student_k4(z_tau,tau,d=0), epsilon_anchor-a)`.

The target is the original data flow-matching target, not the teacher's predicted velocity. The anchor encourages retention but does not guarantee exact parent behavior. Because the video expert is frozen, the elastic stage does not add a trainable video prediction objective. [R5]

### 5.5 Truncated trajectory gradients

Preserve the existing final-three-step truncation. Numbering student steps j=0,...,N-1, detach z and every carried history tensor at the start of a step when `j <= N-3`.

For N=10, the endpoint graph spans steps 7,8,9; for N=5 it spans 2,3,4; for N=2 both steps remain connected. Earlier steps still receive their own local-loss gradients. This is truncation across denoising time, not truncation across shared core depth. All executed core loops within a forward participate in backpropagation. [R4, R5]

### 5.6 Conceptual training iteration

```
choose one supported schedule for the optimizer update
load a global batch, common action noise, and anchor noise/time
prefill frozen video cache once per example
compute teacher endpoint from the common initial noise, without gradients
build student text/proprio conditioning and K/V with gradients
z = initial noise; history = None
for each student denoising step:
    detach z and history at the prescribed truncation boundary
    compute detached teacher macrostep target starting at z
    compute student velocity with the selected core depth and positive d
    accumulate masked local loss
    update z with the negative Euler increment
    save the pre-coda core residual when the method uses transport
compute endpoint and command-space gripper losses
compute the cold full-depth d=0 data-flow anchor
backpropagate the weighted loss, clip gradients, and update student parameters
```

This is algorithmic pseudocode, not a drop-in replacement for distributed batching and checkpoint logic.

---

## 6. Residual transport: the optional second idea

### 6.1 Separate the questions

Elastic training teaches the action network to work at different depths and step lengths. Transport asks whether computation already performed at the preceding denoising step is useful at the next one.

Transport memory is local to one action-sampling query. It does not carry robot episode history and is not ChronoLoop memory.

### 6.2 What is transported?

Let p_j be the action hidden state after the prelude and h_j the state after the last core loop, before the coda. Save:

`r_j = h_j - p_j`.

At the next step, recompute p_{j+1} from the new noisy actions. If transport is enabled, initialize the core with:

`h_{j+1}^{(0)} = p_{j+1} + T(r_j; tau_j,tau_{j+1})`.

Run the requested number of correcting core loops, then coda/head. Saving h-p rather than h retains a direct route for the new action input through the fresh prelude representation.

**Exact semantic detail:** when the previous step was itself warm-started, h_j-p_j includes the carried contribution and subsequent processing. Do not replace it with h_j-h_j^(0), which would store only the latest increment and define a different method. [R4]

### 6.3 E0: no transport T0

Every denoising step starts from its fresh p_j. Train P-c and the step-condition module over all nine flat settings. This is the main baseline, not an intentionally capacity-restricted model.

### 6.4 E1: identity transport T1

For steps after the first and with fewer than four core loops, add the previous residual unchanged:

`T(r)=r`.

No learned transport parameters are introduced relative to E0, but residual computation/storage/addition still has cost. E1 still has the common learned step-condition module.

At [1,1], the first one-loop evaluation starts cold. Its residual warm-starts only the second evaluation. There is no hidden four-loop first step.

### 6.5 E2: learned T4 transport

With time features f_tau formed from the previous and current normalized denoising times:

`g = sigmoid(MLP_gate(f_tau))`

`r_corrected = r + B((A r) * (1 + MLP_film(f_tau)))`

`T4(r) = g * r_corrected`.

A maps 512→32 and B maps 32→512. The gate has 512 channels and is broadcast over action tokens. B starts at zero and g starts near 0.95. Thus initial T4 is approximately 0.95r, not exactly identity and not zero transport. All common parameters can still have identical initial tensors across methods. [R4]

In `action_core`, the argument named `next_tau` means the destination time of the transport from the previous residual—i.e. the time of the current action evaluation. Do not accidentally pass the end time of the upcoming Euler step. The current step length is provided separately to StepCondition. [R4, R5]

### 6.6 Bypass rules

Transport is bypassed on the first denoising step and whenever the current action evaluation uses four loops. Every robot query resets the transport history. Full-depth settings therefore probe retention and indirect training effects, not active transport.

Full-depth E0/E1/E2 use the same operator structure; trained outputs need not be identical. A valid bypass test compares one fixed checkpoint with transport toggled, not different trained checkpoints.

### 6.7 Added parameter accounting

Derived from the reviewed module shapes:

| Module | Additional parameters |
|---|---:|
| Common step-condition module | 400,512 |
| T1 transport | 0 |
| T4 transport | 76,448 |

Thus the nominal deployed policy totals are 584,936,647 for E0/E1 and 585,013,095 for E2, excluding VAE/text components. These are arithmetic derived from the declared base count and source shapes; verify them with a deduplicated runtime count. Do not count a retained training teacher or duplicate wrapper modules as required deployed parameters. [R4, R7]

---

## 7. First-full schedules and why they are separate

A first-full schedule starts with four action loops, then uses k loops for the remaining N-1 steps:

`front(k,N) = [4] + [k]×(N-1)`.

Its action calls are:

`C_front(k,N) = 30 + (N-1)(6+6k)`.

Relative to flat, the extra first-step cost is `6(4-k)` calls:18 for k=1 and12 for k=2, independent of N.

| Historical ID | Schedule | NFE | Calls |
|---|---|---:|---:|
| S1 | [4]+[2]×9 | 10 | 192 |
| S2 | [4]+[1]×9 | 10 | 138 |
| S5 | [4,2,2,2,2] | 5 | 102 |
| S6 | [4,1,1,1,1] | 5 | 78 |
| S7 | [4,2] | 2 | 48 |
| S8 | [4,1] | 2 | 42 |

A better result for [4,1] than [1,1] cannot, by itself, establish a transport gain: the former pays for an extra 18 action-block calls. Compare T0/T1/T4 under the same schedule to isolate transport, and compare measured accuracy–latency frontiers to decide deployment.

Historical RT-A2 uses P-b and these six first-full schedules. It also has full-depth first steps and a d=0 full-depth anchor; it was not trained on complete full-depth distilled rollouts. Evaluating its three full-depth deployment schedules remains necessary. [P1]

---

## 8. Every initial run and its purpose

All five new runs initialize from BASE10, use BASE10 as the fixed teacher, and use the shared contract in Section9.

| ID | Type | Transport | Scope | Training rollout distribution | Evaluation |
|---|---|---|---|---|---|
| E0 | New training | T0 | P-c | Balanced flat9 | flat9 |
| E1 | New training | T1 | P-c | Same flat9 sequence | flat9 |
| E2 | New training | T4 | P-c | Same flat9 sequence | flat9 |
| SP11 | New training | T0 | P-c | Fixed [1,1] | [1,1] |
| F0-PB | New training | T0 | P-b | Exact historical front6 stream | front6 + full3 |
| F2-PB | Existing RT-A2 | T4 | P-b | Already trained | front6 + full3 |
| Z10 | No training | T0 | Frozen parent | None | flat9 |

### E0 — Does elastic adaptation work without transport?

This checkpoint is the reference for every claimed transport gain. Its most important point is flat_k1_n2. Compare it to Z10 at the same settings to measure the benefit of adaptation. Because this involves additional training, it is not a pure architecture-only comparison.

### E1 — Is copying the residual enough?

E1 differs from E0 in residual reuse. It establishes whether a parameter-free warm start adds useful information after full-action elastic training.

### E2 — Is learned transport worth more than identity?

Compare E2 to E0 for the effect of the full learned-transport package and E2 to E1 for its incremental value over identity. E2-versus-E1 does not isolate low-rank correction from gating; a later T3 ablation would be needed for that narrower claim.

### SP11 — What does elasticity cost at the fastest point?

Train the same P-c T0 student only on [1,1] for the same total 21,700 updates. Keep step conditioning and the same full-depth data-flow anchor. It is a rollout specialist, not a training process that never evaluates full depth.

Compare E0 and SP11 at [1,1]. The specialist receives about nine times as many rollout updates at that setting. Therefore this measures the practical equal-total-budget cost of multi-budget training, not a pure weight-sharing penalty separated from data/configuration exposure. Do not use SP11 to draw that narrower conclusion.

### F0-PB — Complete the historical no-transport control

Reproduce RT-A2's P-b recipe with transport off. Do not use the new balanced schedule sampler here. Match the old source, data/noise streams, optimizer, schedule ordering, export semantics, and common initial tensors.

The historical schedule selector is `random.Random(seed + update*65537).choice(names)`, where names are ordered `S1,S2,S5,S6,S7,S8`. Reconstruct the actual historical step-condition and LoRA initialization, accounting for constructor RNG order. Use the original initialization procedure, not trained step-conditioning weights from RT-A2. [P1]

### F2-PB — Reuse RT-A2 conditionally

The supplied specification pins epoch10 checkpoint SHA-256 `25ad8ca4705e41c805464f9d7fe99e4a3c4264f192a9f205612078b75a208c3d`. Re-evaluate it through the same validated deployment path and environment as F0-PB.

If strict common-initialization/protocol reconstruction fails, keep it as a historical reference or perform a separately labeled matched retraining F2-PB-R. That contingency adds one training run; do not conceal it inside the five-run count.

### Z10 — Unmodified parent under budget overrides

No distillation, no learned new step condition, no transport. Evaluate supported explicit depth/NFE overrides through a validated evaluator. Preserve the parent metadata: an evaluation override is not evidence that the parent was trained at that depth. Native checkpoint factories may require an explicit override route rather than silently accepting asymmetric settings.

### Optional existing specialist reference

RT-B2a at [4,4], NFE2 is an existing full-action step-distillation specialist. It adds one evaluation condition, not a new main training run, and does not replace SP11 or a newly matched step-only elastic baseline.

---

## 9. Exact shared training contract

| Setting | Value |
|---|---|
| Data | All four LIBERO suites;1,712 demonstrations |
| Windows/epoch | 277,713 |
| Epochs | 10 |
| Global batch | 128 real-window capacity/update |
| Updates | 21,700 |
| Real presentations | 2,777,130 |
| Last batch/epoch | 81 real windows; weight by real count |
| Discovery seed | 42 |
| Optimizer | AdamW, betas(0.9,0.95), eps1e-8 |
| P-c action LR/decay | 3e-5 /0.01 |
| P-b core LoRA | rank16,alpha16; LR2e-4,decay0 |
| Step/T4 LR/decay | 1e-3 /0 |
| Warmup | 3% of planned updates; ceil gives 651 |
| LR decay | Preserve the existing cosine implementation |
| Gradient clipping | Global norm1 |
| Precision | FP32 weights/optimizer; BF16 compute |
| Checkpoint selection | Final raw epoch10 only |
| Diagnostic checkpoints | update2000, epochs8/9; not best-test selection |

Keep the common module initialization in a hashed artifact and verify tensor equality before training. T4 must use an independent initialization RNG. Same seed is not sufficient if constructors consume different random numbers.

Do not load BASE10 optimizer momentum into the new fine-tuning runs. Each begins with the same fresh optimizer recipe. Recovery checkpoints must retain optimizer state, schedule/data cursors, all RNG states, update number, and parameter-scope metadata.

The current RT recipe retains EMA state, but the supplied plan selects raw final weights to match the existing exported results. Do not choose between raw and EMA using test success.

These values are retained from the final specification, not an assertion that they are optimal for elasticity. [P1, P2]

---

## 10. Implementation work packages and acceptance tests

The following are planned changes. No new functions or commands below are represented as already existing in the repository.

### Task A: Schedule/experiment registration and sampling

**Modify:** `src/fastwam/models/wan22/residual_transport.py`, `scripts/transport_experiments.py`, `scripts/train_transport.py`.

**Create:** `scripts/elastic_schedules.py`, `tests/test_elastic_schedules.py`.

**Interface to implement:** `build_schedule(kind: str, action_loops: int, nfe: int) -> tuple[int,...]`; `make_balanced_sequence(seed: int, updates: int) -> list[str]`.

- [ ] Write tests for all nine flat schedules, six first-full schedules, exact counts24/36/60/120/180/300 and42/48/78/102/138/192, and invalid inputs.
- [ ] Verify each balanced block contains every configuration once and the 21,700-update sequence has frequencies2,411 or2,412.
- [ ] Implement explicit new IDs without altering legacy schedule names, ordering, or scientific-gate evidence.
- [ ] Persist checksum/cursor and test interrupted-resume schedule equality.
- [ ] Run the new tests and existing schedule tests, then commit the independently working change.

### Task B: Common initialization and scope contract

**Modify:** `transport_policy.py`, experiment setup in `train_transport.py`.

**Create:** `tests/test_elastic_initialization.py`.

- [ ] Write tests that P-c enables prelude, core, coda, input/output, and student conditioning, while teacher/video/proprio remain frozen.
- [ ] Copy common initial action and step-condition tensors into E0/E1/E2/SP11; verify equality and distinct teacher/student storage for trainable action tensors.
- [ ] Test that changing transport kind does not change common initialization.
- [ ] Audit deduplicated parameter counts and explicit optimizer groups.
- [ ] Add a separately versioned legacy reconstruction path for F0-PB and record whether strict RT-A2 matching passes.

### Task C: Loss and temporal semantics

**Modify only as required:** `transport_policy.py`, `residual_transport.py`.

**Create:** `tests/test_elastic_distillation.py`.

- [ ] Write linear/toy-field tests that NFE2 targets reproduce five teacher microsteps and NFE5 targets reproduce two.
- [ ] Assert teacher targets have no gradient path into student z.
- [ ] Verify signed Euler deltas, native tau×1000 units, positive d, and anchor d=0.
- [ ] Verify residual=h_after_core-p_after_prelude, including a step that was warm-started.
- [ ] Verify late alignment for K_a1/2/4 and no future-information leakage.
- [ ] Verify truncation matches the final-three-step rule and local losses remain trainable earlier.
- [ ] Keep existing local/anchor padding behavior and first-ten endpoint/gripper semantics unchanged; any change must be versioned and applied to matched arms.

### Task D: Dynamic DDP and optimizer behavior

**Modify:** DDP construction in `train_transport.py`.

**Create:** `tests/check_elastic_distributed.py`.

T4 is unused in full-depth updates and used in shallow updates. Start validation with `find_unused_parameters=True`, `static_graph=False`, non-reentrant activation checkpointing, and `zero_grad(set_to_none=True)`. Official PyTorch documentation explains these contracts; validate against the pinned training environment rather than upgrading dependencies during the sweep. [W1, W2]

- [ ] Compare one-rank and two-rank finite losses/gradients for the same global data and schedule.
- [ ] Test a shallow→full→shallow sequence across multiple updates, with accumulation and resume.
- [ ] Check every trainable tensor participates when intended; zero gradients at a zero-initialized projection are not the same as an unused parameter.
- [ ] Check inactive transport parameters and optimizer state follow the intended skip semantics. Do not add a dummy zero-valued loss merely to satisfy DDP: zero gradients and absent gradients can produce different optimizer behavior.
- [ ] Preserve real-sample normalization, including padded distributed slots and partial final batches.

### Task E: Evaluation and teacher-free deployment

**Modify:** `transport_inference.py`, evaluation contracts in `scripts/evaluate_rt_matched.py`, and latency measurement in `scripts/measure_rt_latency.py` as needed.

**Create:** `tests/test_elastic_inference.py`.

- [ ] Load explicit trained schedule metadata, current step conditioning, correct normalization, and transport weights without silent dropping.
- [ ] Validate parent Z10 overrides without a trained/new step-condition contribution.
- [ ] Test transport reset across queries and bypass at four loops within the same checkpoint.
- [ ] Compare eager and fixed-schedule CUDA-graph outputs over common real observations.
- [ ] Export a deployment path without a resident action teacher or unnecessary duplicate training modules; establish numerical parity before reporting its memory.
- [ ] Rebuild student text K/V when weights change during training; retain gradient through P-c text projections. During inference, cache only where inputs and weights are unchanged.

### Task F: Evidence, preflight, and release

**Create:** a campaign manifest, schedule manifest, preflight report, and result-schema validator; suggested filenames may be chosen within a dedicated experiment directory.

- [ ] Pin source commit, parent/checkpoint hashes, VAE/text/normalization identities, dataset order, environment, and seeds.
- [ ] Validate all nine paths with forward/backward and recovery before production.
- [ ] Benchmark both long/deep and long/shallow graphs; select a common feasible layout for E0/E1/E2/SP11.
- [ ] Preserve the existing one/two-GPU launcher contract unless extending it explicitly with tests.
- [ ] Verify result rows, checkpoint labels, suite totals, paired episode keys, and missing-run handling before releasing a report.

The implementation tasks are complete only after their tests pass. This document does not assert that they have passed.

---

## 11. Logging and diagnostics

Log local, endpoint, gripper, anchor, and total losses separately for each named configuration. Also log valid windows, update count, time/update, teacher evaluations, student block calls, peak allocated memory, gradient norms, and learning rates. Raw mean losses can have different scales across step lengths; a larger loss at NFE2 does not by itself prove under-training.

For E1/E2, log per-step residual norm, transported residual norm, prelude norm, and the ratio `||T(r)||/(||p||+epsilon)`. For T4 also log gate mean/range and correction-to-residual norm. These diagnose exploding or irrelevant transported states; they do not establish that reuse improves control.

For retention, keep the separate d=0 anchor loss and deployed d=0.1/0.2/0.5 full-depth behavior distinct. For the specialist, report exactly how many updates and valid windows used [1,1].

All model selection uses the predeclared final checkpoint. Debugging catastrophic numerical failures is allowed, but repairs must be versioned; do not selectively tune a struggling arm based on repeated official-test queries while presenting the final comparison as preregistered.

---

## 12. Evaluation protocol and order

### 12.1 Full condition

One condition is one checkpoint, one schedule, and one evaluation protocol. It contains:

`4 suites ×10 tasks ×50 official states ×3 evaluation seeds =6,000 episodes`.

Preserve the supplied protocol: evaluation seeds 42/43/44, fixed-noise primary setting,30 settling actions,400 policy-step cap for Spatial/Object/Goal and700 for Long,32 predicted and10 executed actions, canonical preprocessing and command conversion. Repeated seeds for a state are not independent scenes. [P1]

Use the same GPU architecture, evaluator build, backend, environment, task/state identities, noise generation, and action normalization for decisive comparisons. Rerun missing/failed jobs; never silently score a partial grid as a full result.

### 12.2 First decisive pass:19 conditions

| Models | Schedules | Count |
|---|---|---:|
| E0,E1,E2 | [1,1], [1]×10, [4]×10 | 9 |
| SP11 | [1,1] | 1 |
| F0-PB,F2-PB | [4,1], [4]+[1]×9, [4]×10 | 6 |
| Z10 | [1,1], [1]×10, [4]×10 | 3 |
| Total | | 19 |

This is 114,000 episodes. It tests the primary fast point, the long shallow schedule, and full-quality retention. It is not a license to omit inconvenient points from the later full-grid report.

### 12.3 Complete initial matrix:55 conditions

E0/E1/E2 contribute 27 conditions. F0-PB/F2-PB contribute 18. SP11 contributes 1. Z10 contributes 9. Total 55, or 330,000 episodes, including the first 19. Adding the existing RT-B2a reference makes 56.

Reused evaluations reduce new work only if their complete contracts match. The two conditional P-c first-full runs would add 18 conditions if each receives its nine-point grid.

### 12.4 Latency and memory

Use 500 shared replayed observations, batch 1, warm execution on an otherwise idle common GPU. Measure raw observation to denormalized host actions, including VAE, video prefill, action denoising, residual work, and output transfer. Cache instruction encoding consistently. Report p50 andp95, peak inference memory, and the number of resident captured graphs.

Do not compare E0's optimized student-only loader against E2's training wrapper containing a frozen teacher. Also distinguish memory for one operating point from memory when nine separate graph captures are simultaneously resident.

No speed claim can be substituted with action-block counts or aggregate simulator evaluation throughput.

---

## 13. Statistical estimands and fixed decisions

### 13.1 Primary question

At flat_k1_n2, compare:

- E1−E0: identity reuse effect.
- E2−E0: learned transport package effect.
- E2−E1: learned versus identity transport.

Use paired episode outcomes. Bootstrap clusters `(suite,task,initial_state)`, preserving all three evaluation seeds in each cluster, with 10,000 resamples. Report overall and per-suite differences. Account for the three primary tests, for example with Holm correction; define a valid paired test before scoring rather than treating arbitrary bootstrap tail counts as automatically calibrated p-values. Grid results are secondary unless separately preregistered. [P1]

The proposed state-cluster analysis conditions on the benchmark's task set. Also report task-level variation/sensitivity; it does not automatically establish generalization to new tasks.

### 13.2 Practical targets

Retain the previously proposed engineering criteria:

| Decision | Criterion to assess |
|---|---|
| Keep transport for the fast method | Reproducible meaningful gain; target≥0.5 percentage point overall plus statistical evidence and a useful measured latency trade-off |
| Prefer identity over learned | Identity noninferior within0.5 point, without a deployment disadvantage |
| Preserve full quality | Target within0.5 point overall of the parent at [4]×10; inspect Long separately |
| Interpret SP11 | Quantify practical equal-total-budget elasticity cost |

A nonsignificant difference does not establish equivalence. Wide intervals mean the result is inconclusive. Choosing a simpler method for engineering reasons is not proof that transport never helps.

### 13.3 What each possible outcome means

- E0 strong, E1/E2 not convincingly better: main method is elastic without transport; report transport as inconclusive or unhelpful under the tested recipe.
- E1/E2 consistently improve E0: transport complements elasticity; validate across training seeds and another benchmark.
- E1 matches E2 within the predeclared margin: prefer identity warm start; do not claim learned correction is necessary.
- F2-PB improves F0-PB but E1/E2 do not improve E0: transport may compensate for a restricted first-full/adapter regime; the P-c question remains different.
- SP11 clearly outperforms E0: the fastest point may pay a multi-budget exposure/interference cost. Investigate sampling or objective trade-offs with matched controls.
- E2 wins only by damaging full-depth quality: report the actual trade-off, not a universal elastic improvement.

The historical +1.35-point S2 aggregate difference motivates the experiment. The exact S2 interval quoted earlier was not independently reproduced in the source plan; do not upgrade it to a verified claim here. [P1]

---

## 14. Conditional runs: a clean first-full comparison

Only add these when the first-full question affects the paper or deployment choice.

| ID | Transport | Scope | Training | Paired comparator |
|---|---|---|---|---|
| F0-PC | T0 | P-c | front9 | E0 |
| FW-PC | T1 or T4, selected and recorded | P-c | front9 | E1 or E2 |

Here front9 includes front6 plus full3. Use the same nine configuration-label sequence as flat training and transform only `[k]×N` into `[4]+[k]×(N−1)`. Keep common initialization, data, noise, objective, optimizer, and exposure matched.

This yields a 2×2 design: flat versus first-full, crossed with no transport versus the selected transport. It isolates schedule-family changes from trainable scope and configuration distribution. It still does not equalize block calls, so report cost with success.

The planned initial campaign is five new runs; these add at most two planned scientific runs. A failed historical provenance gate may separately require F2-PB-R.

---

## 15. Deferred mechanism studies—not an initial launch queue

These are defined so the final plan does not accidentally become an unlimited sweep.

| ID | Definition | When justified |
|---|---|---|
| D0-PC | T0,P-c; K_a1/2/4, NFE10 only; same distillation/anchor losses | Isolate depth elasticity |
| S0-PC | T0,P-c; K_a4 only, NFE2/5/10 | Isolate step elasticity |
| SPW11 | Same transport as the winning elastic method, specialized to [1,1] | Measure elasticity cost for that transport, not against a T0 specialist |
| E2-GATE005 | E2 recipe but initial gate 0.05 instead of 0.95; fresh run | Diagnose initialization sensitivity if warranted |
| E0-WEIGHTED / EW-WEIGHTED | Same explicitly chosen reweighted grid for both arms | Test a fastest-point exposure hypothesis without changing only one arm |
| T3 follow-up | Learned gate without T4 low-rank correction, matched recipe | Determine whether T4's low-rank term adds value beyond gating |
| Loss ablations | Remove one loss at a time from the selected method, matched data/budget | Support a specific objective-component claim |

A no-teacher multi-depth flow baseline needs its own definition: sample K, train instantaneous d=0 flow loss, and deploy that instantaneous field at different N with d=0. It is not automatically the same nine finite-step-target training problem with the teacher simply deleted. Its training-compute differences must be disclosed. It is deferred rather than ambiguously specified as a drop-in arm.

Do not run T2/T5/T6, multiple transport ranks, memory variants, or new donor grids before the core question is resolved.

---

## 16. Main-paper runs and independent seeds

### 16.1 Main LIBERO runs

After fixing the method from BASE10 experiments, use the predeclared stronger memoryless CL-0 checkpoint for headline validation, with a corresponding frozen teacher from that same checkpoint:

| ID pattern | Method | Fine-tuning seeds |
|---|---|---|
| MAIN-LIB-E0-s42/s43/s44 | Elastic without transport | 42,43,44 |
| MAIN-LIB-EW-s42/s43/s44 | Selected transport variant, if retained | 42,43,44 |

These are separate from BASE10 ablations. Do not initialize the student from CL-0 while quietly keeping BASE10 as the teacher, and do not substitute the carried-memory CL-A model.

Three fine-tuning seeds from one parent measure conditional fine-tuning variability. An end-to-end three-training-seed claim requires independent base/continuation/distillation pipelines. CL-0 is a continued-training memoryless control, not automatically a fresh 20-epoch-from-scratch model. Preserve that distinction. [P1]

If no transport survives, do not create an artificial “winner” only to fill the second row; main E0 can stand on its own, with transport results reported as ablations.

### 16.2 RoboTwin runs

Build a robot-compatible base and teacher with the correct action dimension, cameras, proprioception, normalization, and task data. A five-epoch base budget is a proposal to validate, not a guaranteed convergence criterion.

Use IDs `MAIN-RT-E0-s42/s43/s44` and, when relevant, `MAIN-RT-EW-s42/s43/s44`. The LIBERO window counts,32×7 action shapes, and ten-executed-action protocol must not be copied blindly into RoboTwin; its dataset contract must be recorded separately.

If the paper claims no embodied pretraining for this benchmark, do not silently initialize its base from a LIBERO-trained robot policy. General Wan video initialization and embodied policy initialization are different claims.

### 16.3 Experimental claims allowed

Elastic-grid evidence supports a finite-grid single-checkpoint deployment claim. It does not establish optimal adaptive compute allocation, monotonic performance, or novelty relative to all prior work. No “first” claim follows from this experiment plan alone.

---

## 17. Resources, ordering, and completion artifacts

### Launch order with four available GPUs

After capacity and correctness gates, use two two-GPU lanes:

| Lane A | Lane B |
|---|---|
| E0 | E2 |
| E1 | SP11 |
| F0-PB | Evaluation or remaining validation |

If the historical reconstruction fails before F0-PB starts, resolve its status without delaying the primary E0/E1/E2 experiment. Never cancel or alter unrelated running jobs as part of executing this plan unless separately authorized.

Keep global batch 128 using gradient accumulation. Use a common feasible P-c layout for matched arms; do not assume the old LoRA microbatch64 will fit the new P-c unroll. The pinned launcher accepts one/two GPUs per run; add and test four-GPU support explicitly if needed. [P1]

### Cost accounting

Record GPU-hours as the sum of allocated GPUs times measured elapsed hours. Profile the full graph, not just student deployment. At NFE2 the local teacher still takes ten micro-evaluations in total, plus the separate ten-step endpoint trajectory. Evaluation costs also depend on episode lengths and simulator throughput.

No fixed training-time or GPU-hour forecast is justified by the current plan. The55-condition total is exact bookkeeping; a per-condition time assumption is not a measurement.

### Required artifact per run

Each run should retain a machine-readable manifest containing run ID, parent/source/environment hashes, common-initialization hashes, scope and trainable names/counts, transport type, supported schedules, schedule sequence hash, data/noise seeds, optimizer groups, precision, loss weights, truncation policy, total updates/valid windows, raw-versus-EMA selection, and checkpoint hashes.

Each evaluated condition should additionally retain the evaluator/backend/hardware contract, action schedule, deployed d, episode identities/outcomes, missing/failed job status, aggregate counts, per-suite/task results, paired comparison outputs, latency definition, and resident-memory policy.

---

## 18. Final checklist and compact interpretation

- [ ] Understand the three axes: robot queries, denoising steps, and core loops.
- [ ] Use BASE10—not CL-0—for every initial ablation and its teacher.
- [ ] Keep video K_v=4 and late-aligned cache slots.
- [ ] Train E0/E1/E2 with P-c over the same nine flat settings.
- [ ] Train SP11 at [1,1] with the same total exposure and anchor.
- [ ] Reproduce F0-PB against RT-A2 only after historical provenance checks.
- [ ] Preserve own-state teacher targets, separate teacher endpoint, d conditioning, and exact gradient boundaries.
- [ ] Validate dynamic-unused DDP, resume, and teacher-free deployment before launch.
- [ ] Score19 decisive conditions, then complete55 total conditions.
- [ ] Select a method using paired evidence, training-seed confirmation, and actual latency—not only block counts.
- [ ] Add F0-PC/FW-PC only if a causal first-full claim is needed.
- [ ] Keep stronger-parent main results and robot-specific RoboTwin results separate.

**Core idea:** retain v0's deep cached video understanding, teach its action expert to solve both shallow-depth and long-step prediction tasks, and only then ask whether the preceding denoising step's hidden residual is still useful.

---

## 19. Source map

[P1] Conversation attachment `LoopWAM_Elastic_RT_Final_Run_Plan.md`.

[P2] Conversation attachment `LoopWAM_Elastic_RT_Run_Matrix.json`.

[R1] Repository `src/fastwam/models/wan22/loop_mot.py`, v0 snapshot `56e33285a82c6abe075565b3d60c26a5ba7a148f`; shared-depth layout and late-aligned cache mapping.

[R2] `src/fastwam/models/wan22/loopwam_init.py`, same v0 snapshot; dimensions and canonical Wan donor mapping.

[R3] `src/fastwam/models/wan22/loopwam.py`, same v0 snapshot; base training noise, targets, forward exits, and losses.

[R4] `src/fastwam/models/wan22/residual_transport.py`, snapshot `944a07f1e6f3d6ca8b96034370c5fc88e0cf9ac2`; schedules, T0–T6, step conditioning, LoRA, residual definition, teacher target, and gripper loss.

[R5] `src/fastwam/models/wan22/transport_policy.py`, same RT snapshot; scope, frozen prefill, teacher/student trajectories, loss aggregation, truncation, and rollout.

[R6] `scripts/train_transport.py`, same RT snapshot, as specified by P1/P2; training/recovery/export behavior must be verified again during implementation.

[R7] `plans/LoopWAM_S_Technical_Report.md`, v0 snapshot; declared parameter accounting and tensor contract. Parameter counts are architectural arithmetic, not memory measurements.

[R8] `structured_mixed_attention` in `loop_mot.py`, v0 snapshot; F/U/A visibility rules.

[W1] Official PyTorch DistributedDataParallel documentation, consulted 28 October 2026: `https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html`.

[W2] Official PyTorch Optimizer.zero_grad documentation, consulted 28 October 2026: `https://docs.pytorch.org/docs/main/generated/torch.optim.Optimizer.zero_grad.html`.

No simulated success rates, latency targets, or real-robot results from the ideal CVPR draft are treated as completed elastic experiments in this document.

