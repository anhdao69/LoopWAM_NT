# LoopWAM-S: a ~0.6B depth-recurrent world-action model
## Implementation specification, versioned experiments, and verification plan

**Prepared for Anh Dao · October 5, 2026**  
**Primary benchmark:** LIBERO. **Primary goal:** parameter-efficient manipulation with two generative experts and shared Transformer depth.  
**Status:** researched design and static source inspection. No LoopWAM implementation, checkpoint, success rate, latency, or GPU-memory result is claimed. The parameter totals below are analytical calculations for the declared modules, not measurements from a running model.

---

## 0. Executive decision

Build **LoopWAM-S from `Wan-AI/Wan2.1-T2V-1.3B`, inside a fork of FastWAM**, keeping separate video and action experts. Use the following architecture from the first robot-training run:

| Item | Video expert | Action expert |
|---|---:|---:|
| Hidden width | 1536 | 512 |
| Mixed-attention inner width | 1536 | 1536 |
| Mixed-attention heads × head dimension | 12 × 128 | 12 × 128 |
| Text cross-attention inner width | 1536 | 1536 |
| Text cross-attention heads × head dimension | 12 × 128 | 12 × 128 |
| FFN width | 6144 | 2048 |
| FFN activation | GELU, `approximate="tanh"` | Same |
| Unique prelude blocks | 3 | 3 |
| Unique shared-core blocks | 6 | 6 |
| Unique coda blocks | 3 | 3 |
| Default core repetitions | 4 | 4 |
| Unique blocks per expert | 12 | 12 |
| Effective block applications per expert | 30 | 30 |
| Output | Video-latent flow velocity | 32 × 7 action flow velocities |

The proposed model contains approximately **584.5M policy parameters**, excluding the frozen video VAE and text encoder. It contains **12 unique block pairs**, or **24 actual Transformer blocks across the two experts**. Saying “12 unique layers” without specifying “per expert” is ambiguous.

The initial development sequence is:

> **v0:** ordinary shared-depth MoT, final-exit flow matching only.  
> **v1:** v0 plus joint video/action deep supervision at all four loop exits.  
> **v2:** v1 plus XSA inside shared-core mixed attention.  
> **v3:** after v2 works, keep visual depth fixed and learn shorter action-depth configurations.  
> **v4:** optional further compression or adaptive budgets, not a prerequisite for the first result.

**Do not begin with a trained Fast-WAM teacher, knowledge distillation, per-loop LoRA, per-loop normalization, reinjection, a routing network, or separate video/action loop counts.** Those are optional later experiments. Initialize from a general video foundation model and learn manipulation directly from robot demonstrations.

The earlier `5 + 5×4 + 5` sketch was for a less aggressively compressed model. At the widths in this report, it would contain roughly **723.1M**, not 584.5M, policy parameters. The main design is therefore consistently:

\[
\boxed{3+6K+3,\qquad K\in\{1,2,3,4\}.}
\]

This is a transfer of Looped-DiT's *mechanism*, not a claim to reproduce its complete image-generation architecture or its published scaling numbers. [S17–S19]

---

## 1. What is verified, what is proposed, and what is uncertain

### 1.1 Source-derived facts

The official Wan2.1-T2V-1.3B configuration has width 1536, FFN width 8960, 12 heads, 30 layers, and 16 input/output latent channels. The corresponding VAE configuration has temporal/spatial stride `(4,8,8)`. Wan2.2-TI2V-5B instead has width 3072, FFN width 14336, 24 heads, 30 layers, 48 latent channels, and a VAE with spatial stride 16. [S01–S04]

FastWAM at the inspected `7faa711` revision is built around Wan2.2-TI2V-5B. Its action backbone is initialized by resizing Wan weights, not by loading a robot-trained action model. The selected implementation has separate video and action experts, with mixed self-attention and separate text cross-attention. [S05, S07–S11]

Light-WAM contains a working Wan2.1 model-family loader, including the correct VAE class and configuration preset. However, its default policy uses a state-fusion action decoder; that is **not** the two-generative-expert architecture specified here. [S06, S20]

Looped-DiT repeats a middle block group, decodes intermediate loop states through a shared coda, and optionally modifies shared-core attention through a head gate or XSA. Its inspected model uses a MiniT2I-derived pixel-space MMDiT, with no active diffusion-timestep conditioning in the model forward. Its image/text streams are not the same as Fast-WAM's video/action streams. [S17–S19]

### 1.2 Proposed choices

The 584.5M architecture, FFN pruning rule, core-layer donor selection, no-teacher first training, normalized deep-supervision loss, and versioned experiment sequence are **proposals in this report**. Their success on LIBERO is not established by the cited papers.

### 1.3 Important unresolved questions

We do not yet know whether the 6144-wide video FFN preserves enough of Wan's prior, whether four core repetitions improve control relative to a dense 12-layer model, whether XSA helps action denoising as much as visual processing, or whether shallow action networks benefit from deeper cached visual features. These are experiments, not assumed properties.

---

## 2. Foundation model choice: use Wan1.3B first, not Wan5B

### 2.1 The relevant small checkpoint is 1.3B

Use the exact repository name:

```text
Wan-AI/Wan2.1-T2V-1.3B
```

The small checkpoint reviewed here is not called “Wan1.5B.” Do not substitute a similarly named model or quantized derivative without checking its actual tensor configuration. [S01]

### 2.2 Why this is the better first initialization

Our target video residual width is 1536. Wan1.3B already has that width and the required 12 × 128 attention layout. This preserves the dimensions of its attention projections, residual stream, text projection output, timestep embeddings, Q/K normalization, and video head interface. Only the video FFN must be reduced to reach the chosen parameter budget.

Starting from Wan5B would require additional video residual-width conversion from 3072 to 1536, head selection/reduction from 24 to 12, and more extensive transformations of conditioning and output parameters. Such surgery is possible, but introduces another source of failure into an experiment whose central question is weight sharing. [S01, S02]

This is an engineering-risk argument, not evidence that Wan1.3B must outperform a compressed Wan5B initialization.

### 2.3 The real disadvantage of Wan1.3B: more visual tokens

At two concatenated 224×224 camera views, the RGB image is 224×448. With the native VAE and a `(1,2,2)` latent patch size:

| Quantity | Wan2.1 VAE | Wan2.2 VAE |
|---|---:|---:|
| Latent spatial size | 28 × 56 | 14 × 28 |
| Transformer patch grid | 14 × 28 | 7 × 14 |
| Tokens per latent frame | 392 | 98 |
| Tokens for three latent frames | 1176 | 294 |

These token counts are derived from the verified VAE strides and patch size. [S03, S04]

**A smaller source model does not automatically give a faster final policy.** The Wan1.3B route has four times as many visual tokens at this resolution. Linear token-projection work increases accordingly at fixed model dimensions; dense attention-pair work can increase approximately quadratically. Actual wall-clock differences depend on kernels, masks, projections, and preprocessing.

For the first study, accept this cost and preserve the native visual representation. Measure it. Do not silently downsample current-observation latents merely to make a timing table look better.

### 2.4 Do not mix the two VAEs

A Wan2.1 backbone with a Wan2.2 VAE is not a plug-compatible combination. Channel count, grid size, latent statistics, and pretrained input/output meanings differ. Changing only `in_dim` from 16 to 48 does not solve that mismatch.

The primary model uses **Wan2.1 weights + Wan2.1 VAE + matching latent normalization**. A Wan5B-derived model using Wan2.2's VAE is a separate model-family experiment requiring its own conversion and controls.

### 2.5 The 6144-wide FFN is an explicit trade-off

Wan1.3B does **not** natively have an MLP ratio of four. Its ratio is:

\[
8960/1536\approx5.8333.
\]

Our target ratio is four, which gives FFN width 6144. Keeping all 8960 FFN channels in the proposed 12-unique-block architecture would produce approximately **688.4M** policy parameters.

Therefore:

> Use **584.5M / FFN6144** as the requested main model. Keep **688.4M / FFN8960** as the first near-size diagnostic if FFN reduction is harmful.

Do not jump immediately to a 1B or 3B model when a 0.6B run underperforms. A native-FFN diagnostic isolates a much narrower question.

A different way to obtain approximately 0.6B is `2 + 6×4 + 2` with the native FFN, giving about 578.7M parameters and 28 effective layers. This avoids FFN pruning but changes the boundary depths and controlled depth comparison. It is a legitimate alternative, **not the default configuration in this report**. Do not train both designs before validating the main one.

---

## 3. Codebase decision and source pins

### 3.1 Main host: FastWAM

Fork:

```text
yuantianyuan01/FastWAM
```

Start from the known revision:

```text
7faa711
```

The resolved source revision is `7faa71108368fbb3b6885649f112af607427a2d4`. Store that full hash in every experiment manifest and use a separate development branch. Do not develop against a moving default branch while comparing experiments.

FastWAM is the appropriate host because the requested model retains a generative action DiT and the existing MoT conditioning/cache structure. Starting from Light-WAM's default state-fusion policy would require removing precisely the action decoder that defines that method and restoring a different action branch. [S05–S11, S20]

### 3.2 Borrow only the Wan2.1 loading support from Light-WAM

Reference revision observed in the code search:

```text
L1ziang/Light-WAM
db7d05e724f3eac2d4f0c9c3ac727fe93a44d727
```

Use its Wan2.1 family specification and VAE loading/conversion path as a reference. In particular, its source preset identifies `WanVideoVAE`, `Wan2.1_VAE.pth`, 16 latent channels, width 1536, FFN8960, and 30 source layers. [S06]

Do not copy its frozen-backbone policy, sparse WAM adapters, LoRA recipe, or `StateFusionActionExpert` into the main LoopWAM-S.

### 3.3 Looped-DiT is a reference implementation, not the training host

Inspected main revision:

```text
OpenSenseNova/Looped-DiT
65a7705ad2954fa127721bd1131ea8f865688848
```

Read `looped_dit/model.py` for the pre/core/post organization and XSA, and `looped_dit/diffusion.py` for intermediate-exit supervision. Do not replace FastWAM's latent/action scheduler with the pixel-image scheduler or load the Looped-DiT image checkpoint. [S18, S19]

### 3.4 Separate source configuration from target configuration

Maintain two explicit objects:

```text
source_wan_config: native 30-layer Wan1.3B, FFN8960
student_config: two experts, FFN6144/2048, 3/6/3 sharing
```

This is essential. Light-WAM's preset overwrites dimensions, including FFN width and number of layers. Applying that preset to a supposed 6144-wide, 12-unique-block student could silently reconstruct the native model instead. [S06]

The correct order is:

```text
Load and validate native source tensors
    -> construct target architecture independently
    -> perform explicit parameter transformations
    -> strict-load the transformed target state
```

Changing only `model_id` in the stock FastWAM YAML is insufficient. The pinned source uses a Wan2.2-specific loading function and preprocessor. [S05, S11]

---

## 4. Architecture: two experts, one shared-depth schedule

### 4.1 What is shared and what is not

Each expert owns three parameter groups:

```text
video:  P_v[0:3], B_v[0:6], C_v[0:3]
action: P_a[0:3], B_a[0:6], C_a[0:3]
```

`B_v[j]` is reused across all video loop iterations. `B_a[j]` is reused across all action loop iterations. **Video and action do not share their projection or FFN weights with each other.** Their widths differ; their attention outputs interact through a compatible inner head space.

Prelude and coda weights are distinct from each other and from the core. In v0–v2, core biases, norms, Q/K RMSNorm weights, and modulation tables are also shared across repetitions. There are no per-loop copies of these parameters.

The model is recurrent across **network depth**. It does not carry a recurrent state from one environment observation to the next. Clear observation-specific caches at each policy query.

### 4.2 The state entering the Transformer

Define:

- \(t_{\mathrm{env}}\): environment/control time;
- \(\sigma_v,\sigma_a\): video/action noise fractions;
- \(k\): core loop index;
- \(s\): action sampler step index;
- \(K\): number of core repetitions;
- \(H_a=32\): predicted action horizon.

These are different quantities. In particular, four depth loops do not mean four denoising steps or four environment actions.

The video input is a noisy latent clip with its first latent frame replaced by the clean current-observation latent. The action input is a noisy normalized action chunk. Each branch receives its own noise conditioning; they need not share the same sampled noise level.

### 4.3 Shared context

Frozen text encoding gives:

\[
E_\ell\in\mathbb R^{B\times128\times4096}.
\]

Project the current 8-D proprioceptive state to 4096 and append it as one valid context token:

\[
c=[E_\ell;W_p p_t+b_p]
\in\mathbb R^{B\times129\times4096}.
\]

The video and action experts then use their own context projections:

\[
c_v=\operatorname{MLP}_{4096\to1536\to1536}(c),
\qquad
c_a=\operatorname{MLP}_{4096\to512\to512}(c).
\]

Keep the padding mask from the text encoder. Appending proprioception does not make padded language tokens valid. Proprioception is *current state only*, not a future-state sequence. The FastWAM input path already selects the first proprioceptive state. [S10]

### 4.4 Why action width 512 can use 12 heads

The action residual stream is 512-dimensional, but its mixed-attention Q/K/V are projected to 1536 dimensions:

\[
W^a_Q,W^a_K,W^a_V:\mathbb R^{512}\to\mathbb R^{1536},
\qquad
W^a_O:\mathbb R^{1536}\to\mathbb R^{512}.
\]

Thus, action and video both produce 12 heads of width 128. Their K/V can be concatenated along the token dimension. We are **not** forcing the action residual stream to have width 1536.

Keep action text cross-attention at the same 12 × 128 inner width initially. Narrowing that branch to four or six heads is unnecessary additional surgery. The resulting action expert is still only about 105.3M parameters.

### 4.5 Exact inherited block equations

For expert \(m\in\{v,a\}\), let \(x_m\) be its current residual state. The physical block has a learned modulation table \(M_m\in\mathbb R^{1\times6\times d_m}\). Its timestep projection supplies \(T_m\), potentially per token in the video branch:

\[
(\beta_1,\gamma_1,g_1,\beta_2,\gamma_2,g_2)=M_m+T_m.
\]

Here the notation means splitting the final six modulation groups, not six independent learned networks.

\[
u_m=(1+\gamma_1)\odot\operatorname{LN}_1(x_m)+\beta_1.
\]

Compute Q/K/V with the inherited Q/K normalization and RoPE. Mixed attention gives \(o_m\). Then:

\[
x'_m=x_m+g_1\odot W^m_O\widetilde o_m,
\]

\[
x''_m=x'_m+\operatorname{CrossAttn}_m(\operatorname{LN}_{3,m}(x'_m),c_m),
\]

\[
x_m^+=x''_m+g_2\odot W^m_2\operatorname{GELU}\left(
W^m_1[(1+\gamma_2)\odot\operatorname{LN}_2(x''_m)+\beta_2]+b^m_1
\right)+g_2\odot b^m_2.
\]

For v0/v1, \(\widetilde o_m=o_m\). For v2, apply XSA to the head outputs before \(W_O\).

Preserve FastWAM's normalization details: LN1/LN2 are affine-free, LN3 is affine, and the Q/K RMSNorm modules operate on the inherited flattened attention representation. Do not silently replace them with Looped-DiT's normalization layout. [S07, S18]

### 4.6 Joint masking is the control interface

Order tokens as `[current-video F; future-video U; action A]`. The allowed-attention matrix is:

| Query group \ Key group | F | U | A |
|---|---:|---:|---:|
| F | Yes | No | No |
| U | Yes | Yes | No |
| A | Yes | No | Yes |

For text cross-attention, each branch reads valid context tokens separately. Language tokens are **not** appended to this mixed self-attention sequence.

This mask preserves FastWAM's first-frame-only policy: actions do not read noisy or clean future video tokens, and video does not read action tokens. This is what makes current-observation K/V reusable during action denoising. [S10]

The future-video objective nevertheless updates shared video weights. The policy therefore benefits, if the hypothesis holds, from world-supervised representation learning without running future imagination online.

### 4.7 Loop recurrence and exit decoding

Let \(P\), \(B\), and \(C\) denote joint video/action prelude, shared core, and coda computations:

\[
(V^{(0)},A^{(0)})=P(V_{\mathrm{in}},A_{\mathrm{in}}),
\]

\[
(V^{(k)},A^{(k)})=B(V^{(k-1)},A^{(k-1)}),\quad k=1,\ldots,K,
\]

\[
(\widehat u_v^{(k)},\widehat u_a^{(k)})
=\operatorname{Heads}(C(V^{(k)},A^{(k)})).
\]

Every application of \(B\) reuses the same six physical block pairs. Every exit reuses the same coda and output heads.

**There is no extra outer residual `h + B(h)`.** The inherited blocks already contain residual connections. There is also no reset to the original noisy token embedding between core loops.

Do not feed an intermediate coda's output back into the next loop. The next loop receives \((V^{(k)},A^{(k)})\), not \(C(V^{(k)},A^{(k)})\).

### 4.8 Timestep and loop-index conditioning

Keep Wan/FastWAM noise conditioning. Set clean first-frame tokens' noise timestep to zero; noisy future tokens use their sampled video timestep. Action tokens use the sampled action timestep. Inspect and preserve the tokenwise construction in `WanVideoDiT.prepare`.

Use **no additional loop-index embedding in v0–v2**. Reusing a block with a different hidden state is already recurrence; the model does not require a learned loop label to be well-defined. This keeps the first implementation closer to pure weight sharing.

The spatial/temporal RoPE coordinates remain the coordinates of the original video patches or action positions. Do not advance an action-position or video-time coordinate merely because another depth loop is being executed.

---

## 5. Parameter accounting

For hidden width \(d\), attention inner width \(D\), and FFN width \(f\), the declared FastWAM-style block has:

\[
P_{\mathrm{block}}=8dD+2df+10D+11d+f.
\]

The first two terms are large matrix weights. The remaining terms count the specified projection biases, FFN biases, Q/K RMSNorm weights, affine cross-attention normalization, and six-way modulation table. LN1/LN2 have no affine parameters.

| Quantity | Calculated parameters |
|---|---:|
| One video block | 37,787,136 |
| One action block | 8,411,648 |
| Twelve video blocks plus video globals | 479,221,312 |
| Twelve action blocks plus action globals | 105,277,959 |
| Proprioception projection | 36,864 |
| **LoopWAM-S v0/v1/v2 total** | **584,536,135** |
| Dense-S12, same dimensions | 584,536,135 |
| Dense-S30, same dimensions | 1,416,114,247 |
| LoopWAM with native video FFN8960 | 688,378,951 |

XSA adds no parameters. A learned head gate would add a small number of parameters and must be counted separately.

The analytical compression ratio against Dense-S30 is approximately **2.42×**, or **58.7% fewer policy parameters**. It is not fourfold, because the prelude, coda, and global modules are not looped.

BF16 storage for 584,536,135 policy parameters is approximately **1.169 GB decimal** before file-format overhead. This is not total inference VRAM and not total pipeline storage. Add the actually loaded VAE, cached tensors, workspaces, and any online text encoder. A training checkpoint with optimizer state has a different size again.

The sidecar `LoopWAM_S_Analytical_Counts.json` records these calculations. Replace the predicted total with a deduplicated runtime parameter count once the implementation exists.

---

## 6. Data, latent shapes, and action alignment

### 6.1 Retain the FastWAM LIBERO sample contract

Use the preprocessed `yuanty/LIBERO-fastwam` data with the pinned environment/data revision. The reviewed configuration has two 224×224 views, a 224×448 horizontal concatenation, 7-D actions, 8-D proprioception, and a 33-timestep raw window. Video is sampled every fourth timestep. [S13]

A window anchored at environment step \(t\) contains:

\[
(o_t,o_{t+4},o_{t+8},\ldots,o_{t+32})
\]

for video supervision, and:

\[
(a_t,a_{t+1},\ldots,a_{t+31})
\]

for action supervision.

Check the dataset's actual index convention by inspecting a sample and its timestamps. Do not assume that a stored action at index \(t\) has already been shifted correctly merely because tensor shapes match.

### 6.2 Primary tensor contract

| Tensor | Expected shape |
|---|---|
| RGB video after camera concatenation | `[B,3,9,224,448]` |
| Wan2.1 encoded clip | `[B,16,3,28,56]` |
| Clean observation latent | `[B,16,1,28,56]` |
| Future target latent portion | `[B,16,2,28,56]` |
| Video tokens | `[B,1176,1536]` |
| First-frame video tokens | `[B,392,1536]` |
| Action data / flow target | `[B,32,7]` |
| Action hidden tokens | `[B,32,512]` |
| Video/action Q, K, V after head reshape | `[B,12,N,128]` |
| Raw cached text context | `[B,128,4096]` |
| Context after proprio append | `[B,129,4096]` |
| Mixed training token count | 1208 |

Derive grids from actual encoder output and patch size; assert these expected LIBERO values in tests. Do not use the old 98-token first-frame constant from Wan2.2.

### 6.3 VAE and text caching

Freeze the native VAE and text encoder. For the initial smoke test, use online frozen VAE encoding and cached text. For longer runs, precompute deterministic clip latents, keyed by dataset revision, episode ID, anchor timestep, camera order, image transform, VAE hash, normalization, and temporal sampling.

**Do not cache one whole episode and then casually treat its latent frame at time t as the standalone observation encoding at time t.** A causal temporal encoder can still depend on past frames. The observation token must match the inference-time current-image encoding. Validate equality for the first frame of each encoded clip with encoder state reset.

Use per-clip encoding, or a more sophisticated cache only after proving the same anchoring semantics. A cached latent from Wan2.2 is not reusable for Wan2.1.

Apply the VAE's own normalization exactly once. Keep the text encoder's token mask. Caching instructions used at evaluation does not expose action labels, but its cold-start encoding cost still belongs in a deployment report.

### 6.4 Padding and normalization

Use training-set action normalization statistics and store their hash. The same statistics must be used for every matched architecture and for evaluation. Do not load a statistics file with a different action ordering, gripper convention, or suite mixture.

Retain `action_is_pad` and `image_is_pad`. The reviewed video-loss code groups the tail RGB padding mask by temporal VAE compression and marks a latent group padded when all its source tail entries are padded. Preserve this behavior for the initial comparisons; changing it is a separate data-policy change. [S10]

Never let zero padding lower a loss by increasing its denominator. Every branch loss is normalized over its own valid entries. Keep the loader's end-of-episode sample policy the same across baselines.

### 6.5 Prediction horizon versus executed horizon

Predict and supervise all **32 actions** in this plan. At evaluation, execute the first **10** before acquiring a new observation and replanning, matching the inspected FastWAM evaluation configuration. [S14]

Do not reuse the earlier eight-action Light-WAM proposal here. This design has a generative 32-token action branch, not the compact direct-action head discussed previously.

The number of actions returned per query is not the frequency of fresh-observation decisions.

---

## 7. Initialization: native Wan → compact expert pair → looped model

### 7.1 No robot-trained teacher is required

The first training uses no `libero_uncond_2cam224.pt` initialization and no teacher forward. It uses general Wan pretraining plus robot demonstration supervision. This is **Wan-initialized robot training**, not random initialization of the entire system.

Initialization and distillation are different operations. Copying or resizing pretrained parameters does not require a teacher prediction loss.

### 7.2 Make one canonical 30-layer donor pair

Construct a reusable initialization artifact containing:

```text
V1 ... V30: width1536, video FFN6144, 12×128 attention
A1 ... A30: width512, action FFN2048, 12×128 attention
video globals
new action input/output modules
action globals derived from Wan
new proprioception projection
metadata: source hashes, tensor map, transformations, random seed
```

These are *initialization donors*, not robot-trained teacher layers. All new dense and looped architectures are constructed from this same artifact. This removes inconsistent conversion code as a source of experimental differences.

### 7.3 Video FFN conversion

Load the native Wan1.3B model/state at its native dimensions first. For every source video layer, choose 6144 retained FFN neuron indices using:

\[
J_q=\operatorname{round}\left(q\frac{8959}{6143}\right),
\quad q=0,\ldots,6143.
\]

Assert that `J` is sorted, unique, and in range. Use the same index rule for every source layer and store the actual index array in the manifest.

For an FFN with native weights \(W_1\in\mathbb R^{8960\times1536}\), \(W_2\in\mathbb R^{1536\times8960}\):

\[
W'_1=W_1[J,:],\quad b'_1=b_1[J],
\]
\[
W'_2=W_2[:,J],\quad b'_2=b_2.
\]

This keeps incoming/outgoing weights aligned for each retained neuron. Do not choose different neuron subsets for the two projections. Do not independently interpolate the two FFN axes and call that neuron selection.

Leave all video attention matrices, norms, modulation tables, patch embedding, text/time globals, and output head at their native shapes. Use no extra variance-rescaling heuristic for the pruned video FFN in v0.

This rule is simple and data-free, **not claimed to be the optimal pruning method**. It preserves the selected neuron's input/output pairing, not the complete native FFN function. An equivalence test can compare the converted FFN against the original FFN with all omitted intermediate activations explicitly zeroed.

### 7.4 Action backbone conversion

Derive the action donor layers from the **native Wan1.3B tensors before video FFN pruning**, using FastWAM's existing sequential linear interpolation and last-dimension scaling rule as the reference. This avoids making the action prior depend on an additional video pruning decision. [S11]

The target configuration is:

```text
hidden_dim: 512
ffn_dim: 2048
num_heads: 12
attn_head_dim: 128
num_layers: 30
text_dim: 4096
freq_dim: 256
eps: 1e-6
```

For each eligible backbone tensor, resize to the corresponding target tensor shape. Preserve the reference scaling rule:

\[
\alpha=\sqrt{d_{\mathrm{source,last}}/d_{\mathrm{target,last}}}
\]

when the source tensor has at least two dimensions and its last dimension changes. One-dimensional tensors are not multiplied by this matrix scaling factor.

The reference resizes tensors by sequential one-dimensional linear interpolation with `align_corners=True`. It is an initialization heuristic, not an exact function-preserving map. Do not claim that a width512 action block is numerically equal to the native video block. [S11]

Retain the 1536-dimensional Q/K norm weights where their shapes already match. Attention head count and head width stay at 12 × 128. Do not reuse the preprocessed 1024-D action checkpoint derived from Wan5B; it has a different source family and attention layout.

### 7.5 Explicit initialization table

| Module | Initialization |
|---|---|
| Video attention, norms, modulation | Copy native Wan1.3B values |
| Video FFN | Paired neuron selection to 6144 |
| Video patch/text/time/output globals | Copy native compatible values |
| Action attention/FFN/norm/modulation | Reference Wan→action resizing from native Wan1.3B |
| Action text/time globals | Same documented backbone resizing map |
| Action input projection, 7→512 | Newly initialized |
| Action output projection, 512→7 | Newly initialized |
| Proprioception projection, 8→4096 | Newly initialized |
| VAE/text encoder | Native pretrained, frozen |
| Loop-index embeddings | Absent |
| Per-loop LoRA / per-loop norms | Absent |

Use the existing action input/output initialization unless an explicit alternative is tested. Copy those freshly initialized tensors into every matched model so that different model constructors do not generate different action heads accidentally.

The action output is the existing linear flow-velocity head; do not add a timestep-modulated head just because an unused `ActionHead` class exists in a source file. Follow the instantiated `ActionDiT.head` path. [S08]

### 7.6 Default core donor selection: copy, do not average first

Use one-based numbering below and convert it explicitly to zero-based slices in code:

| Student group | Source layers | Python source slice |
|---|---|---|
| Prelude | 1, 2, 3 | `[:3]` |
| Shared core | 10, 11, 12, 13, 14, 15 | `[9:15]` |
| Coda | 28, 29, 30 | `[27:30]` |

Apply the **same donor-layer map to both expert families**. Each of the six core blocks is copied once and called four times; it is not copied four times.

This chooses one central contiguous donor segment. The other central segment, layers 16–21, is a reasonable alternative; the first choice is not evidence-based superiority. We specify one to remove ambiguity and avoid selecting among many initializations before the first run.

The reason to prefer copying over cycle averaging initially is practical: copying preserves an actual pretrained block's internal combination of attention, FFN, norm, and modulation parameters. Averaging different-depth blocks is not equivalent to averaging their functions and can obscure the usefulness of the original prior.

The earlier conversation described cycle averaging as the preferred choice. That preference was not supported by an experiment. In this report, **central-copy is the conservative default and cycle averaging is a named ablation**, not an established better method.

The cycle-mean alternative for core block \(j\in\{1,\ldots,6\}\) is:

\[
\theta_{B_j}=\frac14\sum_{r=0}^{3}\theta_{3+j+6r}.
\]

Do not introduce per-loop residual LoRA simply to make this averaging reconstruct the original 30-layer model. That would turn the first study into a different relaxed-sharing method.

### 7.7 Matched dense controls

**Dense-S12:** use exactly the same twelve donor blocks, global modules, and new heads as LoopWAM-S, but execute the six middle blocks once. At initialization, its output must match LoopWAM-S at K=1.

**Dense-S30:** use all thirty compact donor layers in order. It isolates the effect of tying at fixed dimensions and effective depth. It is about 1.416B, so it is a diagnostic/control model, not the headline deployment model.

An optional separate diagnostic expands the looped model into thirty untied copies initialized from the looped model itself. This gives exact K=4 forward equivalence at initialization and tests subsequent tied versus untied optimization without donor-map differences. Do not confuse it with the stronger all-native-layer Dense-S30 control.

### 7.8 Loading, serialization, and precision

Perform transformation in FP32 on CPU where practical, validate shapes, then export an initialization checkpoint. Avoid holding a source Wan5B model, an unrelated robot teacher, and the student on GPU during training.

The checkpoint must include source configuration, target configuration, source checkpoint revision/hash, core donor indices, FFN indices, action resizing policy, random seed for new heads, tokenization/normalization metadata, and architecture version.

Strict-load the final target state. Do not use `strict=False` to suppress a missing action timestep projection or wrong latent head shape. A requested initialized tensor with no source mapping is an error unless it is explicitly declared newly initialized.

---

## 8. v0: pure looping with ordinary world/action losses

### 8.1 Goal

Determine whether shared depth is useful **before** adding loop-specific training techniques. The scientific comparison is not “can we compress a trained robot teacher?” It is “what changes when a Wan-initialized WAM learns manipulation with shared rather than distinct depth?”

### 8.2 Noise convention

Use FastWAM's continuous flow-matching scheduler, not Looped-DiT's pixel-space scheduler. For clean target \(y\), noise \(\epsilon\sim\mathcal N(0,I)\), and noise fraction \(\sigma\):

\[
y_\sigma=(1-\sigma)y+\sigma\epsilon,
\qquad u^*=\epsilon-y.
\]

For video and actions, sample separate noise tensors and separate noise fractions. On the video input, overwrite the first latent frame with the clean observation and assign it zero noise timestep.

At the pinned source revision, the scheduler samples \(u\sim U(0,1)\) and applies:

\[
\sigma=\phi_s(u)=\frac{su}{1+(s-1)u},
\qquad t=1000\sigma.
\]

Its training-weight function is also explicit in the implementation. Reuse it rather than quietly replacing it with an unweighted MSE. [S12]

**Source discrepancy:** Fast-WAM's paper describes a logit-normal schedule, while the inspected revision implements shifted-uniform sampling. These are not the same distribution. This report uses the pinned **code** as the operational specification and records that choice; it does not silently treat the descriptions as equivalent. [S12, S16]

### 8.3 Explicit scheduler values

For newly trained models in this report:

```text
video train/inference shift: 5.0
action train/inference shift: 1.0
training timestep scale: 1000
action inference evaluations: 10
CFG: 1.0
```

The action shift of 1.0 is the pinned configuration's current value. Original released robot checkpoints may use 5.0; evaluate them with their own documented settings. Because the main model has no teacher, there is no requirement that its schedule equal an old teacher's schedule. [S05]

### 8.4 Masked branch objectives

Let \(m^a_{b,h}\) mark valid actions. First average error over seven action dimensions, then average over valid horizon positions per example:

\[
\ell_a^{(k)}(b)=
\frac{\sum_h m^a_{b,h}\operatorname{mean}_{d_a^{\mathrm{out}}}
(\widehat u_a^{(k)}-u_a^*)^2}
{\max(1,\sum_hm^a_{b,h})}.
\]

For video, average over channels/spatial coordinates per latent frame and then valid future latent frames. Exclude the clean first frame from both numerator and denominator.

Apply the inherited scheduler weighting per sample:

\[
L_a^{(k)}=\frac1B\sum_b w_a(t_{a,b})\ell_a^{(k)}(b),
\qquad
L_v^{(k)}=\frac1B\sum_b w_v(t_{v,b})\ell_v^{(k)}(b).
\]

Define:

\[
\ell_k=L_a^{(k)}+\lambda_vL_v^{(k)},\qquad\lambda_v=1.
\]

The v0 objective is simply:

\[
\boxed{L_{v0}=\ell_4.}
\]

Log raw per-branch losses, weighted per-branch losses, action/world gradient norms, and control metrics. The different number of action/video elements must not automatically determine their relative loss weights.

### 8.5 Training-step contract

Construct the noisy inputs once per batch. Pass the same tensors through the entire recurrent computation. Do not resample diffusion noise between depth loops, advance the action sampler inside the core loop, or change the target at different exits.

All trainable student layers receive ordinary backpropagation through every core application. Do not detach the state between loops or divide shared-parameter gradients by K. The gradient of the recurrent computation already sums the contributions from the different uses of each parameter.

### 8.6 What counts as success

The important first result is whether the looped model exceeds Dense-S12 at a comparable parameter budget and narrows the gap to Dense-S30. It need not exceed the pretrained source model's general video-generation quality or immediately equal a published Fast-WAM score obtained under another model family.

A poorly trained v0 is not sufficient evidence that recurrence is impossible; the next test is explicit intermediate supervision, after verifying conversion and training correctness.

---

## 9. v1: Looped-DiT-style deep supervision of both experts

### 9.1 Shared exit decoders

After each of the four core repetitions, run the same three coda block pairs and the same video/action heads. Supervise both outputs against the same noisy-input targets used by the final exit.

Use the current loop state as a branch point:

```text
core state k ───> shared coda ───> action/video losses at exit k
      │
      └────────> next shared-core repetition
```

The decoded state is not fed back into recurrence. Avoid in-place changes to the tensors saved for the next core loop.

### 9.2 Recommended primary weighting

Looped-DiT's implementation uses final-plus-mean exit weights `(1/3,1/3,1/3,1)`, summing to two. [S19]

For the main controlled WAM comparison, preserve those **relative** weights but normalize the total to one:

\[
\boxed{L_{v1}=\tfrac12\ell_4+\tfrac16(\ell_1+\ell_2+\ell_3).}
\]

This is an explicitly proposed scale normalization, not a literal copy of the paper's absolute loss multiplier. It avoids changing the overall action/video objective scale merely by enabling deep supervision, which matters when gradient clipping is fixed.

A literal-paper setting is available as `exit_weight_scale: 2.0`. If it is tested, also compare the correspondingly scaled final-only baseline. Do not attribute a global gradient-scale effect to deep supervision.

### 9.3 Training cost

For each expert, final-only v0 executes:

\[
3+4\times6+3=30
\]

blocks per training sample forward. All-exit v1 executes:

\[
3+4\times6+4\times3=39.
\]

Thus it adds nine coda-block executions per expert, approximately 30% more block applications before accounting for heads, kernels, and backward. The measured training overhead may differ.

At inference at a selected K, run the coda **once**. v1 does not require decoding every exit during deployment. Its same-K graph and parameter count match v0.

### 9.4 Memory fallback, not a new method

First implement all four exits; this is closest to Looped-DiT and easiest to validate. If memory is limiting, sample \(r\sim U\{1,2,3\}\) and use:

\[
L_{v1,\mathrm{sampled}}=\tfrac12\ell_4+\tfrac12\ell_r.
\]

Its expectation matches the normalized all-exit objective, although the gradient variance differs. This runs 33 rather than 39 block applications per expert. Label it sampled deep supervision and use the same setting in relevant controls.

### 9.5 Versions are not automatically extra training stages

For architecture comparisons, train v0 and v1 from the **same initialization artifact** with equal data exposure and a documented compute budget. Do not compare v0 after ten epochs to v1 obtained by adding ten more epochs to v0 and call the difference a pure deep-supervision gain.

Warm-started continuations can be useful for engineering, but require matched continuations in the final ablation.

---

## 10. v2: self-modulating mixed attention, beginning with XSA

### 10.1 Exact mechanism

Let \(o_{i,h}\) be the mixed-attention output for query token i and head h. Let \(v_{i,h}\) be **that same token's own value projection**, prior to attention mixing. Define:

\[
\widehat v_{i,h}=\operatorname{normalize}(v_{i,h}),
\]

\[
\boxed{
\widetilde o_{i,h}=o_{i,h}
-\langle o_{i,h},\widehat v_{i,h}\rangle\widehat v_{i,h}.
}
\]

This is a parameter-free projection orthogonal to the token's own value direction. The reference implementation performs normalization and projection arithmetic in FP32 and converts the result back to the attention output dtype. [S18]

**XSA is not diagonal masking, and it is not just subtracting `attention_weight_ii * value_i`.** Projection can remove components of other tokens' contributions that align with the query token's value direction. Do not describe it as retaining all non-self information unchanged.

### 10.2 Placement in FastWAM's MoT path

For a joint pass:

```text
per-expert normalized/modulated inputs
    -> video Q/K/V and action Q/K/V
    -> concatenate along token dimension
    -> mixed attention with the unchanged F/U/A mask
    -> XSA using each query's own value vector
    -> split video and action outputs
    -> each expert's O projection
    -> inherited gated residual update
    -> text cross-attention
    -> inherited FFN update
```

Only shared-core block pairs use XSA. Prelude, coda, and text cross-attention remain unchanged in the primary v2 experiment.

Apply XSA to both expert token groups. There are no recurrent text hidden states in this WAM to modify.

### 10.3 Cached action inference requires different indexing

In action-only cached inference:

```text
queries = action Q
keys    = [cached observation K; current action K]
values  = [cached observation V; current action V]
```

The query sequence length is 32; the value sequence length is 392+32. For XSA, use **current action V** as the query-aligned own-value tensor. Do not use the first 32 cached video values or pass the full concatenated V tensor directly to a query-aligned projection.

In video prefill, each video query's own value is its current video V. This distinction must be covered by joint-versus-cached equivalence tests.

### 10.4 Reference-level pseudocode

The following is an implementation sketch, not a drop-in patch for the repository:

```python
# Both tensors: [batch, heads, query_tokens, head_dim].
def xsa_projection(attention_out, own_query_value):
    own_unit = F.normalize(own_query_value.float(), dim=-1)
    out32 = attention_out.float()
    correction = (out32 * own_unit).sum(dim=-1, keepdim=True) * own_unit
    return (out32 - correction).to(attention_out.dtype)
```

Use the same epsilon convention as the chosen reference implementation; include zero/near-zero value tests. Do not materialize attention matrices solely to implement XSA: it is applied to the head outputs after the attention operation.

### 10.5 Wan already has gates, but they are not XSA

The inherited Wan block has timestep-dependent channelwise residual gates. These remain in place. XSA changes the geometry of the head output before the output projection. The mechanisms are distinct; do not remove Wan's pretrained gates when adding XSA.

A learned head gate is an optional alternative:

\[
\widetilde o_{i,h}=\operatorname{sigmoid}(w_h^\top u_i+b_h)o_{i,h}.
\]

Do not enable a learned gate and XSA together in the first experiment. If testing a sigmoid gate, state its initialization: zero bias is not identity initialization. XSA itself is also not a no-op at initialization.

### 10.6 Minimal v2 ablations

The primary result is v1 versus v2 with XSA in both core experts. If v2 hurts, compare video-only and action-only XSA to locate the issue before adding a new mechanism. An optional gate variant can distinguish directional projection from generic update attenuation.

For a final paper, include a dense-XSA control. Otherwise the measured gain may be due to XSA generally rather than its interaction with recurrence.

---

## 11. Deployment: prefill visual depth once, then denoise actions

### 11.1 Action-only policy inference

At an environment query:

1. Encode the current two-camera image with the native VAE and obtain `[B,16,1,28,56]`.
2. Retrieve or compute the instruction embedding and append current proprioception.
3. Run video prelude, K core repetitions, and video coda at zero video timestep, storing first-frame K/V at every virtual layer.
4. Initialize a 32×7 action chunk from standard Gaussian noise.
5. For ten action-sampler evaluations, run the action expert at K loops with the matching visual K/V, then perform the scheduler update.
6. Denormalize actions, apply the pinned gripper handling, execute ten actions, and reobserve.

No future RGB generation and no VAE decoding are required for policy execution. The video head and decoder can be omitted from the policy forward, but report whatever modules actually remain resident in memory.

### 11.2 Cache keys are virtual-layer keys

Use cache identifiers such as:

```text
("pre", j)
("core", k, j)
("coda", exit_k, j)
```

The same physical block can produce different K/V at different loop iterations because its input state is different. **Weight sharing does not mean K/V sharing across loop iterations.** Keying a cache only by physical block ID would overwrite or reuse incorrect representations.

At K=4, each expert has thirty effective layer applications. BF16 storage for observation K and V alone is approximately:

\[
2\times30\times392\times1536\times2
=72{,}253{,}440\text{ bytes}\approx68.9\text{ MiB}
\]

at batch one. This is an analytical cache payload, not peak VRAM.

### 11.3 Action state resets between denoiser evaluations

The *noisy action sample* advances through the Euler sampler. However, its Transformer hidden states are freshly computed from that sample at every denoiser evaluation. Do not carry the previous diffusion step's final hidden state into the next call unless creating a separate recurrent-sampler method.

For the inherited convention, \(\sigma\) decreases from one to zero:

\[
A_{s+1}=A_s+(\sigma_{s+1}-\sigma_s)\widehat u_a(A_s,\sigma_s).
\]

Do not copy Looped-DiT's noise-to-image time direction into this implementation without transforming the target and scheduler consistently. [S12, S19]

### 11.4 Compile only after correctness

First run eager inference and compare it to the joint reference. Then compile one fixed-shape graph per K for visual prefill and action denoising. Report compilation time separately and warm each measured graph.

The pinned FastWAM core rejects its existing gradient-checkpointing switches in the compiled path. A boolean flip is not a supported implementation of checkpointed looping. Keep compilation/checkpointing interactions out of the first correctness test; add a non-reentrant checkpointed joint-block wrapper only if memory measurements justify it. Pass the checkpoint variant explicitly and validate backward equivalence. [S09, S23]

---

## 12. v3: independent visual and action depth, only after v0–v2

### 12.1 Begin with one-sided elasticity

Do not implement the entire ten-pair grid immediately. First fix visual depth:

\[
K_v=4
\]

and train/evaluate:

\[
(K_v,K_a)\in\{(4,1),(4,2),(4,3),(4,4)\}.
\]

This keeps the expensive visual prefill stable while asking whether the repeatedly executed action network needs fewer layer applications.

Call this **visual-depth/action-depth decoupling**. In the FastWAM first-frame policy, larger K_v is deeper current-observation processing, not extra future imagination online.

### 12.2 A simple late-aligned interface

For action core repetition r, use the same physical action core weights and select video cache loop:

\[
\sigma_v(r)=K_v-K_a+r.
\]

The action prelude reads video-prelude caches; its coda reads the video coda associated with the selected video exit. For K_v=4 and K_a=1, all six action-core blocks read their counterpart caches from video loop four.

Crucially, **there are no per-loop action slots in this main architecture**. Changing K_v does not silently select a different action LoRA or normalization table. The action parameter set stays fixed.

Nevertheless, different alignment schedules change the conditioning interface and need training. Do not infer good asymmetric performance solely from coupled K=1…4 exits.

### 12.3 Extension objective and controls

Fork the selected coupled checkpoint. For each extension batch, retain full-depth action supervision and sample one shorter action budget. A simple normalized action loss is:

\[
L_a^{\mathrm{ext}}=\tfrac12L_a^{(4,4)}+\tfrac12L_a^{(4,K_a)},
\quad K_a\sim U\{1,2,3\}.
\]

Keep the video objective explicitly fixed in both the asymmetric run and its continuation control. Either retain v2's all-exit video objective in both, or switch both to final-video-only supervision; do not change it for just one arm of the comparison.

Compare against a coupled continuation with the same number of additional updates and comparable extra action passes, and against fixed-depth compact action baselines. Report training cost separately.

Once this works, add the full triangular pair set K_a≤K_v. Treat prefix elasticity and coda-at-exit behavior as tested contracts, not assumptions.

### 12.4 What this version must demonstrate

The strongest result would be a trained (4,1) or (4,2) operating point that improves the measured success/latency trade-off over shallower coupled settings and dense compact controls.

The fixed source model's dimensions, data, action horizon, sampler, and perception preprocessing must remain matched. A single higher SR at a larger total compute budget is not sufficient evidence of a useful budget allocation.

### 12.5 v4: optional further compression

The main model is already small. Do **not** follow the old example's later `3072→2048` width reduction: this model starts at video width1536/action width512.

One optional ~0.49B design keeps these widths but uses `3 + 4×6 + 3`: ten unique blocks per expert and thirty effective block applications. Its analytical parameter count is approximately 492.1M. It requires training at six loops; it is not obtained by merely asking an unprepared four-loop checkpoint to extrapolate.

Other future choices include token reduction, a Wan2.2-derived low-token variant, or calibrated adaptive K. Each is a separate mechanism, not a prerequisite for the first paper-quality comparison.

---

## 13. Concrete LIBERO training recipe

### 13.1 Initial scientific screen

Use LIBERO-Long (`libero_10`) for the first substantive model comparison. For development, split demonstrations by episode, not by overlapping windows: for tasks with fifty demonstrations, use forty-five for training and five for open-loop validation, with a recorded deterministic split. Generate all sliding windows only after this split.

A tiny single-task/single-batch test precedes this screen, but is only an implementation check. Do not select architecture hyperparameters using a succession of test-set checkpoints without disclosing the selection procedure.

After the recipe is fixed, train one multitask policy on all four suites following the chosen FastWAM-style setup. Retrain from the same generic Wan initialization artifact; do not initialize the final policy from a Long-specialized screening checkpoint. Record whether the final data includes all demonstrations or retains a validation split.

### 13.2 Main hyperparameters

The following is a concrete proposed recipe based on the pinned FastWAM training style, not a claim that these settings are optimal for a recurrent model:

| Setting | Main value |
|---|---|
| Global batch | 128 windows |
| Robot-training duration | 10 effective data passes |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Betas | (0.9, 0.95) |
| Epsilon | 1e-8 |
| Weight decay | 0.01, matching the initial reference policy |
| Warmup | 5% of total optimizer updates |
| Schedule | Cosine to 1% of initial LR |
| Gradient clipping | Global norm 1.0 |
| Mixed precision | BF16 compute; FP32 loss reductions and optimizer/master-state configuration |
| Frozen modules | VAE and text encoder |
| Trainable modules | Both complete compact experts, all heads/globals, proprio projection |
| Video/action loss weights | 1 / 1, with per-branch mean normalization |
| EMA | Off initially; consistent across matched runs |
| Attention modification | Off in v0/v1; core XSA in v2 |
| Dropout/augmentation | No new augmentation or dropout beyond the retained data/code path |
| Initial training compilation | Off |
| Teacher forward | None |

The source task configuration uses ten epochs and LR1e-4, while the paper describes a 20k-step setting. Here the operative budget is explicitly **ten passes over the actual training-window manifest**. Do not specify both ten epochs and an unrelated fixed 20k/60k update count and allow a hidden stopping rule to choose between them. [S15, S16]

If \(N\) is the number of training windows and global batch is 128, the approximate planned updates are:

\[
S_{\mathrm{train}}=10\left\lceil N/128\right\rceil.
\]

Use the actual distributed sampler and gradient-accumulation semantics to compute the final integer count. Log windows seen, optimizer updates, and effective passes. A second, matched longer-budget run is a learning-curve experiment, not an undocumented extension only for LoopWAM.

### 13.3 Hardware starting point

Use **four H100 80GB GPUs** as a practical planning configuration, with microbatch two per GPU and accumulation sixteen:

\[
4\times2\times16=128.
\]

On eight GPUs, microbatch two and accumulation eight give the same global batch. On fewer GPUs, increase accumulation. These are resource proposals, **not verified memory-fit or throughput measurements**.

Start with ZeRO-1 and no activation checkpointing. The model is small enough that profiling this straightforward path is worthwhile before writing extra memory-saving machinery. If it does not fit, first reduce the microbatch while preserving global batch; then consider sampled deep supervision or a verified checkpointed block wrapper.

Report the actual optimizer/master-weight dtype. “BF16 training” does not by itself specify every resident tensor's precision or memory cost.

### 13.4 Caching storage estimate

One BF16 `[16,3,28,56]` clip latent is 150,528 bytes before metadata/container overhead. For a hypothetical 100,000-window manifest, that is approximately 15.05GB of raw latent payload. This is a calculation, not a claim about the actual LIBERO window count.

Cache a text embedding once per distinct instruction, not once per overlapping training window. Avoid many tiny files if I/O becomes the bottleneck; a sharded cache can preserve the exact same sample keys and semantics.

### 13.5 Monitoring

Record per-exit action/video losses, their weighted values, global and per-expert gradient norms, core-state norms by loop, attention-update norms, learning rate, steps/windows processed, GPU peak allocated/reserved memory, and step throughput.

Track open-loop action prediction error on held-out demonstrations using a fixed small set of action-sampler seeds. It is a diagnostic, not a substitute for closed-loop success.

Do not require a batch with freshly resampled noise each step to overfit to zero. For a deterministic overfit unit test, freeze the examples, noise tensors, and timesteps; for normal stochastic FM training, nonzero residual error is not automatically a bug.

---

## 14. Implementation map: files and interfaces

### 14.1 Changes to the existing code

| File or component | Required change | What must stay unchanged initially |
|---|---|---|
| `helpers/loader.py` | Support native Wan2.1 source loading; separate source and target configs | Correct source state-dict conversion and VAE statistics |
| VAE loading and state-dict converters | Port/reference Wan2.1 path from Light-WAM | Native latent representation; no cross-family VAE swap |
| `scripts/preprocess_action_dit_backbone.py` | Generalize source loader and target dims; support 512-wide action donors | Documented interpolation/scaling convention |
| New `loopwam_init.py` | FFN pruning, donor selection, canonical initialization artifact, strict key map | No robot teacher, no hidden distillation |
| `wan_video_dit.py` / expert containers | Represent prelude/core/coda modules; expose virtual schedule | Inherited attention, norms, RoPE, time conditioning, head semantics |
| `action_dit.py` | Use width512 and corresponding loop container | 32 action tokens, generative linear velocity head |
| `mot.py` | Schedule-driven joint block application; intermediate exits; XSA hook | F/U/A masks and two expert parameter families |
| `fastwam.py` | Shared noisy-input preparation, exit-loss reduction, K-aware inference/cache | Data conditioning and scheduler convention |
| `trainer.py` | Correct trainable-parameter traversal, version config, logging/checkpoint metadata | Comparable optimizer/data-budget protocol |
| Evaluation loader | Reconstruct target architecture from checkpoint metadata, not native Wan preset | Action normalization, camera order, gripper handling |

### 14.2 Avoid the most dangerous MoT editing mistake

FastWAM's MoT path performs parts of a block itself rather than always calling `DiTBlock.forward`. It uses helpers for Q/K/V construction, modulation splitting, and post-attention updates. [S09]

Therefore, adding XSA only to `SelfAttention.forward` or adding loop support only to `DiTBlock.forward` is insufficient. Inspect and update all actual paths:

```text
forward_joint_core
prefill_video_cache_tensor
forward_action_with_video_cache_tensor
legacy/reference forward paths retained for tests
```

Prefer one canonical `apply_joint_pair` / `apply_post_attention` implementation shared by eager training and inference wrappers. Duplicate code paths easily diverge in normalization, masks, or XSA placement.

### 14.3 Do not overload `num_layers`

Store separate metadata:

```text
unique_depth = 12
pre_depth = 3
core_depth = 6
post_depth = 3
trained_max_loops = 4
effective_depth(K) = 6 + 6*K
```

A stock loop such as `for layer_idx in range(len(expert.blocks))` will execute only twelve layers if the container holds twelve physical blocks. Conversely, constructing thirty physical modules and merely calling some repeatedly fails the parameter-efficiency goal.

Use an explicit virtual schedule whose entries identify stage, physical block, and loop index. In v0–v2 the two experts have identical schedules; v3 introduces separate schedules and cache alignment.

### 14.4 Suggested internal interfaces

The following names describe **new code to implement**, not existing upstream APIs:

```python
init_artifact = build_wan_initialized_donors(source_cfg, target_cfg)
model = build_loopwam(target_cfg, init_artifact)
noisy = model.prepare_training_batch(batch, rng)
outputs = model.forward_exits(noisy, loops=4, exits=(4,))          # v0
outputs = model.forward_exits(noisy, loops=4, exits=(1,2,3,4))    # v1/v2
loss, logs = model.reduce_flow_losses(outputs, noisy, exit_weights)
cache = model.prefill_observation(obs, text_context, proprio, kv=4)
actions = model.sample_actions(cache, ka=4, steps=10, seed=seed)
```

The batch object must contain the actual noisy tensors, clean targets/flow targets, per-branch timesteps, masks, and context. RNG seeds alone are not enough to guarantee equivalent noise across different forward implementations.

### 14.5 Joint forward sketch

```python
# Design pseudocode. prepare_pair/apply_pair/decode_pair are new interfaces.
def forward_exits(noisy, K, selected_exits):
    video, action, conditioning = prepare_pair(noisy)
    for j in range(3):
        video, action = apply_pair(pre_v[j], pre_a[j], video, action, conditioning)

    result = {}
    for k in range(1, K + 1):
        for j in range(6):
            video, action = apply_pair(
                core_v[j], core_a[j], video, action, conditioning,
                use_xsa=(version == "v2")
            )
        if k in selected_exits:
            # Decode a branch; do not replace the recurrent state with its output.
            result[k] = decode_pair(video, action, conditioning)
    return result
```

`decode_pair` must not mutate the saved recurrent tensors in place. The coda is one shared module group, not a different group for each exit.

### 14.6 Trainable parameters and optimizer construction

The model root must register both expert globals, every unique Transformer block, action encoder/head, video head, and proprio projection exactly once for optimization purposes. Shared module aliases can appear in multiple Python attributes, so deduplicate parameter identities when combining explicit lists.

At K=1, pure sharing still executes every physical prelude/core/coda module. It does not leave unused per-loop slots because no such slots exist. Do not automatically enable DDP's unused-parameter traversal merely because K can be smaller than four.

The no-video-loss ablation has genuinely unused world-only output parameters; handle that explicitly rather than conflating it with loop elasticity.

### 14.7 Native loading versus student loading

Use the native Wan loader only to create the initialization artifact or load the frozen VAE/T5 assets. A saved LoopWAM checkpoint must reconstruct its declared compact architecture directly. Do not route a 12-unique-block checkpoint through a native preset that insists on thirty independent blocks and FFN8960.

The evaluation config's `skip_dit_load_from_pretrain` behavior must mean “load the trained LoopWAM weights into the LoopWAM structure,” not “instantiate the wrong native model and ignore missing keys.”

### 14.8 Commands and configuration status

Repository preparation and source download use the ordinary tools for those repositories. Example preparation commands are (HF CLI syntax: [S22]):

```bash
git clone https://github.com/yuantianyuan01/FastWAM.git LoopWAM
cd LoopWAM
git checkout 7faa71108368fbb3b6885649f112af607427a2d4
git switch -c loopwam-s

hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B
```

Use the environment pinned by the chosen repository and record the installed package versions. The proposed `LoopWAM_S_Config_Spec.yaml` is a **configuration contract for the new implementation**, not a configuration that stock FastWAM can already run. Commands such as `task=loopwam_s_v0` will become valid only after the corresponding builders and task configs are implemented.

---

## 15. Required tests before substantive training

The tests below are acceptance criteria for the implementation. They have not been executed against a new LoopWAM model in this report.

| Test | Required property |
|---|---|
| Native source shape validation | Width1536, FFN8960, 12×128 attention, 30 layers, 16 latent channels match the selected asset |
| Target shape validation | Video1536/6144; action512/2048; twelve unique blocks per expert; correct heads and globals |
| Parameter count | Deduplicated count is consistent with ~584.5M; explain any extra module |
| Core storage sharing | Each physical core parameter has one storage regardless of K; checkpoint size does not scale with K |
| FFN selection test | Converted video FFN equals the native FFN with omitted intermediate neurons zeroed |
| Initialization key coverage | Every loaded/new tensor has an explicit source/transformation rule; strict load succeeds |
| K=1 dense equivalence | LoopWAM K=1 equals Dense-S12 built from the same exact initialization |
| Expanded K=4 equivalence | Shared core equals an explicit thirty-application reference at fixed weights |
| Shared-gradient equivalence | Gradient of a shared parameter equals the sum of corresponding untied-copy gradients for the same scalar loss |
| Coda isolation | Decoding an early exit does not modify later core states or the final-exit prediction |
| Exit/truncation equivalence | Exit-k output from the full pass equals a direct k-loop pass, with deterministic conditions |
| VAE anchor equivalence | Current-frame latent matches between reset clip encoding and standalone current-image encoding |
| First-frame causality | Changing future latent tokens leaves first-frame representations/K/V unchanged |
| Policy leakage test | With noisy action input fixed, changing future targets or unused clean labels does not change action prediction |
| Joint/cache output equivalence | Joint action prediction matches video-prefill plus cached-action prediction |
| Joint/cache gradient equivalence | Differentiable factorization preserves gradients, including action-loss gradients into video parameters |
| Virtual cache distinction | Repeated core weights produce separate cache entries for different loop states |
| XSA reference | Projection matches an FP32 mathematical reference, including near-zero values |
| XSA action alignment | Cached action XSA uses action-own V, not observation V or a wrongly indexed concatenation |
| Scheduler target/sign | At fixed tensors, target is noise−data and negative sigma deltas move the sample toward data |
| Mask/padding reduction | Loss normalization excludes padded entries and the clean video anchor |
| Save/load equivalence | Same predictions at every supported K after strict round-trip reload |
| Multi-GPU smoke | No hangs; intended parameters get finite gradients; accumulation/global batch are correct |

Choose numerical tolerances appropriate to the operation and dtype. Do not demand bitwise equality from different fused kernels. Run algebraic tests in FP32 first, then validate BF16 errors relative to the reference output scale.

For causality tests, do not accidentally change the noisy action input while changing its clean label: the denoiser legitimately depends on its noisy input. Hold that input fixed to test whether an illicit clean-label pathway exists.

A deterministic one-batch overfit test should freeze noise/timesteps and show a strong decrease in loss. It does not establish closed-loop performance, but it is a useful check before a long run.

---

## 16. Minimal experiment matrix and version gates

### 16.1 Start with three trainings, not dozens

| Run | Architecture | Objective | Purpose |
|---|---|---|---|
| D12 | Dense-S12, ~584.5M | Final-only FM | Parameter-matched compact baseline |
| V0 | LoopWAM-S, K=4 | Final-only FM | Pure recurrence |
| V1 | LoopWAM-S, K=4 | Normalized joint deep supervision | Test Looped-DiT's training mechanism |

Evaluate V1 at K=1,2,3,4. Evaluate V0 at those depths as truncation diagnostics, but do not assume it was trained to produce useful shallow exits.

If these are numerically sound and V1 is competitive, train V2 with core XSA. The four principal method rows become D12, V0, V1, and V2. Keep their initialization, data exposure, preprocessing, and sampler identical.

### 16.2 Essential compute control

Add Dense-S30 when establishing the final architectural claim. It is larger in storage but has the same effective depth and widths as LoopWAM K=4. For a deep-supervised compute control, organize its distinct middle blocks into four six-block segments and branch through a shared coda at the same exit locations.

The no-XSA deep-supervised Dense-S30 versus V1 comparison matches the 39 block-pair training applications. For V2, add the analogous dense-XSA control, with XSA applied at corresponding middle-layer applications.

If compute resources are constrained, Dense-S30 can begin as a short diagnostic rather than a mandatory large training run before any small model. It is still needed, or must be explicitly reported missing, for a strong equal-compute conclusion.

### 16.3 Same initialization is not one ambiguous phrase

For all controls, keep the same source checkpoint and conversion artifact. Additionally distinguish:

- **Native-layer Dense-S30:** all compacted source layers in order;
- **Expanded-copy Dense-S30:** identical initial function to the looped network, then untied;
- **Dense-S12:** same selected donor blocks as LoopWAM K=1.

These controls answer related but different questions. Do not report one as though it establishes all three.

### 16.4 First additional ablations

After the main rows work, prioritize:

| Priority | Change | Question |
|---:|---|---|
| 1 | K=1,2,3,4 of a fixed V1/V2 checkpoint | Is compute elasticity useful? |
| 2 | Video/world loss removed, action objective unchanged | Does world supervision actually help? |
| 3 | Central copy versus cycle mean | Is the conclusion fragile to shared-core initialization? |
| 4 | FFN6144 versus native FFN8960 | Is the fixed small FFN causing failure? |
| 5 | XSA both/video-only/action-only, as needed | Which modality benefits from update projection? |

Per-loop norm/modulation parameters, KD, rank-32 LoRA, and adaptive routers are not mandatory first ablations. Introducing all of them would obscure the simplest transfer test.

### 16.5 Practical gates

**Infrastructure gate:** shapes, storage sharing, loss reductions, cache semantics, and gradients pass their tests.

**Learning gate:** V0/V1 losses decrease and closed-loop behavior is meaningfully better than a nonlearning/random policy; no claim is based solely on low teacher disagreement or a loss plateau.

**Architecture gate:** V1 or V2 improves on D12 in SR at fixed parameter count, and its trade-off against D30 is understandable. A one-seed one-point difference is preliminary, not a definitive result.

**Extension gate:** begin v3 only after at least one coupled model is reliable at K=4 and preferably has usable smaller-K exits.

Do not declare looping invalid after a single bad FFN-pruned initialization. Conversely, do not add complexity indefinitely when simpler compact dense models dominate measured deployment trade-offs.

---

## 17. Evaluation and efficiency protocol

### 17.1 Closed-loop settings

Use the same pinned LIBERO runner, simulator version, camera convention, instruction strings, initial states, gripper handling, and suite-specific episode caps for all matched models. The inspected evaluation configuration uses thirty initial waiting steps, ten-action replanning, gripper binarization, no action ensembling, and CFG1. [S14]

Set action horizon32 and denoiser evaluations10 explicitly in the new evaluation config. Do not let inherited nulls or unrelated task configs choose them. Read the chosen runner's resolved per-suite maximum steps into the manifest rather than applying a new universal cap silently.

For a first screen, use fifty fixed initial states per task on the ten-task Long suite, giving 500 episodes per configuration. For the final experiment, evaluate all four suites and use multiple independent training seeds. Report paired outcomes on the same initial states, together with uncertainty and per-task/suite results.

Different action-sampler seeds on the same initial states are repeat measurements, not automatically fully independent new task samples. For strong claims, analyze variability across training seeds and tasks as well as episode outcomes.

### 17.2 Three different fairness axes

| Axis | Primary comparison | Interpretation |
|---|---|---|
| Equal stored parameters | D12 versus LoopWAM K=1…4 | What does extra recurrent compute buy without more weights? |
| Equal inference effective depth/width | D30 versus LoopWAM K=4 | Can sharing reduce weights while preserving behavior at comparable arithmetic? |
| Equal measured latency | Best measured operating points across models and K | Does the model improve an actual deployment trade-off? |

Do not describe equal updates as equal training FLOPs. All-exit deep supervision costs more forward/backward work. Report both training windows/updates and total GPU-hours or measured/estimated training FLOPs.

### 17.3 Required metrics

Report task/suite SR, policy parameter count, frozen resident parameters, trainable parameters, checkpoint bytes, loaded bytes, inference FLOPs per action chunk, peak allocated and reserved GPU memory, policy-query throughput, and p50/p95 latency.

Break latency into image preprocessing/transfer, online VAE encoding, text cold start if any, visual prefill, and the action denoising loop. Report warmed fixed-instruction latency separately from full cold-start latency.

Measure batch-one closed-loop deployment first. For throughput measurements at larger batch sizes, label the batch and distinguish independent policy queries from emitted action tokens.

### 17.4 Profiling method

Warm each compiled graph and checkpoint/model configuration. A proposed starting protocol is fifty warm-up queries followed by five hundred timed queries over real benchmark observations. Synchronize GPU timing appropriately. Record GPU model, clock/power settings when available, PyTorch/CUDA/attention backend, precision, and compilation mode.

Use the same hardware and preprocessing for cross-model latency claims. Include both eager and compiled results if compilation changes the ranking. Do not time a cached current-frame latent for LoopWAM while timing online VAE encoding for another model.

### 17.5 FLOPs and token count

Count the actual executed graph. A shared weight used four times contributes four matrix multiplications, not one. Conversely, visual prefill performed once per chunk must not be multiplied by the number of action denoiser evaluations.

The useful structural expression is:

\[
F_{\mathrm{chunk}}=F_{\mathrm{VAE}}+F_v(K_v)
+S_{\mathrm{denoise}}F_a(K_a;N_{\mathrm{obs}})
+F_{\mathrm{other}}.
\]

State whether F includes attention masking inefficiency, text processing, projection/KV caching, and the VAE. The Wan1.3B native grid has 392 observation tokens; it is incorrect to reuse a Wan2.2-based 98-token FLOP estimate.

### 17.6 Efficient-WAM, Light-WAM, and original Fast-WAM

Efficient-WAM is a required compact generative-WAM reference; Light-WAM is a required efficient world-supervised policy reference. Their architectures and deployment paths differ from the proposed recurrent MoT. [S20, S21]

Do not compare Light-WAM's reported trainable parameter count with LoopWAM's total trainable policy and call that an apples-to-apples size comparison. Count total loaded model components separately from trainable ones.

An Efficient-WAM LIBERO port must be identified as a port unless its official release provides the exact tested setting. Match data/actions/environment and report initialization/pretraining differences. Do not reuse its Wan2.2 latent caches for the Wan2.1 model.

The released large Fast-WAM policy can be evaluated as an external anchor without making it a teacher. Its result is not a controlled architecture comparison because its foundation model, VAE grid, widths, and training history differ. D12/D30 are the core attribution controls.

### 17.7 Beyond standard LIBERO

Once a promising model exists, add an unselected robustness benchmark and a broader manipulation setting, such as an appropriate LIBERO distribution-shift benchmark and RoboTwin. Verify their current protocols when implementing those extensions.

For real-time claims, add delay-injected evaluation where policy computation has consequences for execution timing. A simulator that pauses while an action chunk is computed cannot by itself establish real-time robustness.

---

## 18. Failure analysis and bounded fallbacks

| Observation | Plausible explanation | First diagnostic |
|---|---|---|
| D12, D30, and LoopWAM all perform poorly | Data, source conversion, action normalization, or training budget | Validate native loading, source/target maps, actions, and native-FFN diagnostic |
| D30 works but V0 is weak | Sharing/initialization/optimization difficulty | V1, core donor alternative, tied-versus-untied diagnostic |
| V1 works at K4 but not smaller K | Inadequate early-exit learning or conflicting objectives | Per-exit losses, normalized weights, correct coda branching |
| XSA helps video loss but hurts SR | Action denoising or cross-stream evidence is harmed | Video-only/action-only XSA; ensure own-V indexing is correct |
| Low FM loss but weak closed-loop behavior | Distribution shift, action convention, horizon/replanning errors | Rollout inspection, denormalization, gripper, action-timestamp checks |
| No measured speed benefit at K4 | Same executed depth; parameter reduction is not arithmetic reduction | K1–K3 operating points, component timings |
| Wan1.3B model slower than expected | Fourfold visual-token count relative to Wan2.2 grid | Profile prefill and online VAE separately |
| Smaller model fails but FFN8960 works | FFN compression is too aggressive under this recipe | Keep ~0.688B diagnostic result; test better FFN selection only afterward |
| Asymmetric (4,1) underperforms (1,1) | Conditioning-depth mismatch or ineffective deeper visual work | Matched continuation and alignment control, not immediate new modules |

Do not treat a 1B model rescuing performance as proof that capacity alone caused failure. Initialization, optimization, and tokenization can interact with size. A targeted diagnostic is stronger than jumping to a different scale and drawing a single-cause conclusion.

Teacher distillation is a later recovery option if the architecture is worthwhile but direct training remains weak. If added, apply the same noise/input contract to teacher and student, track teacher cost, and retain expert-data supervision. It is not part of v0–v2.

---

## 19. Research claim and realistic publication bar

The following mechanisms are already supplied by Looped-DiT and related shared-depth work: repeated middle blocks, shared exit decoding, intermediate supervision, and XSA. Applying them to a WAM is a valuable baseline and empirical study, but does not automatically establish a strong new architecture contribution. [S17–S19]

The first scientific result to seek is:

> At the same ~0.6B policy size, recurrent depth improves manipulation compared with a shallow dense WAM; with appropriate loop supervision, one model provides useful control across several compute budgets.

A stronger WAM-specific result is:

> Video and action branches exhibit different useful depth budgets, and a trained cross-depth interface improves success at a measured deployment cost without a second model or an online teacher.

That is the motivation for v3, but it must be demonstrated. Deeper visual processing need not be useful for every task or action horizon.

A credible final package should show parameter-matched and compute-matched controls, improvements that survive independent training seeds, meaningful lower-K operating points, and benefits beyond one saturated standard-LIBERO mean. World-supervision on/off and dense-XSA controls help establish where gains come from.

No precise SR threshold guarantees a CVPR acceptance. A near-equal-SR compact model may be useful engineering; a compelling research paper additionally needs evidence that the recurrent mechanism solves a specific problem better than simpler shallow or narrow dense alternatives.

---

## 20. First implementation checklist

The first engineering milestone is deliberately limited:

**Build and validate the Wan1.3B source loader.** Confirm source tensor shapes and native VAE anchoring. Separate loading from compact-model construction.

**Build one canonical compact donor artifact.** Select video FFN channels; derive action weights from native Wan; record new heads, metadata, and hashes.

**Implement D12 and v0.** Make K=1 equality and explicit K=4 unrolling tests pass. Keep only final-exit FM and ordinary attention.

**Implement v1.** Add shared-coda branches and normalized final-plus-mean losses on both outputs. Confirm early-exit decoding does not change later recurrence.

**Train D12, v0, and v1 under the same declared budget.** Establish basic action learning and measure the K curve.

**Only then implement XSA and v3.** Do not begin by adding every extension from the earlier 1B compression plan.

The complete initial design is:

\[
\boxed{
\begin{gathered}
\text{Wan2.1-T2V-1.3B initialization}\
\Downarrow\
\text{video }1536/6144\; +\;\text{action }512/2048\
\Downarrow\
(3\text{ pre},6\text{ shared},3\text{ post}),\ K=4\
\Downarrow\
\text{ordinary FM}\ \to\ \text{joint deep supervision}\ \to\ \text{core XSA}.
\end{gathered}
}
\]

---

## 21. Source register and audit trail

The links below are primary model assets, official source repositories, or papers. `[Sxx]` references in the report identify their role. Source-code facts are scoped to the indicated revisions, not assumed to describe all versions of a project.

**S01 — Native Wan1.3B model configuration.** Official checkpoint configuration; establishes width1536, FFN8960, 12 heads, 30 layers, and 16 channels.  
https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/blob/main/config.json

**S02 — Native Wan5B model configuration.** Official checkpoint configuration; establishes width3072, FFN14336, 24 heads, 30 layers, and 48 channels.  
https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/main/config.json

**S03 — Wan1.3B VAE stride and architecture configuration.** Official code; reviewed blob `ea9502b0df685b5d22f9091cc8cdf5c6a7880c4b`.  
https://github.com/Wan-Video/Wan2.1/blob/main/wan/configs/wan_t2v_1_3B.py

**S04 — Wan2.2-TI2V-5B model card.** Official documentation of the high-compression VAE.  
https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B

**S05 — FastWAM model config at the selected revision.** Includes source Wan model, action checkpoint policy, attention dimensions, and scheduler shifts.  
https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/model/fastwam.yaml

**S06 — Light-WAM family loader.** Wan2.1 loader/preset reference; the default policy architecture itself is not copied. Reviewed loader blob `b315d7e0a49a1647414115ee3e6c489925c7cef5`.  
https://github.com/L1ziang/Light-WAM/blob/db7d05e724f3eac2d4f0c9c3ac727fe93a44d727/src/lightwam/models/wan22/helpers/loader.py

**S07 — FastWAM video DiT and block definitions.** Reviewed blob `c1ab84700f72ab3de8916cfc4d30d9b531def171`.  
https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/wan_video_dit.py

**S08 — FastWAM action DiT.** Action input/output and backbone structure; inspected during this conversation.  
https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/action_dit.py

**S09 — FastWAM MoT execution and caches.** Joint attention, direct block-internal calls, cached action inference, and checkpointing restrictions; inspected during this conversation.  
https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/mot.py

**S10 — FastWAM input construction, masks, and losses.** Reviewed blob `f312ea04a5ec0cff9286263328668752a3b61b58`.  
https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py

**S11 — FastWAM action-backbone preprocessor.** Official interpolation/scaling reference; reviewed blob `737f9c05779518d976550727251fa0fa9a1ecd48`.  
https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/scripts/preprocess_action_dit_backbone.py

**S12 — FastWAM continuous scheduler.** Noise construction, shifted sampling, training weights, and Euler delta direction; reviewed blob `9028f6c86e2e35bc9efd75dda26d9cf88ff898bb`.  
https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/schedulers/scheduler_continuous.py

**S13 — FastWAM LIBERO data config.** Cameras, sampling, action/state dimensions, and normalization.  
https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/data/libero_2cam.yaml

**S14 — FastWAM LIBERO evaluation config and runner.** Replanning, waiting, gripper processing, loading, and preprocessing.  
https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/sim_libero.yaml  
https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/libero/eval_libero_single.py

**S15 — FastWAM task/trainer recipe.** Ten-epoch task configuration and optimizer/scheduler implementation, inspected during this conversation.  
https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/task/libero_uncond_2cam224_1e-4.yaml  
https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/trainer.py

**S16 — Fast-WAM paper.** World/action objectives and the broader experimental framing; paper/code schedule descriptions are distinguished in the report.  
https://arxiv.org/html/2603.16666v1

**S17 — Looped Diffusion Transformer paper.** Architectural and conceptual reference, not evidence of robotic success.  
https://arxiv.org/html/2609.40305v1

**S18 — Looped-DiT model code.** Pre/core/post organization and XSA; reviewed model blob `5048f45230a51853ba1974c4d8fe3acc5c735b74`.  
https://github.com/OpenSenseNova/Looped-DiT/blob/65a7705ad2954fa127721bd1131ea8f865688848/looped_dit/model.py

**S19 — Looped-DiT objective code.** Exact final-plus-mean weights and pixel-space scheduler; reviewed blob `c9bb03b50b0f4533e86cad7139a1c179381c4043`.  
https://github.com/OpenSenseNova/Looped-DiT/blob/65a7705ad2954fa127721bd1131ea8f865688848/looped_dit/diffusion.py

**S20 — Light-WAM paper.** Comparator and distinction between trainable and total deployed parameters.  
https://arxiv.org/abs/2606.08242

**S21 — Efficient-WAM paper.** Compact generative-WAM comparator.  
https://arxiv.org/abs/2606.10040

**S22 — Hugging Face CLI documentation.** Download syntax and explicit revision/local-directory options.  
https://huggingface.co/docs/huggingface_hub/en/guides/cli

**S23 — PyTorch checkpoint documentation.** Explicit checkpoint variant and non-reentrant autograd behavior. Match API details to the installed pinned PyTorch release.  
https://docs.pytorch.org/docs/stable/checkpoint.html

---

## 22. What is delivered and what remains to execute

This report delivers a complete **design contract**: primary foundation choice, target dimensions and analytical size, initialization mapping, two-stream recurrence, masks, objectives, version sequence, cache semantics, code modification map, training/evaluation protocols, and acceptance tests.

The implementation, actual parameter census, GPU fit, speed, and manipulation performance still need to be executed and measured. Do not quote the proposed target architecture's analytical counts as an experimentally validated system result, and do not mark the test table passed until the new code has been run.
