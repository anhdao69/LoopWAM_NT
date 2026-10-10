# ChronoLoop: implementation, validation and launch (October 10, 2026)

Branch `chrono` (git worktree `/lustre/fs1/home/an221229/code/LoopWAM_chrono`, based on `3eac53a`).
Production code snapshot: `97d55b1` at `/groups/yshang/an221229/checkpoints/ChronoLoop/code/97d55b1cfc09`.
Run root: `/groups/yshang/an221229/checkpoints/ChronoLoop/runs/<experiment>`.
Hugging Face: [anhdao69/ChronoLoop](https://huggingface.co/anhdao69/ChronoLoop) (public).

**Status at writing (13:30 EDT): no ChronoLoop run has finished training, and no run has been evaluated.**
CL-0 and CL-A are training in the interactive allocation (job 894901). The other five runs, and the
continuations of CL-0 and CL-A, are queued as 4 x H100 Slurm jobs. Use `python scripts/chronoloop_status.py`
for the live table (section 9).

## 1. What the audit established

- **Parent.** All runs fine-tune the full-suite LoopWAM v0 4/4 checkpoint, downloaded from
  `anhdao69/LoopWAM_NT/libero-all-suites/v0-video4-action4/policy.pt`. Its SHA-256 is `1c659a7c…602b`,
  which matches the release export record. It has 584,536,135 parameters, 21,700 updates (10 epochs) and
  scored 388/400 on the matched four-suite evaluation.
- **Data.** Full LIBERO: four suites, 1,712 demonstrations, 277,713 windows. Each window is 9 frames at
  stride 4 plus a 32-action chunk. The normalization hash `d5ad351e…` is identical to the parent's.
- **Cluster.** Interactive job 894901 runs on evc103: 4 x H100 80GB, 16 CPUs, ending 2026-10-11 04:53 EDT.
  All 32 highgpu GPUs are allocated and the queue is long. Your pending interactive jobs 894902 and 894903
  were not touched.
- **Earlier H100 run.** The earlier full-LIBERO v0 run on this cluster (2.56 s/update, 4 GPUs, cold
  cache) had stopped at update 44. Only 1,280 latent-cache entries existed, so a fresh shared cache was built.

## 2. Architecture, as implemented (plan section 2)

The model code is `src/fastwam/models/wan22/chronoloop.py`, with `ChronoLoopMoT` and `ChronoLoopWAM`.

- **Prefix tokens.** P (16 memory tokens, or 392 history tokens for CL-FRAME) is prepended to the video
  stream after `video_expert.prepare`, giving `[P | obs | fut]`. P uses clean (t = 0) modulation, the
  observation's text-context mask, and identity RoPE (unit complex numbers; this is not `freqs[-1]`).
  Video frame indices are unchanged.
- **Attention (G-a).**
  - P queries attend over `[P, obs]` with no gate.
  - Obs, future-video and action queries add `tanh(alpha_h) · Attn_h(q, K_P, V_P)` to their parent F/U/A
    attention. All of these memory reads happen in one mask-free SDPA call.
  - There is one alpha per head per physical block per expert: 2 x 12 x 12 = 288 scalars, shared across
    loops, initialized to 0.
  - The parent's three structured SDPA calls are kept unchanged. P is stripped before `video_expert.post`,
    so the video loss stays on future frames only.
- **Readout and update (U1).**
  - The readout `h` is the P state after the last core loop, before the coda.
  - The update is `w = c·tanh(LN(h)/c)` with c = 3, then `s_k = λ⊙s_{k−1} + (1−λ)⊙w` with
    λ = σ(a), a initialized to 2.2.
  - The next query's memory tokens are `m = γ⊙s + e`.
  - γ is initialized to the per-channel RMS of the obs tokens after `prepare`, divided by c.
    e is drawn from N(0, std(obs)²). Both statistics come from 32 fixed windows.
  - The memory has 31,008 parameters: e 24,576 + γ 1,536 + a 1,536 + LN 3,072 + alpha 288.
- **Functional state.** Callers pass `s_{k−1}` and receive `(loss, logs, s_k)`. Nothing is stored on the
  module inside checkpointed regions.
- **Variants.**
  - **CL-REG:** s = 0 at every query.
  - **CL-W2:** memory tokens are read-only inside the backbone; they are not updated by the blocks, but
    their K/V are recomputed at every layer. The update comes from one Wan block (a deepcopy of physical
    block 3, the first core block) with query `γ⊙s + e` and keys `[itself; obs after the last core loop]`.
    Its output goes through the same squash and λ-mix. This adds 37.8M parameters.
  - **K_a = 1:** the existing late alignment (action reads video loop 4) is used, with the 4/4 parent
    weights loaded.
  - **CL-FRAME:** the clean frame from query k−3 is taken as the first latent frame of the cached clip
    at window τ − 10·min(k, 3), i.e. the traversal's first query when k < 3. It sits at temporal RoPE
    position −2 (the complex conjugate of position 2). It is read through the same G-a gate.
  - **CL-ORACLE:** only `mem_source=oracle` exists, a label embedding replacing s. Stage labels are not
    produced; see section 10.
- **Inference.** `chrono_infer_action` prefills `[P | obs]` and caches obs and P K/V separately per virtual
  layer. Actions are then denoised reading both caches, the second through the gate. The call returns
  `{'action', 'state'}`, and the caller owns the state.
- **Evaluator.** `scripts/evaluate_chronoloop_libero.py` uses the matched harness: 700 steps, 30 settling
  steps, replan every 10 actions, 10 denoising steps, and memory reset at episode start. It supports
  `--noise fixed|fresh` and the CL-FRAME history buffer.

## 3. Training setup (plan section 3) and deviations

| Item | Implemented |
|---|---|
| Init | Parent v0 4/4 for every run; fresh AdamW |
| Batch | 128 window slots per update = 32 streams x 4 consecutive windows (TBPTT S = 4). Rank r owns streams r, r+R, … |
| Schedule | One shared file, `shared/schedule_e10_s42_n32_t4.json`, SHA-256 `599fb41a…0834`: 10 epochs x 17,120 traversals (demo, phase u ∈ 0..9), shuffled with seed + epoch |
| Budget | **10 epochs**, which is 23,862 updates and 2,777,130 valid windows, the same data exposure as the parent's 10 epochs. Slot utilization is 90.9% |
| Loss | Parent v0 flow loss (video + action, the parent's weights and shifts), averaged over the valid windows of each update |
| Optimizer | AdamW (0.9, 0.95), eps 1e-8, clip 1.0. Backbone LR 3e-5 with weight decay 0.01; memory LR 3e-4 with weight decay 0. Warmup 3% (715 updates), then cosine to 1% of peak (the parent's shape) |
| Precision | FP32 master weights and Adam moments, BF16 autocast, per-block activation checkpointing |
| EMA | None, matching the parent |
| Checkpoints | `checkpoints/epoch_08.pt`, `epoch_09.pt`, `epoch_10.pt` (weights only, saved when every window of that epoch has trained, at updates 19,091, 21,475 and 23,862). `update_002000.pt` for the plan's early read. `latest.pt` (resumable, 7 GB, about 19 s to save) every 500 updates and at any planned stop |

Deviations, with the reason for each:

1. **10 epochs instead of 6,000 updates.** This was your explicit requirement. The plan's final checkpoint
   at update 6,000 is not kept separately; epochs 8, 9 and 10 are.
2. **Noise and timesteps are fixed per window slot**, seeded by hash(seed, update, stream, window).
   Every run therefore sees identical data and identical noise for paired windows (variance reduction),
   and the noise does not depend on the GPU layout or on resuming.
3. **Masked slots are dropped, not computed.** The loss and its denominator use only valid windows, as the
   plan requires; this also saves about 9% of compute.
4. **Stateless runs** (CL-0, CL-0@1, CL-REG, CL-FRAME) put the 4 windows of a segment into one batch.
   Without carried state this is mathematically the same and runs faster. Stateful runs (CL-A, CL-A@1,
   CL-W2) run the 4 windows in sequence with gradient through s, and detach s at segment ends.
5. **`torch.compile` of each block function**, about 1.8 to 2.2x faster.
   - The math is unchanged. Against an FP32 eager reference, compiled BF16 gradients have 0.95% global
     relative error versus 0.99% for eager BF16.
   - `comprehensive_padding` is disabled. With it on, inductor 2.7 failed checkpoint recomputation when
     the batch size changes inside one TBPTT graph; this was reproduced and fixed.
6. **Decode-free data.** Training reads frozen VAE latents from a fully precomputed shared cache, and
   non-video fields come from in-memory parquet columns.
   - Field-level equality with the parent dataset is tested.
   - The cache uses the parent's provenance, with 277,713 entries built in 39 minutes on 4 GPUs. 192
     sampled clips were checked bit-exact against the parent dataset before encoding.
   - VAE latents depend slightly on batch size (up to 0.031 between batch 1 and batch 8). The parent's own
     cache has the same property; every ChronoLoop run reads the same cached values.
7. **CL-FRAME details** not given by the plan: RoPE position −2, gated reading (so it is the exact parent
   at init, the same mechanism as CL-A), and the first query repeated when k < 3.
8. **Run order.** The plan gates round 3 (CL-0@1, CL-A@1) on Gate 1. You asked for every run to be
   submitted, so they are queued now. If Gate 1 fails, cancel them with `scancel 897640 897641`.

## 4. Tests

| Test (plan 3.3) | Result |
|---|---|
| Memory disabled = parent path, 584,536,135 params | pass (CL-0 has no memory module) |
| Init equivalence, alpha = 0 | pass. Outputs are exactly invariant to the memory or history content for CL-A, W2, REG and FRAME, and match the parent within 8.7e-4 relative (GEMM shapes differ by 16 rows) |
| No leakage | pass. Changing future frames, noisy actions or timesteps leaves s_k bit-identical |
| Train ↔ inference over 5 queries | pass. Stateful prefill s_k matches the training path (bf16 tolerance) |
| Bounded state | pass: \|s\| ≤ c after 1,000 updates of 1e4-scale input |
| Temporal credit (alpha = 0.5) | pass. The window-4 loss gives non-zero gradient to s at windows 1–3; it is exactly 0 when s is detached each query; the core still gets gradient |
| Recompute safety | pass, within the BF16 run-to-run noise floor (attention-backward atomics are not bitwise reproducible) |
| Sampler | pass. Each (demo, window) appears exactly once per epoch in 2 epochs; suite shares are exact; consecutive windows are 10 frames apart; scheduler resume is exact |
| Layouts | pass. 2-GPU vs 4-GPU update-1 loss: CL-0 loss_video 0.06119899 vs 0.06119899, grad norm 0.15409 vs 0.15406; CL-A grad norm 0.15482 vs 0.15464 |
| Resume | pass. CL-A trained to update 10 on 2 GPUs and resumed on 4 GPUs reproduces updates 11–20: identical valid windows, memory RMS equal to 4 digits, losses within 3e-3 relative (the 2-vs-4-GPU noise floor is up to 2.4e-3) |
| Compiled vs eager | pass (see deviation 5) |
| Checkpoint guard | pass. Loading a checkpoint with different memory flags is refused |
| All variants through the trainer | pass, 3-update smokes of CL-REG, CL-W2, CL-A@1, CL-FRAME and CL-0@1 |
| CUDA graph, 70 queries | **not done**: the ChronoLoop inference path is eager |

Commands: `CHRONO_PARENT=… python -m pytest tests/test_chronoloop.py tests/test_chronoloop_data.py`.

- On an idle GPU, all 12 original model tests and the 4 data tests passed.
- The final full re-run (13 tests, including the compile test) was done on GPU 3 while CL-A was training
  there. 11 passed. `test_temporal_credit` and `test_recompute_safety` ran out of memory on the shared GPU;
  both had passed earlier on an idle GPU, and their code did not change.

Expected K_a = 1 behaviour: removing three action loops from the 4/4 parent raises the initial action loss
from 0.017 to 0.336. It falls to 0.236 by update 3 and is learned back during fine-tuning.

## 5. GPU scheduling

Benchmark with `scripts/chronoloop_benchmark.sh`, steady state over updates 11–30, compiled, warm cache:

| Layout | Run | s/update | windows/s | Peak GB |
|---|---|---|---|---|
| 2 x 2 (concurrent) | CL-0 | 2.20 | 52.9 | 29.7 |
| 2 x 2 (concurrent) | CL-A | 2.80 | 41.6 | 24.8 |
| 1 x 4 | CL-0 | 1.33 | 87.8 | 20.0 |
| 1 x 4 | CL-A | 1.80 | 64.7 | 18.2 |

- Two concurrent 2-GPU runs deliver 94.5 windows/s in total, more than any single 4-GPU run. Larger
  per-GPU batches and fewer DDP participants are the likely reasons.
- Benchmark projections: running CL-0 then CL-A sequentially on 4 GPUs takes about 21 h; running them
  concurrently on 2 + 2 GPUs takes about 18.5 h.
- **Decision:** 2 x 2 in the interactive allocation, with CL-0 on GPUs 0,1 and CL-A on GPUs 2,3, each with
  16 streams per rank in one micro-batch. Round 1 then advances together, as the plan recommends, and
  aggregate throughput is highest.
- In production (first 80 updates), CL-0 runs at 2.55 s/update and CL-A at 3.0 s/update. Neither is
  expected to finish all 23,862 updates before the allocation ends; CL-0 should come close.
- Both stop cleanly 15 minutes before the end of the allocation with a resumable checkpoint.
  Continuations 897643 (CL-0) and 897644 (CL-A) are queued with `afterany:894901` and resume on
  4 x H100. Resuming at a different world size was tested.

## 6. Commands

```bash
# one-time (done): shared latent cache and schedule
torchrun --standalone --nproc_per_node=4 scripts/chronoloop_precompute_latents.py --shared-dir $CHRONO_SHARED
# interactive (running), from the frozen snapshot
bash $CODE/scripts/chronoloop_interactive.sh $CODE CL-0 0,1 --stream-micro 16
bash $CODE/scripts/chronoloop_interactive.sh $CODE CL-A 2,3 --stream-micro 16
# Slurm, 4 x H100 per job (submitted)
bash scripts/chronoloop_submit.sh --time 16:00:00 CL-REG CL-W2 CL-0@1 CL-A@1 CL-FRAME
bash scripts/chronoloop_submit.sh --time 06:00:00 --after 894901 CL-0 CL-A
# a single run by hand
torchrun --standalone --nproc_per_node=4 scripts/train_chronoloop.py $(python scripts/chronoloop_experiments.py CL-A) --output-dir DIR
```

Each Slurm job (`scripts/chronoloop_job.sbatch`) does the following:

- Requests partition `highgpu`, `gpu:nvidia_h100_80gb_hbm3:4`, 16 CPUs and 256 GB.
- Resumes from `latest.pt` if present, and stops 15 minutes before its walltime.
- If the run is unfinished, uploads any finished epoch checkpoints and resubmits itself, keeping at most
  one pending continuation. It stops after 3 consecutive training failures.
- When the run is complete, uploads epochs 8, 9 and 10 with retries. If that cannot be verified, it queues
  a CPU-only upload job.
- Refuses to train if another live job's heartbeat owns the run directory, which prevents duplicates.

## 7. Job IDs

| Run | Where | Job | Walltime | Output directory (under `…/ChronoLoop/runs/`) |
|---|---|---|---|---|
| CL-0 | interactive, GPUs 0–1 | 894901 | until 04:38 EDT | `CL-0_no-memory_a4` |
| CL-A | interactive, GPUs 2–3 | 894901 | until 04:38 EDT | `CL-A_mem16-learned-loopwrite_a4` |
| CL-0 (continuation) | Slurm 4 x H100, afterany:894901 | 897643 | 6 h | same |
| CL-A (continuation) | Slurm 4 x H100, afterany:894901 | 897644 | 6 h | same |
| CL-REG | Slurm 4 x H100 | 897638 | 16 h | `CL-REG_mem16-reset-registers_a4` |
| CL-W2 | Slurm 4 x H100 | 897639 | 16 h | `CL-W2_mem16-learned-external-updater_a4` |
| CL-0@1 | Slurm 4 x H100 | 897640 | 16 h | `CL-0-at1_no-memory_a1` |
| CL-A@1 | Slurm 4 x H100 | 897641 | 16 h | `CL-A-at1_mem16-learned-loopwrite_a1` |
| CL-FRAME | Slurm 4 x H100 | 897642 | 16 h | `CL-FRAME_history-frame-k3_a4` |

Projected 4-GPU training time is about 9 h for stateless runs and about 12 h for stateful ones (from the
benchmark), plus queue wait, which is unknown.

## 8. Hugging Face

- **Repository:** `anhdao69/ChronoLoop`, created public as you asked, with a model card. Write access was
  verified with a throwaway 60 MB LFS file: the Hub SHA-256 matched, and the file was deleted afterwards.
- **Layout:** `<experiment folder>/epoch_08|09|10/{policy.pt, config.json}`, plus `manifest.json`,
  `metrics.jsonl` and `timing.json`.
- **Upload script:** `scripts/chronoloop_hf_upload.py` is idempotent. It records a file as uploaded only
  when the Hub's size and LFS SHA-256 match the local file, then writes `hf_upload.json` in the run directory.
- **Credentials:** the token is read from `/groups/yshang/an221229/cache/huggingface/token` and never printed.
- **Current state:** no checkpoint has been uploaded yet, because none exists yet.

## 9. Experiment tracking (`python scripts/chronoloop_status.py`)

| Run | Flags (mem/src/write/K_a/hist) | Status | Update | GPUs | Slurm | Checkpoints | HF verified |
|---|---|---|---|---|---|---|---|
| CL-0 | 0/none/none/4/0 | running (interactive) | 81/23862 | 2 | 897643 pending (dependency) | - | - |
| CL-A | 16/learned/loop/4/0 | running (interactive) | 80/23862 | 2 | 897644 pending (dependency) | - | - |
| CL-REG | 16/reset/loop/4/0 | queued | - | 4 | 897638 pending | - | - |
| CL-W2 | 16/learned/external/4/0 | queued | - | 4 | 897639 pending | - | - |
| CL-0@1 | 0/none/none/1/0 | queued | - | 4 | 897640 pending | - | - |
| CL-A@1 | 16/learned/loop/1/0 | queued | - | 4 | 897641 pending | - | - |
| CL-FRAME | 0/none/none/4/3 | queued | - | 4 | 897642 pending | - | - |
| CL-ORACLE | - | not submitted (conditional) | - | - | - | - | - |

There are no evaluation results yet. Following your instruction, nothing is evaluated on the H100 server.

## 10. Not done, and why

- **CL-ORACLE.** The plan makes it conditional on Gate 1 failing. It also needs privileged stage labels
  from simulator replay of every demonstration (BDDL goal conjuncts). The model hook exists; the label
  pipeline and the trainer wiring do not.
- **CL-TF (eval-only).** Not implemented in the evaluator.
- **CUDA-graph inference test.** Not done; the inference path is eager.
- **Pushing `chrono`.** Not possible from this node: there are no GitHub credentials (HTTPS asks for a
  username and the SSH key is not registered). All commits are local on branch `chrono`. Push from a
  machine with credentials:
  `git -C /lustre/fs1/home/an221229/code/LoopWAM_chrono push -u origin chrono`.

## 11. Evaluating on the other server

```bash
git checkout chrono   # after the push
huggingface-cli download anhdao69/ChronoLoop --include "CL-A_mem16-learned-loopwrite_a4/*" --local-dir hf
torchrun --standalone --nproc_per_node=4 scripts/evaluate_chronoloop_libero.py \
  --checkpoint hf/CL-A_mem16-learned-loopwrite_a4/epoch_10/policy.pt --suite libero_10 \
  --episodes-per-task 50 --seed 42 --noise fixed --data-dir <dir with dataset_stats.json + data_manifest.json> \
  --output-dir eval/CL-A/libero_10_seed42_fixed
```

- `--data-dir` needs the training normalization, whose hash is `d5ad351e…`. It is identical to the
  parent's, so the parent release's `dataset_stats.json` and `data_manifest.json` can be used.
- Run Long first, then the other three suites, with seeds 42, 43 and 44.
- Compute Gate 1 as CL-A − CL-0 on Long with fixed noise.

## 12. Next recommendations

1. Once CL-0 and CL-A finish, evaluate Long with fixed noise for seeds 42/43/44, and evaluate tasks 6 and 8
   separately; this is Gate 1.
2. If Gate 1 passes, keep CL-0@1 and CL-A@1 (already queued) and measure latency for Q3 at batch 1.
   The ChronoLoop inference path still needs a CUDA-graph or compiled variant for a fair latency number.
3. If Gate 1 fails, cancel 897640 and 897641 and build the oracle stage-label replay before running CL-ORACLE.
4. Check the gates in `metrics.jsonl`: `tanh_alpha_*_absmean_per_block` should move away from 0 within the
   first 1,000 updates (the run is flagged otherwise). Also watch `mem_saturated_frac`.
