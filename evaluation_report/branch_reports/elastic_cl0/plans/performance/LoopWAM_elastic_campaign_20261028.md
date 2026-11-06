# LoopWAM Elastic campaign: implementation, validation, training, evaluation and latency (2026-10-28, updated 2026-10-30)

Plan: `plans/LoopWAM_elastic.md`. Branch: `elastic`. Training source commit: `57f36a41a67b29fb1e4fed305d5c9a3b1e2da8b6`.
Evidence referenced below is in `plans/elastic_evidence/`.

**Update 2026-10-30.**
- **Training:** all five runs are complete (21,700 updates, 10 epochs each), and their `epoch_10.pt` uploads are verified.
- **Evaluation:** the matched LIBERO evaluation of 37 conditions is complete, 6,000 episodes each (Section 9).
- **Latency:** every Elastic operating point is timed on H100 together with every other model in the repository (Section 10).

## 1. Status

All five new training runs from plan Section 8 ran in the user's interactive jobs. Each has 2 GPUs per run on 4×H100 80GB nodes: 1080499 (evc102), 1080498 (evc103) and, for part of F0-PB, 1084032 (evc103).

| Run | Transport | Scope | Rollout sampler | Init artifact | Completed (EDT) | `epoch_10.pt` sha256 | Launches (jobs) |
|---|---|---|---|---|---|---|---|
| E0 | T0 | P-c | balanced flat9 | `common_v1` | 2026-10-29 00:30 | `5569c47f…` | 1 (1080499) |
| E2 | T4 | P-c | same flat9 sequence | `common_v1` | 2026-10-29 00:39 | `6d92ac13…` | 1 (1080499) |
| E1 | T1 | P-c | same flat9 sequence | `common_v1` | 2026-10-29 05:05 | `be1cbe2c…` | 2 (1080498, 1080499) |
| SP11 | T0 | P-c | fixed `flat_k1_n2` | `common_v1` | 2026-10-28 23:16 | `ab974f35…` | 1 (1080498) |
| F0-PB | T0 | P-b | historical `legacy_choice` over S1,S2,S5,S6,S7,S8 | `rt_a2_legacy_v1` | 2026-10-29 13:34 | `ff59791a…` | 3 (1080498, 1080499, 1084032) |

Each run moved between jobs only through the resume path, whose bit-identity was validated (Section 5). The launch-time status at the original writing follows.

| Run | Transport | Scope | Rollout sampler | Init artifact | Status at launch |
|---|---|---|---|---|---|
| E0 | T0 | P-c | balanced flat9 | `common_v1` | training, lane A (GPUs 0,1) |
| E2 | T4 | P-c | same flat9 sequence | `common_v1` | training, lane B (GPUs 2,3) |
| E1 | T1 | P-c | same flat9 sequence | `common_v1` | queued |
| SP11 | T0 | P-c | fixed `flat_k1_n2` | `common_v1` | queued |
| F0-PB | T0 | P-b | historical `legacy_choice` over S1,S2,S5,S6,S7,S8 | `rt_a2_legacy_v1` | queued |

The two lanes share one ordered queue (E0, E2, E1, SP11, F0-PB). A per-run `flock` lets whichever lane frees first take the next run.

- Measured throughput at launch: 2.42 s/update for both E0 and E2, averaged over updates 6–74. That is about 14.5 h per run.
- The F0-PB time was not yet measured on this layout.
- The plan's lane table puts F0-PB third on lane A. With a shared queue, F0-PB instead starts as soon as either lane is free, which is no later.
- At launch, the report claimed no training outcome. Sections 9 and 10 report the outcomes.

Checkpoints and records are published to `https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/libero-all-suites/Elastic/<RUN>/`:

- **Checkpoints:** `step_00002000.pt`, `epoch_08.pt`, `epoch_09.pt` and `epoch_10.pt`. The predeclared selection is the raw `epoch_10.pt`.
- **Run metadata:** `config.json`, `schedule_sequence.json`, `dataset_stats.json` and `data/`.
- **After completion:** `manifest.json`, `metrics.jsonl`, `status.json`, `launches.jsonl` and the upload receipts.

Each upload is verified by remote size and LFS SHA-256 before a local `<label>.pt.uploaded.json` receipt is written. `latest.pt` (optimizer/RNG recovery state) is never uploaded.

At launch no LIBERO evaluation had been run. Section 9 reports the evaluation that followed.

## 2. What was implemented (plan Section 10)

| Task | Files | Content |
|---|---|---|
| A | `src/fastwam/models/wan22/residual_transport.py`, `scripts/elastic_schedules.py`, `scripts/transport_experiments.py` | Nine `flat_k{1,2,4}_n{2,5,10}` schedules are added after the legacy ones; legacy names and order are unchanged. The balanced sampler shuffles blocks of nine with a private `random.Random("elastic-balanced-v1:<seed>:<names>")`, so it does not touch global RNG and is prefix-stable. The legacy `random.Random(seed+update*65537).choice` selector is kept for F0-PB. A separate `ELASTIC` registry leaves `EXPERIMENTS` and its gates unchanged. |
| B | `scripts/elastic_init.py`, `scripts/train_transport.py` (`load_elastic_init`) | Hashed init artifacts. `common_v1` builds StepCondition and the T4 transport from independent label-seeded CPU generators, so E0/E1/E2/SP11 start from identical student and step-condition tensors. `rt_a2_legacy_v1` replays RT-A2's historical constructor and RNG order; T4 tensors are kept but unused by F0-PB. Loading is strict and verifies with `torch.equal`. |
| C | `src/fastwam/models/wan22/transport_policy.py` | Own-state macrostep targets, separate 10-step teacher endpoint, `d=1/N`, d=0 anchor at K=4, and truncation at `j <= N-3` (existing RT code). New: bit-identical reuse of the j=0 local target from the endpoint trajectory, which saves `10/N` teacher passes per update. Per-step detached diagnostics (Section 11). |
| D | `scripts/train_transport.py` | `find_unused_parameters=True` for E2: globally unused transport grads stay `None` at K=4, so AdamW does not advance their step. Adaptive activation checkpointing is used when block calls are at least `--checkpoint-min-blocks`. A persisted and hashed `schedule_sequence.json` is verified on resume. The contract also records `init_parts`, `latent_content_digest` and `checkpoint_min_blocks`. |
| E | `tests/test_elastic_inference.py`, `scripts/preflight_elastic.py` | Flat schedules load through `TransportInference` and run eager and CUDA-graph rollouts. The schedule is checked against NFE, history is reset per query, and transport is bypassed at four loops. |
| F | `scripts/preflight_elastic.py`, `scripts/run_elastic.sh`, `scripts/elastic_manifest.py`, `scripts/upload_elastic.py` | Real-model preflight report; resume-aware single-writer run wrapper; per-run Section 17 manifest; verified uploader. |

Tests (CPU, `PYTHONPATH=src:scripts:.`):

- New: `tests/test_elastic_schedules.py`, `tests/test_elastic_initialization.py`, `tests/test_elastic_distillation.py` and `tests/test_elastic_inference.py`.
- Existing: `tests/test_residual_transport.py`, `tests/test_transport_training.py` and `tests/test_loop_mot.py`.
- Result: 97 passed, 2 skipped.

## 3. Shared contract as launched (plan Section 9)

| Setting | Value used |
|---|---|
| Teacher / parent | BASE10 `v0-video4-action4/policy.pt`, export sha256 `1c659a7c…602b` (source `4ffe1b14…e5cc`) |
| VAE | `Wan2.1_VAE.pth` sha256 `38071ab5…d981` |
| Data | 277,713 windows; normalization `d5ad351e…b3e4`; staged latent digest `f2b1b21b…b341` |
| Updates / epochs / global batch | 21,700 / 10 / 128, weighted by real windows |
| Layout | world 2, microbatch 64, no accumulation; activation checkpointing only when block calls ≥ 180 (`flat_k2_n10`, `flat_k4_n10`, S1) |
| Optimizer | AdamW (0.9, 0.95), eps 1e-8, fused; step/T4 1e-3 / 0; P-c action 3e-5 / 0.01; P-b LoRA (rank 16) 2e-4 / 0; warmup 651; existing cosine; clip 1.0 |
| Precision | FP32 weights/optimizer, BF16 autocast |
| Seeds | 42 (data order, per-window noise, schedule) |
| Flat9 sequence sha256 | `37e93bb991623cb0a9d44757203307ee140abde4ea5833153ed86d146c8b9f39` (identical for E0/E1/E2) |
| Trainable parameters | E0/E1/SP11 105,678,471; E2 105,754,919; F0-PB 2,464,896 (LoRA 2,064,384 + StepCondition 400,512) |
| Deployed parameters | base policy 584,536,135; +400,512 StepCondition; +76,448 T4 (E2) |

## 4. Initialization artifacts

| Artifact | File sha256 | student_action | step_condition | transport_T4 |
|---|---|---|---|---|
| `elastic_common_init_v1.pt` | `b2815226…076c` | `2554fb66…a6c4` | `4f1ee8d8…a3f2` | `1d17f592…cadf` |
| `rt_a2_legacy_init_v1.pt` | `1ef21388…c703` | `d29f8a41…3adb` | `1350aeb4…256b` | `4f018524…ead7` |

The full digests are in `plans/elastic_evidence/initialization/`. The artifact files are in `/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work/init/`.

## 5. Validation before production

1. **F0-PB provenance (RT-A2 replay).**
   - The `RT-A2-replay` registry entry (T4, P-b, legacy selector, `rt_a2_legacy_v1`, world 1, `--no-checkpoint-blocks`) was trained for 20 updates.
   - It was compared with the historical `single_gpu_production_first20.json` (RT-A2, source `a7403b5`).
   - Schedule, windows, grad norm and all four losses were **bit-identical on 20/20 updates** (`validation/rt_a2_replay_comparison.json`).
   - This confirms the reconstructed LoRA/StepCondition initialization, data and noise streams, schedule selector, optimizer and teacher-prefix reuse against the historical run. F0-PB therefore uses the reconstructed historical init; the F2-PB-R contingency is not needed.
2. **Real-model preflight** (`validation/preflight_report.json`, all passed):
   - **Asset identity:** asset and data identity confirmed.
   - **Frozen tensors:** BASE10 video expert, proprio encoder and action base weights equal the RT-A2 export bit for bit.
   - **Counts:** deduplicated parameter counts match.
   - **Shared init:** E0/E1/E2/SP11 start from identical tensors equal to the parent action expert.
   - **All nine flat paths on E2:** finite, and bit-identical loss, logs and every gradient with teacher-prefix reuse on or off. Teacher passes drop from 20 to 15/18/19 for NFE 2/5/10. Transport receives gradients at K<4 and none at K=4.
   - **Adaptive checkpointing:** bit-identical to no checkpointing.
   - **All six front paths on F0-PB:** finite and bit-identical; all 120 LoRA tensors receive gradients.
   - **Export round trip:** export → `TransportInference` reload gives eager and CUDA-graph rollouts equal to the in-memory student.
3. **Distributed and resume checks** (`validation/smoke_comparison.json`, E2 with the production sequence):
   - Updates 1–8 cover shallow→full (K=4)→shallow transitions.
   - World 1 vs world 2: bit-identical on updates 1–4.
   - Stop at update 4 and resume to 8: bit-identical metrics and final weights versus an uninterrupted run.
   - The transport Adam step counter is 6 after 8 updates, because updates 5–6 had K=4 and correctly skipped transport.

## 6. Throughput and memory (1×H100 benchmarks, `benchmarks/`)

P-c, seconds per 128-window update on one GPU:

| Setting | k1n2 | k1n10 | k2n5 | k4n2 | k4n10 |
|---|---|---|---|---|---|
| mb64, no checkpointing | 2.83 | 3.57 | 3.35 | 3.13 | OOM |
| mb64, checkpointing | 3.34 | 5.27 | 4.47 | 4.09 | 8.15 |

- mb32 was slower (4.67–8.59 s).
- P-b T4 at mb64 (RT-A2 layout): S8 3.19, S2 3.98, S1 4.73 s; this matches RT-A2's recorded speed.
- **Layout chosen:** microbatch 64 per GPU, no accumulation, checkpointing only for the two configurations that do not fit.
- **Production at launch, world 2:**
  - Mean 2.42 s/update.
  - Per configuration: k1n2 1.85, k1n5 1.86, k1n10 2.10, k2n2 1.40, k2n5 2.13, k2n10 3.25 (checkpointed), k4n2 1.73, k4n5 2.10, k4n10 5.17 s (checkpointed).
  - Peak allocated 74.2 GB.
  - The teacher, at 15–19 passes of a 30-block path per update, dominates short schedules.

## 7. Operations

- **Watcher:** `campaign/elastic_watcher.sh` runs on login node evuser2 (nohup setsid) and holds `srun --jobid=1080499 --overlap` steps. Each step runs `campaign/lane.sh <lane> <gpus>`. Training does not depend on any interactive shell. No other job is touched.
- **Run wrapper:** `scripts/run_elastic.sh`.
  - Takes a per-run lock, records each launch in `launches.jsonl` and resumes automatically from `latest.pt`.
  - Saves recovery state every 100 updates and retries up to 3 times.
  - Starts the background uploader.
  - Checks completion: `status.json` complete at 21,700, plus the checkpoint labels.
- **Finalization:** after completion, `lane.sh` writes `manifest.json` (`scripts/elastic_manifest.py`) and uploads it with the training records.
- **Paths:**
  - Runs: `/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic/<RUN>/`.
  - Logs: `/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work/logs/`.
  - Latents: node-local copy `/dev/shm/elastic_latents_1080499`.

## 8. Not done in this campaign

- **Z10:** the unmodified parent under budget overrides (9 conditions) was not evaluated.
- **F2-PB full3:** the full-depth points ([4,4], [4]x5, [4]x10) were not evaluated. RT-A2 has no H100 evaluation at those schedules.
- **Coverage:** the 37 new conditions plus the 6 reused F2-PB front points make 43 of the 55-condition initial matrix (plan Section 12.3).
- **RT-B2a reference:** RT-B2a at parent2 has a complete matched evaluation (94.57%), but it was scored in the remote campaign (`evaluation_report/campaign/run_summary.csv`), not on this H100 stack.
- **Four-GPU training launcher:** not added. The launcher contract stays at 1/2 GPUs (`--expected-gpus`).
- **Independent training seeds:** Section 16 and the deferred or conditional runs of Sections 14–15 were not run.

## 9. Matched LIBERO evaluation (2026-10-29/30)

**Protocol.** `lightwam_20261009_matched` through `scripts/evaluate_rt_matched.py`:
- 4 suites × 10 tasks × 50 official initial states × seeds 42/43/44, i.e. 6,000 episodes per condition;
- fixed noise, 30 settling actions, 400/700 policy-step caps, 32 predicted and 10 executed actions;
- every condition scored from the raw `epoch_10.pt` through the student-only `TransportInference` CUDA-graph backend.

**Before scoring:**
- **Graph gates:** the CUDA-graph backend matched eager bit for bit on real simulator observations: 592/592 queries across every evaluated schedule of every run (E0 48+96, E1 48+96, E2 48+96, SP11 16, F0-PB 32+112; `elastic_eval/gates/*.json`, also on Hugging Face).
- **Parity:** the elastic stack reproduced 20/20 stored RT-A2 episodes exactly (`campaign/parity_rt_result.log.md`).
- **Predeclared analysis:** fixed in `scripts/elastic_eval_report.py` before any final-checkpoint episode was scored, and extended twice before the added conditions were scored. Its text is in `plans/elastic_evidence/eval/results.md`.

**Where it ran.** Eval lanes on 1080499 (evc102, GPUs 0–3) and 1093967 (evc103, GPUs 0–1), each with a private CUDA MPS daemon.

**Evidence.**
- `plans/elastic_evidence/eval/results.json`/`.md`: every condition, per-suite and per-seed rates, and every contrast with its interval and test.
- Hugging Face `libero-all-suites/Elastic/eval/`: the same files plus one episodes CSV, `manifest.json` and `summary.json` per condition, and the gate files.

**Status:** complete on 2026-10-30 at 02:49 EDT. All 37 conditions have all 6,000 episodes (222,000 episodes in all) and a runner-validated `summary.json`. Every group record matches its condition's checkpoint sha256 and schedule. Published to Hugging Face at commit `2da56861` (122 files, verified present). One incident: when the 1093967 lane was stopped at 22:16, its 16 in-flight groups (F0-PB S7 and parent2) kept claims that the 1080499 runners had already passed over. The lanes' automatic retry at 02:36 took over the stale claims and scored them. Separately, the runner logs show 3 of the 4,440 groups run twice, through a claim-takeover race between the two lanes (E1 [1,1] spatial s42 t6, E0 [4]x5 Long s42 t4, F0-PB [4,4] Long s43 t0). Each pair produced identical success counts (50/50, 40/40, 46/46), as the deterministic protocol requires. Each group is a single record file, so every condition still holds exactly 6,000 episodes.

### 9.1 P-c runs (E0/E1/E2/SP11), all nine flat schedules

Success % over 6,000 episodes, with LIBERO-Long success after 'L'. 'Fast ms' is the mean batch-one latency of the exact `fast` backend on H100 (Section 10). The parent (v0 4/4, BASE10) scored 94.92% (Long 87.87%) under the same protocol.

| Schedule | Depths | E0 | E1 | E2 | SP11 | Fast ms E0 / E1 / E2 |
|---|---|---|---|---|---|---|
| flat_k1_n2 | [1,1] | 94.33 (L 86.67) | 93.73 (L 85.00) | 94.07 (L 85.40) | 93.77 (L 85.00) | 24.7 / 24.8 / 24.8 |
| flat_k1_n5 | [1]x5 | 94.53 (L 85.47) | 94.30 (L 86.47) | 94.25 (L 86.87) |  | 33.3 / 33.4 / 33.6 |
| flat_k1_n10 | [1]x10 | 93.68 (L 85.53) | 93.48 (L 84.60) | 94.00 (L 86.13) |  | 47.9 / 47.7 / 48.4 |
| flat_k2_n2 | [2,2] | 94.53 (L 86.80) | 94.58 (L 87.07) | 94.75 (L 87.40) |  | 27.6 / 27.6 / 27.7 |
| flat_k2_n5 | [2]x5 | 94.85 (L 87.80) | 94.52 (L 87.73) | 95.07 (L 89.07) |  | 39.1 / 39.1 / 39.4 |
| flat_k2_n10 | [2]x10 | 93.83 (L 85.27) | 94.00 (L 85.73) | 94.07 (L 86.67) |  | 59.9 / 59.7 / 60.7 |
| flat_k4_n2 | [4,4] | 94.97 (L 87.73) | 95.22 (L 88.27) | 95.27 (L 88.60) |  | 33.2 / 33.4 / 33.4 |
| flat_k4_n5 | [4]x5 | 95.45 (L 89.40) | 95.65 (L 89.73) | 95.28 (L 88.53) |  | 54.5 / 54.3 / 54.2 |
| flat_k4_n10 | [4]x10 | 94.87 (L 88.73) | 94.45 (L 86.73) | 94.70 (L 87.53) |  | 86.2 / 86.3 / 86.0 |

### 9.2 P-b runs: F0-PB (no transport) vs F2-PB (RT-A2, T4 transport)

F2-PB is RT-A2 epoch 10 (sha256 `25ad8ca4...`), from its complete H100 evaluation of 2026-10-13 by the same runner. Contrasts are secondary and unadjusted, with the p-value from the paired cluster permutation test.

| Schedule | Depths | F0-PB | F2-PB (RT-A2, H100) | F0-PB - F2-PB pp (95% CI), p | Fast ms F0-PB / RT-A2 |
|---|---|---|---|---|---|
| S8 | [4,1] | 94.50 (L 87.67) | 94.53 (L 87.80) | -0.03 (-0.72 to +0.65), 0.9593 | 31.8 / 31.9 |
| S7 | [4,2] | 94.70 (L 86.87) | 94.93 (L 87.87) | -0.23 (-0.87 to +0.40), 0.4928 | 31.3 / 31.5 |
| parent2 | [4,4] full | 95.88 (L 91.73) | - |  | 37.0 / - |
| S6 | [4,1,1,1,1] | 93.75 (L 86.13) | 94.48 (L 86.87) | -0.73 (-1.40 to -0.03), 0.0379 | 42.1 / 42.0 |
| S5 | [4,2,2,2,2] | 94.25 (L 86.73) | 94.98 (L 88.00) | -0.73 (-1.37 to -0.10), 0.0218 | 45.1 / 47.7 |
| S2 | [4]+[1]x9 | 92.05 (L 80.93) | 93.22 (L 84.33) | -1.17 (-1.92 to -0.40), 0.0041 | 54.5 / 59.2 |
| parent5 | [4]x5 full | 95.75 (L 90.60) | - |  | 59.7 / - |
| S1 | [4]+[2]x9 | 93.38 (L 84.00) | 94.10 (L 86.87) | -0.72 (-1.37 to -0.03), 0.0361 | 69.2 / 76.4 |
| parent10 | [4]x10 full | 94.68 (L 87.93) | - |  | 103.8 / - |

### 9.3 Predeclared contrasts at the fastest point [1,1]

Paired over identical episode keys. 95% cluster-bootstrap interval (10,000 resamples). Two-sided paired cluster sign-flip permutation test, with Holm correction across the three primary contrasts. Every secondary grid contrast is in `plans/elastic_evidence/eval/results.md`.

| Contrast | Role | Difference pp (95% CI) | p (perm.) | p (Holm) | Long pp |
|---|---|---|---|---|---|
| E1-E0 @ flat_k1_n2 | primary | -0.60 (-1.32 to +0.08) | 0.1012 | 0.3036 | -1.67 |
| E2-E0 @ flat_k1_n2 | primary | -0.27 (-0.93 to +0.38) | 0.4577 | 0.7223 | -1.27 |
| E2-E1 @ flat_k1_n2 | primary | +0.33 (-0.35 to +1.02) | 0.3612 | 0.7223 | +0.40 |
| SP11-E0 @ flat_k1_n2 | secondary | -0.57 (-1.28 to +0.13) | 0.1159 |  | -1.67 |

### 9.4 Findings

All statements below come from the predeclared analysis. Anything not predeclared is labelled descriptive. The parent BASE10 (v0 4/4) scored 94.92% (Long 87.87%) under the same protocol.

1. **Primary question: transport at the fastest point [1,1].** No difference is detectable.
   - E1−E0 is −0.60 pp (95% CI −1.32 to +0.08; Holm p 0.30).
   - E2−E0 is −0.27 pp (−0.93 to +0.38; Holm p 0.72).
   - E2−E1 is +0.33 pp (−0.35 to +1.02; Holm p 0.72).
   - Both transport-vs-E0 intervals exclude the plan's +0.5 pp "keep transport" gain (Section 13.2). Under this recipe, neither identity nor learned transport earns its place at [1,1].
   - Identity is not shown noninferior to learned transport within 0.5 pp: the E1−E2 interval reaches −1.02.
   - A nonsignificant difference is not equivalence. Following plan Section 13.3, the outcome is "E0 strong, E1/E2 not convincingly better": **the main method is elastic training without transport (E0)**, and transport is reported as inconclusive or unhelpful under the tested recipe.
2. **Full-quality retention at [4]x10** (target within 0.5 pp of the parent; point estimates, not a predeclared test):
   - E0 94.87 (−0.05), E1 94.45 (−0.47), E2 94.70 (−0.22). All three are within the target.
   - Long: E0 88.73, E1 86.73, E2 87.53, against the parent's 87.87.
3. **One checkpoint, nine operating points (descriptive).**
   - E0's success stays between 93.68 and 95.45 across all nine flat schedules.
   - [4]x5 (half the parent's denoising steps) is the best flat point for every run: E0 95.45, E1 95.65, E2 95.28, all above the parent's 94.92.
   - At every depth k ∈ {1, 2, 4}, ten denoising steps score below five steps for every run (9/9 comparisons).
   - [1]x10 is each run's weakest point.
4. **SP11 (specialist at [1,1]).** SP11−E0 is −0.57 pp (−1.28 to +0.13; p 0.12, secondary).
   - The [1,1] specialist did not beat the elastic checkpoint. So there is no detectable equal-budget cost of elasticity at the fastest point.
   - The interval excludes an SP11 advantage larger than 0.13 pp.
5. **P-b control (F0-PB, no transport) vs F2-PB (RT-A2, learned transport).** These contrasts are secondary and unadjusted. The arms were scored two weeks apart by the same runner on H100, and the stack reproduced 20/20 stored RT-A2 episodes.
   - F0-PB is below F2-PB at all six front points: S8 −0.03, S7 −0.23, S6 −0.73 (p 0.038), S5 −0.73 (p 0.022), S2 −1.17 (p 0.004), S1 −0.72 (p 0.036).
   - The gaps concentrate in Long, up to −3.40 pp at S2.
   - With E1/E2 not improving on E0, this matches the plan's reading: "transport may compensate for a restricted first-full/adapter regime; the P-c question remains different". In the LoRA-restricted P-b regime, learned transport helps. With full-action fine-tuning (P-c), it does not.
6. **F0-PB at full depth (descriptive).** The P-b student scored [4,4] 95.88 (Long 91.73), [4]x5 95.75 and [4]x10 94.68.
   - [4,4] is the highest success of all 37 conditions, with two denoising steps at full depth.
   - This was not a predeclared comparison and F2-PB has no full-depth H100 result. It suggests that step distillation at full depth, rather than depth reduction, is where these students gain the most.

## 10. Inference latency of the Elastic checkpoints

Every Elastic operating point was timed under the all-models protocol: 500 replayed raw observations, batch 1, raw observation to denormalized host actions, on H100 80GB HBM3. That report is `plans/performance/LoopWAM_All_Models_Latency_20261030.md`.
- **Same loader for every run:** E0, E1, E2, SP11 and F0-PB are timed through the same student-only loader. E2's learned transport is inside the timed graph.
- **Memory:** the report gives memory with one operating point resident and with all nine resident (plan Section 12.4).
- **Exactness:** the `fast` backend reproduces the evaluated actions bit for bit.

Results below use the `fast` backend, batch 1, on H100. The full tables are in Sections 5 and 7 of the latency report.

- **One E0 checkpoint spans 3.5× in latency.** It runs from 24.7 ms at [1,1] (40 Hz) to 86.2 ms at [4]x10.
  - Eager covers 48.8–227.5 ms over the same points, and the campaign CUDA graph 30.0–98.3 ms.
  - Every compared action chunk is bit-identical to eager: 50/50 per point, 1,850 in all over the 37 Elastic rows.
- **Transport adds no measurable latency.**
  - E0, E1 and E2 are within 2% of one another at every flat point. At [1,1] they run at 24.7, 24.8 and 24.8 ms.
  - F0-PB (no transport) and RT-A2 (learned T4 transport) are within 1% at S8, S7 and S6.
  - At S5, S2 and S1, F0-PB's fast mean is 5–9% lower, but the two runs' campaign-graph latencies agree within 2% there. That is within the two-level latency noise (latency report, Section 11), not a resolved cost of transport.
- **All nine points resident at once (plan Section 12.4).** Keeping all nine fast graphs of one checkpoint resident takes 2.69 GB allocated, with a 3.18 GB peak; the caching allocator reserves 8.36 GB.
  - A single [1,1] graph peaks at 2.58 GB.
  - So a deployment can switch operating point on every query, without re-capture, for about 0.6 GB of extra peak memory.
- **Against the parent BASE10** (v0 4/4 at ten steps, 94.92%): the parent ran at 99.3 ms (fast) in this sweep, and other 4/4-depth LoopWAM models at 89–99 ms.
  - E0 [4,4] scores 94.97% at 33.2 ms: about 2.7–3.0× faster at the same success.
  - E0 [4]x5 scores 95.45% at 54.5 ms: 1.6–1.8× faster.
  - These success comparisons with the parent are descriptive. No parent contrast was predeclared.
- **SP11 vs E0 at [1,1].** The two run an identical graph, at 23.4 vs 24.7 ms. The difference is latency-level noise, not a model difference.
- **F0-PB [4,4]** scores 95.88% at 37.0 ms. That is the highest four-suite success of any model in the repository under 50 ms. Only the five ChronoLoop a4 runs (95.90–96.18%, 91–100 ms) and FastWAM (96.63%, 107.5 ms) score higher, and all of them are 2.5–2.9× slower.
