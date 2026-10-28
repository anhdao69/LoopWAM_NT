# LoopWAM Elastic campaign: implementation, validation and launch (2026-10-28)

Plan: `plans/LoopWAM_elastic.md`. Branch: `elastic`. Training source commit: `57f36a41a67b29fb1e4fed305d5c9a3b1e2da8b6`.
Evidence referenced below is in `plans/elastic_evidence/`.

## 1. Status

All five new training runs from plan Section 8 are queued in the user's interactive job 1080499 (evc102, 4×H100 80GB).

| Run | Transport | Scope | Rollout sampler | Init artifact | Status at writing |
|---|---|---|---|---|---|
| E0 | T0 | P-c | balanced flat9 | `common_v1` | training, lane A (GPUs 0,1) |
| E2 | T4 | P-c | same flat9 sequence | `common_v1` | training, lane B (GPUs 2,3) |
| E1 | T1 | P-c | same flat9 sequence | `common_v1` | queued |
| SP11 | T0 | P-c | fixed `flat_k1_n2` | `common_v1` | queued |
| F0-PB | T0 | P-b | historical `legacy_choice` over S1,S2,S5,S6,S7,S8 | `rt_a2_legacy_v1` | queued |

The two lanes share one ordered queue (E0, E2, E1, SP11, F0-PB). A per-run `flock` lets whichever lane frees first take the next run.

- Measured throughput at launch: 2.42 s/update for both E0 and E2, averaged over updates 6–74. That is about 14.5 h per run.
- The F0-PB time is not yet measured on this layout.
- The plan's lane table puts F0-PB third on lane A. With a shared queue, F0-PB instead starts as soon as either lane is free, which is no later.
- This report does not claim any training outcome.

Checkpoints and records are published to `https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/libero-all-suites/Elastic/<RUN>/`:

- **Checkpoints:** `step_00002000.pt`, `epoch_08.pt`, `epoch_09.pt` and `epoch_10.pt`. The predeclared selection is the raw `epoch_10.pt`.
- **Run metadata:** `config.json`, `schedule_sequence.json`, `dataset_stats.json` and `data/`.
- **After completion:** `manifest.json`, `metrics.jsonl`, `status.json`, `launches.jsonl` and the upload receipts.

Each upload is verified by remote size and LFS SHA-256 before a local `<label>.pt.uploaded.json` receipt is written. `latest.pt` (optimizer/RNG recovery state) is never uploaded.

No LIBERO evaluation was run. Evaluation (Section 12), Z10 and the F2-PB re-evaluation are outside this request.

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

- **Evaluation:** all of Section 12 (19/55 conditions), Z10 overrides, F2-PB re-evaluation and the RT-B2a reference.
- **Evaluator and latency scripts:** `scripts/evaluate_rt_matched.py` and `scripts/measure_rt_latency.py` were not modified for flat9.
- **Four-GPU launcher:** not added. The launcher contract stays at 1/2 GPUs (`--expected-gpus`).
- **Independent training seeds:** Section 16 and the deferred/conditional runs of Sections 14–15 were not run.
