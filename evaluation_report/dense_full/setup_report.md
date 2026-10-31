> **Status update — 27 October 2026:** Dense-S12 and Dense-S30 were removed from the queue at the user’s request. Their waiting controller is stopped. The setup below is a historical record; downloaded checkpoints are retained.

# Dense-S12 and Dense-S30 full-LIBERO evaluation setup

Prepared 27 October 2026 (Asia/Ho_Chi_Minh). The user requested both full-suite Dense checkpoints be appended to the existing evaluation queue. These jobs use the standard LIBERO matched protocol from the immediately preceding v0 depth sweep. They are not LIBERO-Plus or LIBERO-Pro runs.

## Queue and episode counts

The existing five conditions run first: v0 Ka=1,2,3,5,6. Then Dense-S12 runs, followed by Dense-S30. Each Dense job has four suites, ten tasks per suite, three seeds (42,43,44), and 50 initial states per task/seed: 6,000 scored episodes/model, 12,000 new scored episodes total. Each suite contributes 1,500 episodes/model; each seed contributes 2,000.

The dependent controller waits for all five predecessor jobs to report exactly 6,000 episodes with the expected v0 checkpoint hash and action depth, and for the predecessor controller lock to be released. It validates the predecessor result summaries again before running Dense. Queue errors stop advancement. A failed predecessor never authorizes Dense GPU work.

## Releases and architecture

Hugging Face repository: `anhdao69/LoopWAM_NT`; pinned revision `0d1614d6fab2af31be6451816b6658fbc780cfec`.

| Model | HF path | Trainable policy parameters | Policy bytes | SHA256 |
| --- | --- | ---: | ---: | --- |
| Dense-S12 | `libero-all-suites/dense-s12` | 584,536,135 | 2,338,643,275 | `a94f1839457abecc6ae756c2933c4ba639765f64be00618fd5089dee8531e5bd` |
| Dense-S30 | `libero-all-suites/dense-s30` | 1,416,114,247 | 5,665,343,810 | `9726eec3f4aafa74c532cf0f369f8cec9bb545af1e8d5126b65c35c1fc457b34` |

Both are epoch-10 full-LIBERO releases with 21,700 optimizer updates. Dense-S12 uses 12 independent block pairs and Dense-S30 uses 30; both have native loop counts one. No v0 action-depth override is applied. Parameter counts cover the policy (MoT plus proprio encoder), excluding frozen VAE/text encoder.

Each release includes policy, export metadata, dataset statistics, data/training manifests, training timing, fairness metadata, and README. Downloads are revision-pinned. LFS policy SHA256 and file sizes are checked against the Hugging Face API; ordinary files are checked against Git blob hashes. CPU validation checks release contracts for every suite, training completion, exact architecture, strict state-dict loading, finite weights, parameter count, and VAE/normalization/text-cache provenance. See each model's `cpu_validation.json` for actual validation evidence.

## Matched rollout protocol

- Standard LIBERO Spatial, Object, Goal, and Long (`libero_10`); task IDs 0–9.
- Seeds 42,43,44. Seed set once per task/seed group; episode initial-state indices 0–49. No per-episode reseeding; each inference query receives the group seed.
- Same saved initial-state arrays, task descriptions, BDDL definitions, simulator assets, trained normalization, text embeddings and token masks as the matched framework.
- Two camera images rendered at 256 pixels, rotated 180 degrees and resized to 224; horizontal concatenation to width448. Eight-dimensional proprioceptive input.
- Thirty settling actions. Predict 32 actions, execute 10 before replanning. NFE10, CFG1, CPU-seeded initial noise; original action denormalization and gripper conversion.
- Maximum policy steps 400 for Spatial/Object/Goal and 700 for Long. Success comes from the unchanged Light-WAM episode loop/environment success signal.
- FP32 stored model weights, BF16 autocast/VAE compute. No reduced NFE, quantization or approximate arithmetic is introduced.

The evaluator calls the original matched Light-WAM episode loop rather than implementing a new termination rule. Same evaluation protocol does not imply identical model training architecture or training compute; those differ and are preserved.

## Inference performance and correctness gates

GPU gates run after v0 completes. Eight real observations across all four suites test finite native actions of shape32×7. CUDA graphs are compared bit-for-bit with native actions over forward and reverse observation order and varying seeds/text contexts. A failed graph gate falls back to native inference.

A native two-worker calibration provides the reference. Memory-safe native four/eight-worker and CUDA-graph two/four/eight-worker configurations are benchmarked on the same sixteen unscored episodes. Candidate runs must match the complete success/step/action-query trace signature. The fastest eligible measured configuration is selected separately for each Dense model. Available memory is checked against measured per-model peak allocation with headroom. Calibration does not contribute to reported SR. Runtime estimates should be based on these measurements once available; none is presented as measured before the GPU gate runs.

## Integrity, resume, and outputs

Source, checkpoint, VAE, normalization, text cache, simulator definitions/assets and initial-state files are hashed and frozen before scoring. Cached gates are bound to the exact model, policy digest, NFE, native depths, frozen manifest and serialized real inputs. Changes fail validation.

Every episode produces a JSON record, video with hash, and action-query JSON trace with hash. A completed task/seed group commits atomically after all50 episodes. Resume validates group identity and every artifact hash. Incomplete groups replay from episode0 to preserve their RNG sequence; calibration timing never reuses a partially completed workload. Scored resumes retain the selected backend and worker count. Aggregate reports include per-suite and per-seed counts/SR. The chain publishes a live status heartbeat while waiting and while either model runs.

Controller SIGTERM propagates to active GPU subprocesses. Chain, predecessor and model queue exclusion locks are inherited into GPU runner/gate processes to reduce orphan-overlap risk. Like the existing campaign, persistence uses tmux on this host; it does not promise automatic restart after host reboot.

## Locations and operation

- Isolated evaluator: `/storage/anhdh35/LoopWAM-dense-eval` (branch `eval/dense-full-20261027`, based on audited v0 evaluator commit `fa42937`). Active v0 source remains unchanged.
- Bulk inputs/logs/videos/results: `/data14t/anhdh35/loopwam_dense_full_20261027`.
- Model output directories: `dense-s12/results/ka1`, `dense-s30/results/ka1`. Here `ka1` is inherited directory naming for native single-pass Dense, not v0 depth sweeps.
- Reports: `/storage/anhdh35/LoopWAM_NT/evaluation_report/dense_full`.
- Launcher: `/storage/anhdh35/LoopWAM-dense-eval/scripts/launch_dense_chain.sh`.
- tmux session: `dense_full_eval`.

For current state consult `queue_status.json` and `report.md`; for each model consult its subdirectory's frozen `protocol.json`, CPU validation, live report and final `summary.json`. No Dense scored episodes or GPU parity results exist until the preceding sweep completes.
