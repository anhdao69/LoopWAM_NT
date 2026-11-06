# v0-video4-action4 — H100 retrain — RoboTwin-10 evaluation setup

Prepared November 6, 2026 (Asia/Ho_Chi_Minh). See queue_status.json for the current validation/queue stage and report.md for scored progress. Preflight and calibration episodes are excluded from all reported scored SR.

## Requested checkpoint

Release: https://huggingface.co/anhdao69/LoopWAM_NT/tree/bd03a4e2fbe0b4c2d653977a8f13a62b18bda234/robotwin-10-h100/v0-video4-action4

HF revision is pinned to `bd03a4e2fbe0b4c2d653977a8f13a62b18bda234`. All 11 release files were downloaded on the local server, then verified with HF file size and LFS SHA256 or Git blob SHA1, plus supplied SHA256SUMS. For the remote run, bytes are copied by rsync and independently reverified against the pinned HF manifest before loading. Original file name is epoch_05.pt; policy.pt is a relative symlink to the same bytes. This release has no config.json; validation reads the real training_manifest.json and native payload, without inventing a release config.

Checkpoint SHA256: `cd02703769b9e71872927d3197837d67d109444aef132e62fcb0c30b54c9aa49`. Size: 2,338,751,845 bytes. Completed epoch 5, optimizer update 45,445; 5,816,755 training-window presentations. Native v0, 3/6/3 stored partition, video/action loops 4/4, late loop alignment, aligned action KV, zero history/memory, 14D qpos actions/proprioception. Policy parameters: 584,567,886. FP32 policy weights with BF16 compute; no quantization. Strict native model load validates architecture and every state-dict key; the preflight also checks finite FP32 parameters, payload variant, epoch, step, data cursor, supplied normalization, VAE and data-manifest hash.

## Training provenance and comparison limits

These are independent H100 retrains, separate from the previous robotwin-10 checkpoints. The release records four H100 GPUs, microbatch 8, global batch 128, five epochs, seed42, and FP32 weights with BF16 compute. Each release declares nine source hashes, all checked against the frozen evaluator checkout. Training source revision is `46e27fd3aa842e94344bf2893ce1c16b614a9a43`. Model/source hashes agree; do not infer identical training weights from that fact.

The supplied normalization JSON has exactly the same parsed values as the old evaluation normalization, and canonical SHA256 `38ed9c8d466356fb8f4688da470b14a036dbc51e0fc5484cc5959b507adbcd48`. Raw-file hash differs because JSON encoding differs. The raw release is retained unchanged.

Two training-contract hashes differ from the prior same-variant release:

| Artifact | Previous | H100 retrain |
| --- | --- | --- |
| Donor initialization artifact | `a990a5831c63e88ae5143689adad2f2aeb02cb1cd151d61ae0dd992990ac3296` | `a3bf2097bafddac278148c1cb0321163f68cfe2b954e206436da0e3f0bd3c535` |
| Training data manifest | `4db3a252025a04552364ee3cf0b8dad8c9facb39c66e1ef8eb16fb52941597af` | `68712d5429f20f99c953107c789f80ba7df82cc7c1128583b722068368b2d4f2` |

The declared task list, 5,500 demonstrations, 1,163,351 windows per epoch, normalization and training schedule match. These artifact-hash differences are disclosed; they are not silently ignored or asserted to prove tensor-level donor/data identity. The new manifest itself is verified against the native training contract. Source-check evidence records every contract difference.

## Matched evaluation protocol

| Setting | Value |
| --- | --- |
| Tasks | adjust_bottle, grab_roller, hanging_mug, move_stapler_pad, open_microwave, press_stapler, scan_object, stack_blocks_two, stamp_seal, turn_switch |
| Settings | demo_clean and demo_randomized |
| Accepted episodes | 100 per task/setting, 1,000 clean + 1,000 randomized = 2,000 per model |
| Evaluation seed | 42 only; candidate scene seeds start at 4,300,000 |
| Scene admission | Unchanged expert planning AND expert-success checks; rejected scene candidates advance seed |
| Language | Unseen instructions; actual strings saved |
| Inference | NFE10, CFG1, fixed CPU noise seed42 reset every query |
| Control | Predict32 actions, execute24 before replan, absolute14D qpos |
| Images | Released FastWAM PIL uint8 bilinear packing of head and both wrists; 384×320; BF16 normalization |
| Renderer | RT32 samples/pixel, depth8, OIDN2.3.3 Blackwell compatibility libraries |
| Termination | Original per-task step caps and success predicates |
| Logging | Every scored episode video; scene seed/instruction, action hashes, query timings, validated traces |

The frozen evaluator is commit `7a2181d0ecc8e30d64806e94535b347bc9e1da2e`. Its model/simulator/scoring code is reused without modification. All nine training-source files must match the H100 release. Inherited simulator/assets/environment files are rehashed before launch and bound into the new run manifest. Remote asset-path relocations reuse the previously verified setup; renderer settings and task configurations are unchanged.

The original FastWAM harness does not seed Python's instruction-template RNG. This is preserved, so equal evaluation seed42 does not guarantee identical instruction strings/trajectories across reruns. GPU hosts also differ. Results are aggregate comparisons under matched protocol, not exact paired episode experiments.

## Host, speed and preflight

Host: VMO-ANHDH35-D (local), RTX PRO 6000 Blackwell Max-Q 96 GB. Concurrent policy workers: 4. Shared GPU T5 server caches prompts; each policy worker uses CUDA graphs only after exact native-versus-graph equality on eight saved real observations in forward/reverse order with seeds42/43/44. The scored benchmark still uses seed42 only. FP32 weights/BF16 computation, NFE and renderer quality are retained. The local96GB host uses four workers; the remote48GB host uses two, matching previously sustained safe concurrency. Avoid the earlier eight/ten-worker configurations that caused GPU/renderer allocation failures.

Preflight requires the 38 existing evaluator tests on each host, renderer-guard tests, verified release/runtime hashes, strict CPU model loading, graph-output gate and four smoke episodes (adjust_bottle and open_microwave, clean/randomized). A smoke policy failure is an evaluation outcome, not an infrastructure failure; the smoke must produce all valid videos/traces and clean renderer logs. Smoke outcomes are excluded. For remote image preprocessing tests, only the reference-file lookup is relocated to the existing pinned FastWAM source; model/harness implementations are unchanged.

## Queue, resume and audit

A persistent tmux queue runs validation, then all20 task/setting cells, then finalization. A nonblocking file lock prevents duplicate controllers. Each cell has up to3 attempts and a1,800-second no-progress timeout. Only complete100-episode cells are resumed after the evaluator validates checkpoint and manifest/episode hashes. Partial interrupted attempts are retained and excluded from the scored denominator. Failed accepted policy episodes remain failures. Infrastructure failure is recorded as a failed attempt, not converted to a policy outcome.

Before final SR publication, the finalizer validates all20 cells,2,000 videos/episode traces, checkpoint/source/runtime hashes, success sums, seed/task/protocol equality with the prior same-variant run and parsed normalization equality. Accepted-cell logs containing OIDN errors, CUDA allocation failures, buffer-allocation errors or Vulkan device loss cause audit rejection. All output is separate from prior run artifacts.

The live report refreshes every10seconds. Remote reports are mirrored to this local evaluation_report folder periodically, including final tables and audit status. Audit is performed on the execution server against its full videos; syncing reports does not move all bulk videos.

## Baseline and outputs

Previous same-variant local seed42 result: clean 77.40%, randomized 80.30%, combined 78.85% (1577/2,000). Final report will compare the new H100 retrain with these preserved results. No seed43 substitutions or best-task selection.

Bulk directory on execution host: `/data14t/anhdh35/robotwin10_h100_20261106/v0-video4-action4`.

Frozen checkout on execution host: `/storage/anhdh35/LoopWAM-robotwin10-dense-eval`.

Report directory on execution host: `/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_h100_v0_4x4_20261106`.

Local report directory: `/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_h100_v0_4x4_20261106`.

Final artifacts: final_report.md, final_results.json, final_audit.json, protocol_comparison.json, task_results.csv, episodes.csv and failed_attempts.json. Live artifacts: report.md, progress.json, queue_status.json, launch.json and preflight.json. No final SR is claimed before the audit passes.
