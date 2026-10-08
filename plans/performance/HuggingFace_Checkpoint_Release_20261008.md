# Hugging Face final checkpoint release — October 8, 2026

Repository: **[anhdao69/LoopWAM_NT](https://huggingface.co/anhdao69/LoopWAM_NT)** (private).

Completed: 2026-10-08T21:39:50.360483+00:00. Verified revision: `252394c467882d0edaab20550985d4141c02ad26`.

Uploaded **11 final policy checkpoints**, **79 release files**, **29.06 GB** total. The server retains the original training checkpoints. The temporary upload credential was removed after completion.

## Classification

| Directory | Final evaluation |
|---|---|
| `libero-long/v0-video4-action4-original` | 96/100 (96%) |
| `libero-long/v1-video4-action4` | 82/100 (82%) |
| `libero-long/dense-s12` | 81/100 (81%) |
| `libero-long/v2-video4-action4` | 81/100 (81%) |
| `libero-long/dense-s30` | 86/100 (86%) |
| `libero-all-suites/v0-video4-action4` | 388/400 (97%) |
| `libero-long/v0-video4-action1` | 81/100 (81%) |
| `libero-long/v0-video4-action2` | 86/100 (86%) |
| `libero-long/v0-video2-action2` | 81/100 (81%) |
| `libero-long/v0-video1-action4` | 86/100 (86%) |
| `libero-long/v0-video4-action4-repeat` | 91/100 (91%) |

## Contents and verification

Each setup contains native `policy.pt`, architecture/depth metadata, normalization statistics, training/data manifests, training timing, evaluation results, and an export record. The repository README includes loading instructions.

Exports retain the final model tensors exactly and omit Adam optimizer states. They support inference and fresh-optimizer fine-tuning; exact optimizer-state resume uses the original server checkpoints. VAE and text features remain external dependencies.

Every original checkpoint SHA-256 was recomputed and matched against its evaluated final checkpoint. Every exported model tensor was checked for exact equality after serialization. Normalization hashes were checked. After upload, all expected file sizes were verified; Hub large-file SHA-256 values were matched to the staged files.

Both original and repeated Long 4/4 runs are included separately. Evaluation-seed reruns reuse the original weights and do not create additional checkpoints. The four-suite 388/400 score is a different training/evaluation scope from the Long-only 100-episode scores.

Evidence: [checkpoint index](../evidence/hf_release_20261008/checkpoint_index.json), [upload verification](../evidence/hf_release_20261008/upload_verification.json), [model card](../evidence/hf_release_20261008/MODEL_CARD.md).
