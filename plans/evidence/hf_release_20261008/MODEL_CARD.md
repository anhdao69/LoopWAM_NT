---
language: en
tags:
- robotics
- libero
- loopwam
- world-action-model
---

# LoopWAM final checkpoints

Final trained policy weights from the LoopWAM experiments. This repository contains **11 checkpoints**, separated by training data scope. Both original and repeated 4/4 runs are retained. All production checkpoints are from completed ten-epoch runs with global batch 128 and training seed 42.

## Checkpoint index

| Directory | Model | Video/action loops | Parameters | Final evaluation |
|---|---|---|---:|---|
| [libero-long/v0-video4-action4-original](libero-long/v0-video4-action4-original) | v0 | 4/4 | 584,536,135 | 96/100 (96%) |
| [libero-long/v1-video4-action4](libero-long/v1-video4-action4) | v1 | 4/4 | 584,536,135 | 82/100 (82%) |
| [libero-long/dense-s12](libero-long/dense-s12) | dense_s12 | 1/1 | 584,536,135 | 81/100 (81%) |
| [libero-long/v2-video4-action4](libero-long/v2-video4-action4) | v2 | 4/4 | 584,536,135 | 81/100 (81%) |
| [libero-long/dense-s30](libero-long/dense-s30) | dense_s30 | 1/1 | 1,416,114,247 | 86/100 (86%) |
| [libero-all-suites/v0-video4-action4](libero-all-suites/v0-video4-action4) | v0 | 4/4 | 584,536,135 | 388/400 (97%) |
| [libero-long/v0-video4-action1](libero-long/v0-video4-action1) | v0 | 4/1 | 584,536,135 | 81/100 (81%) |
| [libero-long/v0-video4-action2](libero-long/v0-video4-action2) | v0 | 4/2 | 584,536,135 | 86/100 (86%) |
| [libero-long/v0-video2-action2](libero-long/v0-video2-action2) | v0 | 2/2 | 584,536,135 | 81/100 (81%) |
| [libero-long/v0-video1-action4](libero-long/v0-video1-action4) | v0 | 1/4 | 584,536,135 | 86/100 (86%) |
| [libero-long/v0-video4-action4-repeat](libero-long/v0-video4-action4-repeat) | v0 | 4/4 | 584,536,135 | 91/100 (91%) |

`libero-long/`: matched 344-training / 44-validation demonstration split, 7,250 updates. Dense-S12/S30 execute their independent blocks once; their 1/1 metadata does not mean the dense networks have equal depth. S12 has 12 independent block pairs; S30 has 30. Loop models use three prelude, six shared core and three coda pairs, with 3 + 6K + 3 effective block applications per expert.

`libero-all-suites/`: one v0 trained on all 1,712 locally available Spatial/Object/Goal/Long demonstrations, 21,700 updates. Its 388/400 result aggregates all four suites, so it is a different training/evaluation scope from the Long-only rows. The same four-suite checkpoint also scored 720/2,000 (36%) on LIBERO-Pro.

All Long rows use 100 final-checkpoint rollouts at evaluation seed 42. Standard-suite evaluations use 700 policy steps, 30 settling steps, 32-action predictions, replanning every ten actions, ten denoising steps and CFG 1. The original 4/4 checkpoint additionally scored 96%, 88%, 91% across evaluation seeds 42/43/44. These are one training seed and cannot establish across-training-seed significance.

## Files and integrity

Each directory contains `policy.pt`, `dataset_stats.json`, `data_manifest.json`, `training_manifest.json`, `training_timing.json`, `evaluation_summary.json`, and `export.json`. `policy.pt` is the native `loopwam-s-v1` checkpoint, containing FP32 model tensors, architecture/depth metadata and the training contract. Adam optimizer states are omitted; use it for inference or fresh-optimizer fine-tuning, not exact optimizer-state resume. All exported model tensors were checked for exact equality with the evaluated final checkpoint. `export.json` records both the original checkpoint SHA-256 and the exported file SHA-256; these hashes differ because optimizer removal changes serialization.

The frozen VAE and text encoder/embeddings are external inference dependencies. The native loader does not require the donor initialization artifact when `checkpoint_path` is provided. Normalization files are specific to each checkpoint and must remain matched. Videos, dataset videos and original optimizer checkpoints remain on the training server.

## Loading

Use the compatible [LoopWAM code](https://github.com/anhdao69/LoopWAM_NT/tree/LoopWAM_NT), at revision `8a29ffdce1537409e2b7e975e838d990ebbe068a` or a compatible later revision. This is a custom policy, not a Transformers AutoModel checkpoint. Install that repository’s inference dependencies first. Authenticate with a Hugging Face account authorized to access this private repository.

```python
import torch
from huggingface_hub import hf_hub_download
from fastwam.models.wan22.loopwam import create_loopwam

repo = "anhdao69/LoopWAM_NT"
variant = "libero-long/v0-video4-action4-repeat"
checkpoint = hf_hub_download(repo, f"{variant}/policy.pt")
stats = hf_hub_download(repo, f"{variant}/dataset_stats.json")
vae = hf_hub_download("Wan-AI/Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth")
model = create_loopwam(checkpoint_path=checkpoint, vae_path=vae,
                       model_dtype=torch.float32, device="cuda").eval()
```

Depth/version are reconstructed from checkpoint metadata. Use the repository’s observation adapter, matched normalization, text embeddings with their padding masks, and BF16 autocast during prediction. See `scripts/evaluate_loopwam_libero.py` for the complete simulator pipeline. Evaluation summaries retain the original checkpoint hashes; use the tensor equality and source/export mapping in `export.json` when auditing exports. The evaluator computes the exported file’s own hash for a new evaluation.

## Provenance and rights

These research checkpoints derive from pretrained Wan and the FastWAM/LoopWAM implementation. Consult the upstream model and code licenses before redistribution or use; this model card does not grant additional upstream rights. The repository is private.

For detailed comparison caveats, runtime accounting and per-task results, see the [consolidated report](https://github.com/anhdao69/LoopWAM_NT/blob/LoopWAM_NT/plans/performance/LoopWAM_Consolidated_Results_20261008.md). That report’s earlier snapshot labels the repeat pending; the repeat is now complete at 91/100, as recorded here.
