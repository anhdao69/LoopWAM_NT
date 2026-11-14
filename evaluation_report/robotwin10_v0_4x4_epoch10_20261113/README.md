# RoboTwin-10 v0 4/4, epoch 10 — H100 evaluation

Completed 2026-11-13 05:27 EST on NVIDIA H100 80GB HBM3 (Slurm job 1187428, node evc104). All 20 task/setting cells are complete: 2,000 scored episodes (10 tasks × clean/randomized × 100 accepted episodes, seed 42, NFE 10), one attempt per cell, no resumes.

| Setting | Successes / episodes | SR | 95% CI (Wilson) |
| --- | ---: | ---: | --- |
| demo_clean | 810/1000 | **81.00%** | 78.45–83.31 |
| demo_randomized | 818/1000 | **81.80%** | 79.29–84.07 |
| combined | 1628/2000 | **81.40%** | 79.64–83.04 |

## Per task (successes out of 100 accepted episodes)

| Task | Clean | Randomized | Combined SR | Clean, epoch 5 (same harness) | Randomized, epoch 5 (same harness) |
| --- | ---: | ---: | ---: | ---: | ---: |
| adjust_bottle | 100 | 99 | 99.50% | 100 | 99 |
| grab_roller | 100 | 100 | 100.00% | 100 | 100 |
| hanging_mug | 55 | 62 | 58.50% | 45 | 52 |
| move_stapler_pad | 71 | 74 | 72.50% | 66 | 67 |
| open_microwave | 38 | 52 | 45.00% | 28 | 41 |
| press_stapler | 92 | 93 | 92.50% | 93 | 92 |
| scan_object | 89 | 80 | 84.50% | 84 | 79 |
| stack_blocks_two | 98 | 97 | 97.50% | 99 | 97 |
| stamp_seal | 93 | 86 | 89.50% | 90 | 92 |
| turn_switch | 74 | 75 | 74.50% | 66 | 72 |
| **mean** | **81.0** | **81.8** | **81.40%** | 77.1 | 79.1 |

## Epoch 10 versus epoch 5 on the same harness (paired)

The epoch-5 parent (`bb1563f16bcd102b…`) was evaluated with the same H100 harness, policy code, scene-seed list, per-seed instructions and query noise between 2026-11-04 23:43 EST and 2026-11-09 11:50 EST (28 attempts for 20 cells; interrupted cells resumed at the next scene seed, so no seed is scored twice). Pairs are the scene seeds accepted in both runs; RoboTwin's expert check is not fully deterministic, so a few seeds per cell differ. On every shared seed both runs received the same instruction (1,882/1,882).

| Setting | Epoch 5 | Epoch 10 | Shared seeds | Epoch 10 only | Epoch 5 only | Paired difference | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 77.10% | 81.00% | 943 | 70 | 38 | +3.39 pp | 0.003 |
| randomized | 79.10% | 81.80% | 939 | 70 | 45 | +2.66 pp | 0.025 |
| combined | 78.10% | 81.40% | 1,882 | 140 | 83 | +3.03 pp | 0.00016 |

Per task, the largest gains are on the hard tasks: hanging_mug (+10 / +10), open_microwave (+10 / +11), turn_switch (+8 / +3) and move_stapler_pad (+5 / +7) clean / randomized; stamp_seal randomized is the largest drop (−6). Single task/setting cells have 100 episodes, and only hanging_mug randomized reaches p < 0.05 on its own (p = 0.024, not corrected for 20 cells). Per-cell counts, shared seeds and p-values are in [task_results.csv](task_results.csv).

## Context: the same epoch-5 checkpoint in this report collection

Three earlier evaluations in this folder tree used the identical epoch-5 checkpoint (their final audits record SHA-256 `bb1563f16bcd102b…`). With this harness's own epoch-5 run, there are four evaluations of epoch 5 on four hosts:

| Checkpoint / evaluation | Clean | Randomized | Combined | Successes | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| epoch 5 — original local | 77.40% | 80.30% | 78.85% | 1577/2000 | [report](../robotwin10_v0_4x4/final_report.md) |
| epoch 5 — server189 rerun | 78.40% | 78.00% | 78.20% | 1564/2000 | [report](../robotwin10_v0_4x4_remote189/final_report.md) |
| epoch 5 — server137 rerun | 76.60% | 79.30% | 77.95% | 1559/2000 | [report](../robotwin10_v0_4x4_remote137_20261105/final_report.md) |
| epoch 5 — H100, this harness | 77.10% | 79.10% | 78.10% | 1562/2000 | [episodes](reference_epoch05_episodes.csv) |
| **epoch 10 — H100, this harness** | **81.00%** | **81.80%** | **81.40%** | 1628/2000 | [episodes](episodes.csv) |

The four epoch-5 evaluations span 77.95–78.85% combined; epoch 10 is +2.55 to +3.45 pp above them. Only the H100 pair is episode-paired; the other hosts used unseeded instruction sampling and different evaluation code, so they are aggregate references. The epoch-10 row is not added to the consolidated `all_results` exports, which are built on ws by `all_results/build_snapshot.py`.

## Model

- Checkpoint: `epoch_10.pt`, SHA-256 `b85ad555fa60f5aab61a35231f26d6ca9c01e32d6111d6893abb815db5b985b8`, update 90,890. Published on Hugging Face as `anhdao69/LoopWAM_NT` / `robotwin-10/v0-video4-action4/epoch_10` (revision `5d3d7a7e`; `policy.pt` has the same SHA-256; [upload verification](evidence/training/hf_upload_verification_epoch_10.json)).
- Training: epochs 6-10 continuation of the epoch-5 release (zero-based epochs 5-9): AdamW state continued from update 45,445, LR 3e-5 for backbone and robot I/O, no warmup, cosine to 1% over 45,445 updates, global batch 128 on 4 H100 GPUs. Updates 45,446–90,890, 17.75 h, 11,633,510 windows seen in total. Code `3de6ecd` with the uncommitted continuation patch (per-file hashes in [manifest.json](evidence/training/manifest.json)).
- Parent check: the continuation resumed from the epoch-5 `latest.pt` (update 45,445, 679 AdamW states), whose tensors equal `epoch_05.pt` `bb1563f16bcd102b…` ([parent_verification.json](evidence/training/parent_verification.json), [resume audit](evidence/training/resume_audit_u45445.json)).
- Mean training loss by epoch: epoch 6 0.0478, epoch 7 0.0461, epoch 8 0.0442, epoch 9 0.0426, epoch 10 0.0418.

## Protocol

- Harness: RoboTwin 2.0 script/eval_policy.py (FastWAM-vendored + ChronoLoop per-seed instruction seeding); policy experiments/robotwin/loopwam_policy (original LoopMoT, stateless); v0 with video/action loops 4/4.
- Seed 42, first scene seed 4,300,000, expert check on, unseen instructions (pool 100), 100 accepted episodes per task and setting.
- Predict 32 actions, execute 24; 10 denoising steps (NFE 10), CFG 1, query noise fixed seed 42 for every query; renderer rt (RoboTwin default).
- The harness files and policy code were last modified before the first epoch-5 attempt, so both runs used the same files (hashes, modification times and git revision in [provenance.json](provenance.json)). Videos were recorded for every epoch-10 episode; they stay on the cluster and their names and SHA-256 are listed in [episodes.csv](episodes.csv).

## Files

- [episodes.csv](episodes.csv): the 2,000 scored epoch-10 episodes (scene seed, outcome, instruction, video SHA-256).
- [reference_epoch05_episodes.csv](reference_epoch05_episodes.csv): the 2,000 scored epoch-5 episodes of the same harness.
- [task_results.csv](task_results.csv) and [results.json](results.json): per-cell counts, paired statistics, references.
- `evidence/epoch10/` and `evidence/epoch05_same_harness/`: per cell `result.json`, the launch `command.json`, the harness summary and the full `eval.log` (gzip). `evidence/training/`: training manifest, timing, precision, resume audit, parent check and Hugging Face upload verification.
- [SOURCES.tsv](SOURCES.tsv): every copied file with its cluster path and SHA-256; [tools/](tools/) builds this folder from the cluster records.
- Independent check: [verify_robotwin10_v0_4x4_epoch10_20261113.py](../verify_robotwin10_v0_4x4_epoch10_20261113.py) recomputes every number above from the episode files and the per-cell logs.
