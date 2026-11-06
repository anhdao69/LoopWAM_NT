# LoopWAM v0 4/4 epoch 10 — full RoboTwin local evaluation

Prepared November 6, 2026 (Asia/Ho_Chi_Minh). Preflight passed: checkpoint verified, strict CPU load passed, native/CUDA-graph gate passed, and four simulator smoke episodes completed. The live report records the actual benchmark status.

## Requested release and provenance

The requested checkpoint is [robotwin/v0-video4-action4/epoch_10](https://huggingface.co/anhdao69/LoopWAM_NT/tree/5d548173cdb550fb60acb668a01b182c5a8f3b98/robotwin/v0-video4-action4/epoch_10), pinned to Hugging Face revision `5d548173cdb550fb60acb668a01b182c5a8f3b98`. All eight release files are downloaded unchanged and checked against HF file size plus LFS SHA256 or Git blob hash, and against the supplied SHA256SUMS. The locally generated evaluation_protocol.json is inherited from the completed epoch-5 local evaluation; it is not a release-supplied file.

Policy SHA256: `70cf5af73db5fe582c50c5f9fb68296d5271a9c0fea51509b3cdb5f9696be1ea`; size 2,338,751,589 bytes. Completed epoch 10, update 474,620, 60,751,030 training window presentations, next epoch cursor 10 and batch 0. The native `loopwam-s-robot-v1` export contains policy weights and training metadata, without the optimizer. It is original stateless v0, stored partition 3/6/3, video loops 4, action loops 4, aligned KV, no memory/history, 14D actions/proprioception, and 584,567,886 policy parameters.

This model was trained on all 50 RoboTwin tasks, 27,500 demonstrations and 6,075,103 windows/epoch. Epochs 6–10 continue the epoch-5 training with a lower learning-rate schedule, as described in the release. This is the full-RoboTwin checkpoint, not the separate robotwin-10 release.

## Evaluation protocol, matched to the local epoch-5 run

| Setting | Value |
| --- | --- |
| Tasks | All 50 original RoboTwin tasks, in the baseline order |
| Environments | demo_clean and demo_randomized |
| Scored episodes | 100 accepted episodes per task/environment; 5,000 clean + 5,000 randomized = 10,000 |
| Evaluation seed | 42 only; candidate scene seeds begin at 4,300,000 |
| Scene admission | Original expert planning AND expert-success checks; rejected candidates advance the seed |
| Instruction split | Unseen; actual instructions logged per episode |
| Denoising | NFE 10, CFG 1 |
| Action noise | CPU RNG, seed 42 reset for every policy query |
| Action schedule | Predict 32, execute 24 before replanning |
| Control | Absolute 14D qpos commands; supplied global z-score normalization |
| Precision | FP32 policy weights, BF16 inference/VAE; no quantization |
| Images | Released FastWAM PIL bilinear uint8 preprocessing; head + two wrists; packed 384×320, then BF16 normalization |
| Renderer | RT, 32 samples/pixel, path depth 8, OIDN 2.3.3 for Blackwell compatibility |
| Termination | Unmodified per-task horizons and success predicates |
| Logging | Every accepted episode video; scene seed, instruction, query timings and action/trace hashes |

The frozen evaluator is `/storage/anhdh35/LoopWAM-robotwin-eval`, commit `0a0c6e57c56db40d3ff244b6857621874cbc38c9`, the same checkout as the epoch-5 evaluation. Preflight checks all baseline manifest-bound source files and runtime assets against their old hashes. The model/loader files are compared with release runtime revision `8326632`. Model, simulator, scoring and rendering code remain unchanged. New operational scripts use a separate checkpoint, output and report directory.

Normalization raw-file SHA256 is `7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095`; canonical sorted-JSON hash is `38ed9c8d466356fb8f4688da470b14a036dbc51e0fc5484cc5959b507adbcd48`. VAE SHA256 is `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`. Strict loading verifies these against the checkpoint's training contract, checks update/epoch/window count, parameter count, FP32 finite tensors and shared architecture/data fields against epoch 5.

## Execution speed and validation

The local host is VMO-ANHDH35-D, with an RTX PRO 6000 Blackwell Max-Q 96 GB. Use four concurrent CUDA-graph workers and a shared GPU T5 server with prompt caching. Thread limits are OMP/MKL 2 and OpenBLAS 1. CUDA graphs must match native inference exactly (atol=rtol=0) on eight saved real simulator observations, forward/reverse ordering and noise seeds 42/43/44. Those extra seeds test the graph implementation; scored evaluation uses seed 42 only.

Four workers reuse the successful epoch-5 recovery setting and later sustained evaluations. Short eight/ten-worker calibrations previously underestimated long-run renderer/GPU memory growth. They are not used for this run. No task, episode, inference-step, task-horizon or rendering-quality reduction is made for speed. This is the fastest concurrency supported by the available sustained-run evidence, not a claim of a hardware-wide optimum.

Preflight requires 26 evaluator tests; renderer-guard tests; inherited source/runtime verification; strict CPU loading; graph equality; and four unscored smoke episodes (adjust_bottle and click_bell, clean and randomized). Smoke episodes and graph inputs are excluded from scored SR. See the copied validation JSON files and tests.log for actual outcomes.

The original runtime environment is reused at `/data14t/anhdh35/loopwam_robotwin_v0_20261030/env`, with PyTorch 2.7.1+cu128, SAPIEN 3.0.0b1, MPLib 0.2.1, CuRobo 0.7.8, Warp 1.12.1 and OIDN 2.3.3. Shared assets and T5/VAE weights are hash verified; no repeated large asset download is needed.

## Queue, fault handling and final audit

The persistent tmux queue performs validation and smoke tests before launching all 100 task/setting cells automatically. Each cell allows up to three attempts and a 1,800-second no-progress timeout. Only fully completed 100-episode cells are accepted on resume; interrupted attempts remain logged and excluded from the denominator. Infrastructure failures are not counted as policy failures. A policy failure in a valid accepted episode is counted as a failure. Failed queues stop and retain their error evidence; no success report is issued merely because a process exited.

The live report updates every ten seconds. Finalization rechecks the checkpoint, runtime/source hashes, all 100 cell records, all 10,000 videos/traces, per-cell success sums and the matched protocol. It rejects accepted-cell logs containing OIDN errors, CUDA allocation failures, Vulkan device loss or buffer-allocation errors. This stricter renderer audit comes from the recent evaluation recovery. Historical epoch-5 SR is quoted as previously audited, not retroactively certified by this new guard.

Final outputs include final_report.md, final_results.json, final_audit.json, task_results.csv, episodes.csv, failed_attempts.json and protocol_comparison.json. No final SR is claimed until those checks pass.

## Comparison and interpretation

Historical local epoch-5 baseline: clean 4,580/5,000 (91.60%), randomized 4,593/5,000 (91.86%), combined 9,173/10,000 (91.73%). See [epoch-5 report](../robotwin_v0/report.md). The final epoch-10 report will show absolute differences in percentage points, including task-level outcomes.

The released FastWAM harness leaves Python's instruction-template RNG unseeded. This behavior is preserved. Equal seed 42 therefore does not guarantee identical instructions or accepted scene sets across independent runs. This comparison matches the aggregate evaluation protocol; it is not an exactly paired trajectory experiment. The local RT/OIDN renderer and PIL image preprocessing also differ from the uploaded H100 evaluator; compare this run primarily with the prior local epoch-5 run.

The epoch-5 campaign took about 21.8 hours including its earlier eight-worker failures and recovery. That is historical context, not a reliable epoch-10 completion estimate; a new ETA should use observed progress across several task types.

Bulk directory: `/data14t/anhdh35/loopwam_robotwin_v0_epoch10_20261106`.

Report directory: `/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin_v0_epoch10`.
