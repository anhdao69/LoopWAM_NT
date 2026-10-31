# LoopWAM v0 RoboTwin evaluation — local RTX run

Prepared 30 October 2026; updated after final verification on 31 October 2026 (Asia/Ho_Chi_Minh). The local evaluation completed all 10,000 episodes at 20:06:56 on October 31. Final SR: clean 91.60%, randomized 91.86%, combined 91.73%. See [the final audited report](report.md) for all 50 tasks, timing, failure accounting and exports. This report describes the local RTX evaluation separately from the H100 evaluation described in the checkpoint release.

## Model and immutable release

- Hugging Face: https://huggingface.co/anhdao69/LoopWAM_NT/tree/4544aebb8c72b41af043efd97373dd0803e80a76/robotwin/v0-video4-action4
- Pinned HF commit: `4544aebb8c72b41af043efd97373dd0803e80a76`.
- Policy SHA256: `38975220e9441303878d2dfb9956d57e6416c6a906c99c43d9bca015b092fc51`; size2,338,751,333 bytes.
- Native format: `loopwam-s-robot-v1`. Original stateless v0, stored partition3/6/3, video4/action4, aligned observation KV, no history or memory.
- 584,567,886 policy parameters;14D joint action and14D proprioception.
- Epoch5 completed;237,310 optimizer updates. Training seed42, global batch128,50tasks,27,500 demonstrations,6,075,103 windows/epoch. This is the requested epoch5 release, not an epoch10 checkpoint.
- Native `create_loopwam_robot` loader; strict MoT/proprio state-dict load; finite policy weights; no LIBERO or ChronoLoop model loader.
- FP32 weights;BF16 inference autocast and VAE. No quantization or model architecture changes.

All nine release files were downloaded at the fixed HF revision and verified against API size and LFS SHA256 or Git blob hashes. The supplied dataset statistics are used for both actions and proprioception. Raw stats file SHA256 is `7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095`; canonical sorted-JSON SHA256 is `38ed9c8d466356fb8f4688da470b14a036dbc51e0fc5484cc5959b507adbcd48`, matching the training contract. These two hashes describe different encodings of the same verified statistics.

VAE SHA256: `38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`, verified against training metadata. The locally available Wan2.1 UMT5 checkpoint was also verified to be byte-identical to the UMT5 file published in Wan2.2-TI2V-5B. This avoids another11GB download without changing text weights.

## Source versions

The evaluation uses `/storage/anhdh35/LoopWAM-robotwin-eval`, branch `eval/robotwin-local-20261030`, based on `robotwin_eval_continue_20261030` commit `0f9e494`. Both that upstream branch and the main `LoopWAM_NT` branch were fetched/pulled during this task; both were already current on the second requested pull. Local adaptations are separately committed and archived with the report.

The released FastWAM simulator harness was copied exactly from `/storage/anhdh35/FastWAM-eval/third_party/RoboTwin/script/eval_policy.py`. The local checkout name/remote alone is not its provenance: the comparison uses the actual frozen source file. RoboTwin missing task configurations were restored from the vendored upstream commit `bf44be51cf5717a5595ce59447f2cf5263d2aa95`.

## Evaluation protocol

| Setting | Value |
| --- | --- |
| Tasks | All50 RoboTwin tasks listed in release protocol |
| Environments | `demo_clean` and `demo_randomized` |
| Accepted episodes |100per task per environment |
| Total |10,000 scored episodes,100task/environment cells |
| Evaluation seed |42 |
| First candidate scene seed |4,300,000 (`100000*(1+42)`) |
| Scene acceptance |Original expert planning AND success check; rejected candidates advance to next seed |
| Instruction split |Unseen |
| Prediction horizon |32actions |
| Replan interval |Execute24actions before next observation/inference |
| Denoising |10steps,CFG1 |
| Noise |CPU RNG, fixed seed42 on each policy query |
| Action command |Absolute14D joint targets, `qpos` |
| Termination |Original task-specific step cap and success predicate |
| Skipping observations |Only within an existing24-action plan, as enabled by FastWAM |
| Video |Every accepted scored episode; original camera montage and10FPS |

The evaluator walks candidate seeds normally. It does not use the optional seed-list bypass present in the training-server branch. A failed policy episode counts as a failure; an expert-rejected scene is not an accepted episode. Infrastructure interruptions during policy rollout fail the whole attempt. The initial total limit was three attempts per pair; recovery raised the total limit to five while reducing concurrency to four workers. Five missing pairs completed on attempt 4. All interrupted-attempt logs are retained and excluded from the scored denominator. Separately, the unchanged FastWAM harness catches expert/setup exceptions and skips those candidate scenes; the final report explicitly accounts for these events.

Clean and randomized settings follow the downloaded task YAMLs without reducing clutter, lighting changes, camera perturbations, or background variation. Task step limits are the pinned RoboTwin limits, not the LIBERO400/700 limits.

## Observation and text processing

Camera order is head, left wrist, right wrist. The evaluation image uses the exact released FastWAM adapter: PIL bilinear resize of uint8RGB to head320×256 and wrists160×128; wrists concatenated horizontally beneath the head; final384×320 image; cast to BF16 before multiplying by2/255 and subtracting1. An AST-based comparison against the reference implementation verifies exact tensor equality on random240×320 and480×640 inputs.

This differs from the training-server LoopWAM evaluator's two-stage torchvision floating-point resize. It is an explicit local evaluation change made to satisfy the requested FastWAM evaluation input procedure; the training code and weights are untouched.

Proprioception and actions use the checkpoint's global z-score statistics, with original action denormalization. No LIBERO gripper conversion is applied.

Text uses the original prompt template and UMT5 length128. A single shared GPU T5 server caches prompts and serves all policy workers. LoopWAM retains its trained true-token mask; released FastWAM internally makes a different all-ones mask choice after zero-padding embeddings. This is a model-internal distinction, not an assertion of identical cross-attention behavior.

The released FastWAM harness does not explicitly seed Python's instruction-template RNG. The local harness preserves that behavior; actual chosen instructions and scene seeds are logged per episode. The uploaded H100 evaluator instead seeds Python RNG from each scene seed. Consequently, equal evaluation seed42 does not guarantee identical instruction strings across the two evaluators or independent reruns.

## Rendering and environment compatibility

The GPU is an NVIDIA RTX PRO6000 Blackwell Max-Q96GB. Rendering uses the released FastWAM settings: `rt` shader,32samples/pixel, path depth8, OIDN denoising. The H100 release protocol used raster `default`; that fallback is not used here.

SAPIEN3.0.0b1 bundles OIDN2.0.1, which reported an unsupported CUDA device on this Blackwell GPU. The isolated environment uses official OIDN2.3.3 instead. A fresh renderer test completed with finite images and no OIDN errors. This retains the RT settings but is **not a claim of pixel-identical rendering to OIDN2.0.1**. Official source: https://github.com/RenderKit/oidn/releases/tag/v2.3.3 .

The isolated environment is `/data14t/anhdh35/loopwam_robotwin_v0_20261030/env`; it inherits the existing PyTorch2.7.1+cu128 base packages without modifying them. Installed simulator dependencies include SAPIEN3.0.0b1, MPLib0.2.1, CuRobo source tagv0.7.8 compiled forSM120, Warp1.12.1, Open3D0.18.0 and Trimesh4.4.3. CuRobo archive metadata may display package version0.0.0 because it was installed from a source tarball; the actual source tag/archive is recorded. The official RoboTwin MPLib collision-condition installation adjustment was applied inside this isolated environment only. Dependency checks passed.

The optional PyTorch3D package is not used by the RGB/qpos benchmark; point-cloud collection is disabled in the official configs.

Official object, embodiment and background assets were pinned to `TianxingChen/RoboTwin2.0` revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`, downloaded and verified before extraction. The runtime manifest hashes21,132 asset/dependency files and approximately28.7GB of content, including T5/VAE weights and environment records. Runtime code, language templates, normalizers, tokenizer files and launch protocol are additionally bound by each evaluation manifest.

## Preflight and speed selection

Four native smoke episodes—adjust_bottle and click_bell, each clean and randomized—completed successfully, with decoded video validation. They are excluded from all scored results.

The CUDA-graph gate compared eight captured real observations in forward and reverse order with varying fixed noise seeds42/43/44. All outputs were finite32×14 tensors and matched native inference exactly (`atol=rtol=0`). Isolated mean query timings were native0.1644seconds and graph0.1300seconds, about20.9% lower query latency. These timings are not end-to-end episode throughput.

Concurrency calibration tested two, four and eight workers on a separate sixteen-episode workload. Eight workers were fastest on that short workload, but were not memory-safe over the full campaign. Calibration uses the original unseeded instruction generation, so language draws can differ; its SR is not a scientific model comparison. Concurrency measurements include task startup costs and are approximate predictors of full-run throughput. The original calibration is retained in `calibration.json`; `initial_launch_snapshot.json` and `initial_run_final_state.json` describe the initial eight-worker run. The current `launch.json` records completed four-worker recovery.

No NFE, episode count, expert acceptance check, task limit, RT quality setting or precision is reduced for speed. Shared T5 and existing observation skipping reduce redundant work. CUDA graphs replay unchanged kernels and are enabled only after the real-input equality gate.

## Logs, resume and reporting

Bulk outputs: `/data14t/anhdh35/loopwam_robotwin_v0_20261030`.
Report directory: `/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin_v0`.

Each task/setting has attempt directories with the exact command, full simulator log, actual instructions, per-query inference timings/action hashes, and per-episode videos. Completed-cell records include scene seeds, success values, video hashes and validation, instruction strings, query counts and full trace digests. Only a complete100-episode cell can be resumed; partial attempts are retained and replayed. Resume rejects mismatched manifests or changed completed artifacts. The full benchmark must have all100cells before `COMPLETE.json` is emitted.

`report.md` and `progress.json` track completed and active cells. Partial SR uses only recorded scored episodes and is provisional. Clean and randomized final SRs each use5,000episodes; overall SR uses10,000. Smoke and calibration outputs are in separate directories.

The persistent tmux controller updated the report during execution and stopped after completion. Its first eight-worker run exited with five pairs unresolved; the recovery controller finished them with four workers. There are no pending jobs. The final report uses observed controller timestamps rather than earlier estimates.

## Selected configuration

Calibration completed: 2 workers in 281.14 s, 4 workers in 185.75 s, and 8 workers in 165.67 s for 16 unscored episodes each. Eight CUDA-graph workers were initially selected, with measured calibration peak GPU usage of 67,168 MiB and approximately 1.70× the two-worker throughput on that workload. During full evaluation, eight workers repeatedly exhausted GPU memory. The four-worker recovery completed without new interruptions. Calibration speed and memory measurements must not be generalized to the full campaign. Runtime manifest environment metadata was refreshed after removing an unused build-only dependency; simulator/model libraries were unchanged between calibration and launch.


## Recovery on October 31, 2026

The eight-worker campaign stopped with 95/100 task/setting pairs complete. Recovery completed all five missing pairs with four workers in 63 minutes 58 seconds, including revalidation. All 95 existing result files remain byte-identical; the original evaluation contract is unchanged. See [recovery details](recovery_20261031.md), [recovery metadata](recovery_20261031.json), and the [final report](report.md). The initial short calibration underestimated the full-run GPU memory requirement.


## Post-completion source update and audit

Main repository was fast-forwarded to `43ef539b4744ebfec1bb7b3864904f3d48465903` on October 31. Its 126-file update contains report synchronization only. The completed run remains attributed to local evaluator commit `0a0c6e57c56db40d3ff244b6857621874cbc38c9`, whose worktree and manifest-bound files remain unchanged. The fresh [artifact audit](final_audit.json) verified all 10,000 episode videos and traces, 782 source/manifest-bound files, 21,132 runtime files, and the 95 preserved results. The original 26 preflight tests remain historical launch evidence.
