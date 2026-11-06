# LoopWAM v0 epoch 6 — RoboTwin evaluation setup

Preflight passed: checkpoint verified, strict CPU load passed, native/CUDA-graph gate passed, and four simulator smoke episodes completed. The live report records the actual benchmark status.

## Checkpoint

Requested release: https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/robotwin/v0-video4-action4/epoch_06

Pinned HF revision: `2fbf6b7d2bdf91bf03a0062d177f516fd02af08f`. Policy SHA256: `6bf8ca2cfaf3d0f1f8ab012f25dba46298706f69ca8b553031410f9f450b14f0`. Native policy size: 2,338,751,589 bytes. All eight downloaded release files are checked against HF metadata (LFS SHA256 or Git blob SHA1), plus the release SHA256SUMS file.

Completed epoch 6: update 284,772; 36,450,618 training window presentations; next epoch cursor 6 and next batch 0. The training contract schedules ten epochs, but these weights are specifically the completed epoch-6 export. Original stateless v0, partition 3/6/3, video/action loops 4/4, aligned KV, 14D action and proprioception, no memory/history, 584,567,886 policy parameters. FP32 weights and BF16 inference. Training metadata says continuation from epoch 5 with restored training state and a lower learning-rate schedule; the inference export itself does not contain Adam state.

## Matched local evaluation protocol

- 50 RoboTwin tasks; `demo_clean` and `demo_randomized`.
- 100 accepted episodes per task/setting: 5,000 clean + 5,000 randomized = 10,000 total.
- Evaluation seed 42; candidate scene seeds start at 4,300,000; unchanged expert planning/success acceptance checks.
- Unseen instructions; actual instruction strings saved. Python instruction-template RNG remains unseeded as in the epoch-5 FastWAM harness, so seed 42 does not imply identical instructions across runs.
- NFE 10, CFG 1, fixed CPU noise seed 42 per query; predict 32 actions, execute 24 before replanning.
- Absolute 14D `qpos` commands; checkpoint z-score normalization; native strict LoopWAM robot loader.
- Same PIL uint8/BF16 image packing, head and both wrists, as the completed epoch-5 local run.
- Same RT renderer, 32 samples/pixel, path depth 8, OIDN 2.3.3 Blackwell compatibility libraries.
- Four CUDA-graph workers and shared T5 on the RTX PRO 6000 Blackwell Max-Q 96 GB. Four workers were chosen from the successful epoch-5 recovery, not a new epoch-6 calibration.
- Full video, scene seed, instruction and per-query action hash logging. Every completed pair requires 100 valid episode records and verified video hashes. Up to three total attempts per pair; no-progress timeout 1,800 seconds. Partial interrupted attempts are retained and excluded from SR.

The checkpoint's epoch-6 folder has no supplied evaluation_protocol.json. The local checkpoint-directory file of that name is explicitly generated from the completed epoch-5 local protocol, with epoch 6 and expected step 284,772. It is not presented as a downloaded release artifact.

## Validation gates

The launch requires successful strict CPU model loading and epoch/update/normalization/VAE/finite-weight checks, a native-versus-CUDA-graph exact-output gate using eight saved real simulator observations in forward and reverse order, and four unscored smoke episodes (adjust_bottle and click_bell, each clean and randomized). The smoke runner validates outcomes/videos and traces; smoke successes are not added to benchmark results. The gate inputs were captured during epoch-5 setup and are intentionally reused to test the epoch-6 policy under the same observation shapes. Their hashes and the new checkpoint hash are recorded.

Targeted evaluator tests passed: 18 tests. Gate results and launch metadata are copied here after execution. A native-versus-graph gate pass is a check on those observations, not proof of equivalence on all possible inputs.

The frozen evaluator worktree remains `/storage/anhdh35/LoopWAM-robotwin-eval`, commit `0a0c6e57c56db40d3ff244b6857621874cbc38c9`. The release's declared model runtime revision `8326632` has no differences in the reviewed model/loader files versus the evaluator base. Only external launcher/report paths and the expected checkpoint step change; the model, simulator and scoring code remain frozen. Runtime hashes are checked before launch.

## Comparison and artifacts

Epoch-5 baseline: clean 4,580/5,000 (91.60%); randomized 4,593/5,000 (91.86%); combined 9,173/10,000 (91.73%). See [epoch-5 final report](../robotwin_v0/report.md). A future epoch comparison uses the same aggregate protocol, but is not a guaranteed episode-paired comparison because instruction draws and accepted scene sets can differ.

Bulk data: `/data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031`. Live report: `report.md`; machine-readable progress: `progress.json`. Checkpoint/configuration evidence, the graph gate, CPU validation, smoke results and controller launch metadata accompany the report. Original epoch-5 outputs are preserved separately.


## Host interruption and resume — November 1

At 11:45 Vietnam time, the controller and workers were absent and the machine had rebooted. Last saved evaluator summary: 08:59:13. Exact shutdown cause is not established. The queue was relaunched with the same four-worker protocol and existing resume checks. All 76 completed pairs are preserved and revalidated; four interrupted pairs are replayed from the beginning, and their abandoned partial outcomes are excluded. See [resume metadata](resume_20261101.json) for archived pre-resume launch/summary records. The original launch timestamp is preserved in that archive; subsequent launch.json start time describes the resumed process.
