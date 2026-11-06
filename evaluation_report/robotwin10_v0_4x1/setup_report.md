# RoboTwin-10 v0 4/1 — evaluation setup

## Release and model

Requested release: https://huggingface.co/anhdao69/LoopWAM_NT/tree/c0a56a65986af8c0f0beed4f17b4e75ad3cbad81/robotwin-10/v0-video4-action1

Pinned revision: `c0a56a65986af8c0f0beed4f17b4e75ad3cbad81`. Checkpoint SHA256: `b0fd9da51d9ff5ab512d5729de6565e85d240a10f078ddf9f803733a77aa757d`;2,338,751,909 bytes. Native epoch5/update45,445;584,567,886 policy parameters;12 stored block pairs, partition3/6/3;four video core loops and one action core loop;aligned late action KV;no history or memory. The model is loaded natively from its own checkpoint; no loop override of another trained model is used.

All nine recorded training-source hashes match frozen evaluator commit `7a2181d0ecc8e30d64806e94535b347bc9e1da2e`, both locally and on the evaluation server. The evaluator and model source remain unchanged. The training recipe, tasks, normalization and VAE are checked against the completed dense-s30 release.

## Scored evaluation protocol

Ten tasks: adjust_bottle, grab_roller, hanging_mug, move_stapler_pad, open_microwave, press_stapler, scan_object, stack_blocks_two, stamp_seal, turn_switch.

**Seed42;NFE10;100 accepted episodes per task/setting;clean and randomized;2,000 scored episodes.** CFG1;predict32 actions and execute24 before replanning;14-D absolute joint positions and checkpoint global z-score normalization. CPU action noise resets to seed42 for every policy query, following the established FastWAM protocol. FP32 model weights and BF16 inference compute;original PIL uint8 three-camera packing at384×320. Unseen instructions;original task horizons and expert scene-plan/success acceptance. Rendering:RT32/depth8/OIDN2.3.3, matching previous runs.

Candidate scene seeds start4,300,000 and advance past rejected expert scenes. Original Python instruction-template sampling is not explicitly seeded: actual instructions are logged, and seed42 does not imply exactly paired instruction strings between model runs. No seed43 substitution or best-seed selection is used.

## Host, environment and isolation

Host:`vm-perception-ws@10.148.183.189` (vmo_aLinh), RTX PRO5000 Blackwell48GB. Approximately47GB free and no compute workloads at inspection. The local RTX PRO6000 is busy with other evaluations, so this run uses the available validated remote host. The local root filesystem is full; downloads and staging use `/data14t`, and scored data uses the remote secondary disk with about2.4TB free.

Bulk run:`/media/newdisk/loopwam_robotwin10_v0_4x1_20261105`. Reused frozen code/environment/assets:`/media/newdisk/loopwam_robotwin10_v0_4x4_20261104`. The previous4/4 evaluation passed its full2,000-episode audit. The same Python3.10.22/PyTorch2.7.1+cu128 environment, SAPIEN/mplib/cuRobo and OIDN binaries are reused. Six absolute robot-planning asset paths were relocated during the original server setup; their recorded hashes are rechecked. No new asset relocation is needed.

A test-only reference path was relocated during the earlier setup; the portable test and byte-identical FastWAM reference are reused with their provenance. Fresh38-test execution is required before this run. New checkpoints, launch scripts, logs, profiles, videos and results are in the isolated run directory. Other workloads and previous run files are not changed.

## Validation and speed selection

Verify all ten release files against pinned HF LFS SHA256/Git blob SHA1 and release SHA256SUMS. Strict loading must validate native4/1 loops, epoch/update, training contract,584,567,886 finite model parameters,12 stored blocks per expert, normalization and VAE hashes. Recheck the portable runtime/source manifest before graph validation and full launch.

Native versus CUDA-graph comparison uses eight saved real observations, seeds42/43/44, forward and reverse order, requiring exact action equality (atol=rtol=0). Model-only latency is recorded separately from simulator throughput.

Unscored simulator calibration:adjust_bottle and grab_roller, clean/randomized, two episodes per task/setting. Profiles:native1,CUDA-graph1 and CUDA-graph2 workers (24 episodes total). Select the fastest passing graph profile with at least8,000MiB observed memory headroom. Full-run admission also checks available GPU memory against measured extra usage plus10,000MiB, or the profile threshold, whichever is larger. One shared T5 encoder serves the scored workers. Maximum two workers follows the previously validated48GB deployment configuration. Results support fastest tested concurrency, not a claim of a global hardware throughput optimum.

## Queue, artifacts and comparisons

Remote tmux:`robotwin10_v0_4x1`. The queue survives SSH disconnects, not a host reboot. Each cell has at most3 attempts and a1,800-second no-progress timeout. Calibration and interrupted-attempt partial episodes are excluded from scoring. Full completed100-episode cells provide final results.

Every episode records actual scene seed, instruction, success, action-query hashes and video. The final audit checks all2,000 videos, counters, checkpoint and bound source/runtime hashes before publishing final Markdown/CSV/JSON tables. Baselines: dense-s30, original4/4, remote4/4 rerun,4/2 and2/2, all full seed42 runs. Protocol fields are checked against the dense baseline. Cross-hardware results and unseeded instruction sampling are disclosed; exact episode pairing is not assumed.

Local report mirror:`/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_v0_4x1`. Reports sync every60 seconds while the queue runs. Videos remain on the remote secondary disk.

[Queue status](queue_status.json) · [Live report](report.md) · [Final report](final_report.md) when generated · [Execution plan](execution_plan.md)

## Verified model admission

All ten release files passed pinned HF metadata and SHA256SUMS checks on the evaluation server. All38 fresh tests passed (3.41 seconds), and pip check found no broken requirements. Native strict load confirms4video/1action loops, epoch5/update45,445,584,567,886 finite parameters, and matching stats/VAE. Runtime integrity checks passed. Eight real-input native/CUDA-graph comparisons passed with exact equality; measured native0.087898s/query versus graph0.075475s/query, model-only peak reserved4,062MiB. Calibration and full scoring remain tracked separately in queue_status.json.

## Completed simulator calibration

All24 unscored episodes passed with24 successes and no failed profiles. Native1:251.08seconds;CUDA-graph1:287.50seconds;CUDA-graph2:169.54seconds (each8 episodes). The two-worker profile was selected. Short profiles include cold model/encoder startup and simulator overhead; lower model-only query latency did not imply lower one-worker wall time. No timing claim assumes identically paired unseeded instructions.

Selected profile peak GPU use29,937MiB, baseline918MiB, incremental29,019MiB, total-memory margin18,998MiB. Full admission requires42,000MiB free over three samples15seconds apart. This configuration is the fastest tested among one/two workers on this48GB GPU.

All38 tests,9 training-source hashes, strict native4/1 loading, runtime integrity, exact graph-action equivalence and24 simulator checks passed. See [preflight](preflight.json) and [calibration](calibration.json).
