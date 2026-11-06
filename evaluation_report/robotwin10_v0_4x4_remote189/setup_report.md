# vmo_aLinh — RoboTwin-10 v0 4/4, seed42

Host: `vm-perception-ws@10.148.183.189` (vmo-linhbn7-d). This runs the original RoboTwin-10 v0 4/4 epoch5 checkpoint on all ten tasks. The user authorized stopping the still-waiting duplicate on10.148.182.137 after this server begins scoring. The separate local v0 4/2 run remains active.

## Protocol and checkpoint

Checkpoint: `robotwin-10/v0-video4-action4/policy.pt`; HF revision `df3d575d3f3fde3cf1b8d5f609fab93dc6a95649`; SHA256 `bb1563f16bcd102b3f4ad4a1a14377ee49de7e461d862b7bbc5dba497f036127`. Epoch5/update45,445;584,567,886 policy parameters;12 stored block pairs, partition3/6/3; native video/action loops4/4; aligned late KV; no memory/history. Same unmodified checkpoint as the original local evaluation.

Ten tasks: adjust_bottle, grab_roller, hanging_mug, move_stapler_pad, open_microwave, press_stapler, scan_object, stack_blocks_two, stamp_seal, turn_switch. Seed42 only; clean/randomized;100 accepted episodes per task/setting;2,000 scored episodes. NFE10;CFG1;predict32/execute24;fixed-per-query CPU action noise seeded42;FP32 weights and BF16 compute;absolute14-Dqpos and checkpoint z-score normalization. Original FastWAM PIL uint8 three-camera packing384×320, task horizons, expert scene acceptance, unseen instructions, and RT32/depth8/OIDN2.3.3 rendering.

Candidate scene seeds start4,300,000 before expert skips. The original Python instruction-template sampler is not explicitly seeded, so seed42 does not guarantee exact instruction pairing across independent runs. Actual instructions, scene seeds, videos and action-query hashes are recorded. No seed43 substitutions are used.

## Environment and integrity

Frozen evaluator commit `7a2181d0ecc8e30d64806e94535b347bc9e1da2e`, transferred by git bundle. Model/harness source unchanged. All ten release files are verified against pinned HF LFS SHA256/Git blob SHA1 and release SHA256SUMS.

The established Python3.10.22, PyTorch2.7.1+cu128 environment was copied using a hash-verified conda-pack archive plus the original RoboTwin overlay. The pre-existing host PyTorch2.11 environment is not used for scoring. GPU model is RTX PRO5000 Blackwell48GB (compute capability12.0), driver570.211.01 and CUDA toolkit12.8. The hardware differs from the original RTX PRO6000 run; native/graph equality is tested freshly here, without claiming cross-hardware bitwise episode reproducibility.

28,172 portable source/runtime paths are verified. Six robot-planning YAML files require only absolute URDF/collision-sphere path relocation; original/relocated hashes are recorded. Interpreter/editable-package paths are adjusted for this isolated workspace. One test-only absolute FastWAM reference path is relocated to a byte-identical reference file; test assertions remain unchanged. All38 existing evaluation/model tests must pass. Dependency checks, strict native model loading and native-versus-CUDA-graph equivalence are required gates.

## Speed and simulator validation

The host had no compute jobs at setup and about47GB GPU memory free. Calibration tests native one-worker inference and CUDA-graph profiles with one and two workers. Each profile runs adjust_bottle/grab_roller, clean/randomized, two episodes per pair (8 unscored episodes). Choose the fastest successful graph profile with at least8,000MiB measured memory margin. Four workers are excluded for this48GB device. A single shared T5 server serves all scored workers. Minimum free memory before simulator calibration is30,000MiB for one worker and42,000 for two; full admission also requires measured additional peak plus10,000MiB or the profile threshold, whichever is greater.

CUDA graph gate: eight real observations, seeds42/43/44, forward and reverse order; require exact native/graph action equality. Simulator calibration and interrupted-attempt partial episodes are excluded from scoring. Max3 attempts per scored cell,1,800-second no-progress timeout. Validation failures are logged and stop progression rather than silently changing the protocol.

## Storage, queue and reporting

Workspace: `/media/newdisk/loopwam_robotwin10_v0_4x4_20261104` (secondary disk, initially2.5TB free). Remote tmux session: `robotwin10_v0_4x4`. It survives SSH disconnects, not host reboot.

Remote report directory: `repo/evaluation_report/robotwin10_v0_4x4_remote189` beneath the workspace. Local mirror: `/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_v0_4x4_remote189`. Videos remain on the remote secondary disk. [Queue status](queue_status.json), [live report](report.md), [final report](final_report.md) once generated.

After all20 cells complete, the auditor verifies2,000 videos, instructions/action traces, result counters, checkpoint and bound runtime/source hashes. It writes final_report.md, final_results.json, task_results.csv, episodes.csv, failed_attempts.json and final_audit.json. Final tables compare this full seed42 run with the original local v04/4 and dense-s30 results.

## Verified launch and handoff

All 38 evaluator tests, checkpoint strict load, runtime integrity checks and native/CUDA-graph exact action comparisons passed. All 24 unscored calibration episodes passed. Selected two CUDA-graph workers: 145.22 seconds/8 episodes versus 229.89 seconds with one worker; peak GPU memory 29,871 MiB and observed headroom 19,064 MiB. These short profiles include startup and do not establish a full-run ETA. Actual profiles are native1, graph1 and graph2; the inherited max4 phrase in calibration.json is stale descriptive text, and no four-worker profile was run.

Scored seed42 evaluation has started. The waiting duplicate on server137 was cancelled after the new full-run process and both scored workers were verified active. No scored episodes had run on server137; all of its artifacts are preserved. See [handoff evidence](handoff.json). Local v0 4/2 remains independent. Remote189 clock is about 210.93 seconds behind the local server; [clock measurement](clock_check.json) records the offset.
