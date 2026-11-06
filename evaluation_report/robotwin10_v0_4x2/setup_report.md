# RoboTwin-10 v0 4/2 — evaluation setup

Requested release: https://huggingface.co/anhdao69/LoopWAM_NT/tree/b24d5ed114c5d812284472c674bee31470a8d8ff/robotwin-10/v0-video4-action2

Native epoch5/update45,445; 584,567,886 policy parameters; 12 stored block pairs, partition3/6/3; four video loops and two action loops, aligned late KV, no memory/history. Checkpoint SHA256: `fea2ce45c09197fe68df9345d1f60e520dab8193dd7ee9b5147001e64efc32c9`. All nine recorded training source hashes match frozen evaluator commit `7a2181d0ecc8e30d64806e94535b347bc9e1da2e`.

## Scored protocol

All ten release tasks: adjust_bottle, grab_roller, hanging_mug, move_stapler_pad, open_microwave, press_stapler, scan_object, stack_blocks_two, stamp_seal, turn_switch. Clean and randomized; 100 accepted episodes/task/setting; **2,000 total**, **seed42**, **NFE10**, CFG1, predict32/execute24. Original CPU fixed-per-query action noise, checkpoint z-score normalization, absolute14-Dqpos, original FastWAM PIL uint8/BF16 three-camera packing at384×320. Original task horizons, expert plan/success scene acceptance, unseen instructions, RT32/depth8/OIDN2.3.3 rendering.

Python instruction-template RNG remains as released (not explicitly seeded); actual scene seeds, instructions and action query hashes are recorded. Therefore independent model runs do not guarantee exactly paired instruction strings. Candidate scene seeds start4,300,000 before expert acceptance skips.

## Verification and automatic queue

The queue downloads all ten release files and verifies pinned HF LFS SHA256/Git blob SHA1 and SHA256SUMS. Strict native loading checks checkpoint metadata, finite weights, parameter count, video loops4/action loops2, source hashes, normalization and VAE. It runs a new native-versus-CUDA-graph gate on eight real observations, seeds42/43/44, forward/reverse order, requiring exact action equality. All38 existing evaluator/model tests passed during setup.

Unscored simulator gates: adjust_bottle and grab_roller, clean/randomized, two episodes per pair. Native one-worker baseline and graph profiles with1,2,4 workers when memory allows. Select the fastest successful graph profile with at least8,000MiB total-memory headroom. Full evaluation also requires measured extra memory plus10,000MiB or the profile admission threshold, whichever is larger. Other GPU jobs are left untouched. One shared T5 server serves all workers. No NFE/precision/renderer changes are used to accelerate scoring.

Only full100-episode completed cells enter final scoring. Calibration and interrupted-attempt partial episodes are excluded; max3 attempts/cell and1,800-second no-progress timeout. All videos, actual instructions, scene seeds and query hashes are retained. Final artifact audit checks2,000 videos, counters, source/runtime hashes and checkpoint integrity.

Bulk: `/data14t/anhdh35/loopwam_robotwin10_v0_4x2_20261104`. Tmux: `robotwin10_v0_4x2`. [Queue status](queue_status.json), [live report](report.md), and [final report](final_report.md) when generated. Tmux survives client disconnects, not host reboot.

Final tables compare v0 4/2 with v0 4/4 and dense-s30 using **seed42 on every task**. Seed43 reruns and post-hoc mixed-seed alternatives are excluded.

## Verified admission

All ten release files passed download integrity checks. Native strict load passed with video loops4/action loops2, correct584,567,886 parameters, finite weights and matching VAE/stats. The real-input graph gate passed exact equality; native 0.1247s/query versus graph 0.0862s/query. Simulator calibration is separate and remains governed by queue status.
