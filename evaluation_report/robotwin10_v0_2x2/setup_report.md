# RoboTwin-10 v0 2/2 — evaluation setup

Requested release: https://huggingface.co/anhdao69/LoopWAM_NT/tree/34af15d8b3fa65c1938c48e806599df29da848d2/robotwin-10/v0-video2-action2

Native epoch5/update45,445; 584,567,886 policy parameters; 12 stored block pairs, partition3/6/3; two video loops and two action loops, aligned late KV, no memory/history. Checkpoint SHA256: `aa835dce6898f92d49275bf6e6b4e8f9202d4a8025a1e810ae3158867369be59`. All nine recorded training source hashes match frozen evaluator commit `7a2181d0ecc8e30d64806e94535b347bc9e1da2e`.

## Scored protocol

All ten release tasks: adjust_bottle, grab_roller, hanging_mug, move_stapler_pad, open_microwave, press_stapler, scan_object, stack_blocks_two, stamp_seal, turn_switch. Clean and randomized; 100 accepted episodes/task/setting; **2,000 total**, **seed42**, **NFE10**, CFG1, predict32/execute24. Original CPU fixed-per-query action noise, checkpoint z-score normalization, absolute14-Dqpos, original FastWAM PIL uint8/BF16 three-camera packing at384×320. Original task horizons, expert plan/success scene acceptance, unseen instructions, RT32/depth8/OIDN2.3.3 rendering.

Python instruction-template RNG remains as released (not explicitly seeded); actual scene seeds, instructions and action query hashes are recorded. Therefore independent model runs do not guarantee exactly paired instruction strings. Candidate scene seeds start4,300,000 before expert acceptance skips.

## Verification and automatic queue

The queue downloads all ten release files and verifies pinned HF LFS SHA256/Git blob SHA1 and SHA256SUMS. Strict native loading checks checkpoint metadata, finite weights, parameter count, video loops2/action loops2, source hashes, normalization and VAE. It runs a new native-versus-CUDA-graph gate on eight real observations, seeds42/43/44, forward/reverse order, requiring exact action equality. All38 existing evaluator/model tests passed during setup.

Unscored simulator gates: adjust_bottle and grab_roller, clean/randomized, two episodes per pair. Native one-worker baseline and graph profiles with1,2,4 workers when memory allows. Select the fastest successful graph profile with at least8,000MiB total-memory headroom. Full evaluation also requires measured extra memory plus10,000MiB or the profile admission threshold, whichever is larger. Other GPU jobs are left untouched. One shared T5 server serves all workers. No NFE/precision/renderer changes are used to accelerate scoring.

Only full100-episode completed cells enter final scoring. Calibration and interrupted-attempt partial episodes are excluded; max3 attempts/cell and1,800-second no-progress timeout. All videos, actual instructions, scene seeds and query hashes are retained. Final artifact audit checks2,000 videos, counters, source/runtime hashes and checkpoint integrity.

Bulk: `/data14t/anhdh35/loopwam_robotwin10_v0_2x2_20261104`. Tmux: `robotwin10_v0_2x2`. [Queue status](queue_status.json), [live report](report.md), and [final report](final_report.md) when generated. Tmux survives client disconnects, not host reboot.

Final tables compare v0 2/2 with v0 4/2, original v0 4/4, and dense-s30 using **seed42 on every task**. Seed43 reruns and post-hoc mixed-seed alternatives are excluded.

## Run-specific validation status

All 38 evaluator tests passed freshly (3.70 seconds, one Transformers deprecation warning). The frozen evaluator explicitly supports native v0-video2-action2, and all nine release training-source hashes match. Download, strict-load, runtime integrity, CUDA-graph and simulator calibration results will be recorded separately; passing test evidence does not imply those later gates have passed.

The completed local 4/2 run is the queue predecessor and passed its final audit. The 4/4 rerun continues independently on vmo_aLinh. The local GPU initially has approximately61,606MiB free; a separate GPU workload remains active. Calibration checks real available memory before each profile.

## Verified model and graph gates

All ten release files passed pinned-revision integrity verification. Strict loading verified native video/action loops2/2, epoch5/update45,445, finite FP32 weights, 584,567,886 policy parameters, 12 stored block pairs and matching normalization/VAE. All 21,150 bound runtime/source files were checked against the established run.

Eight saved real observations passed exact native-versus-CUDA-graph action equality (absolute/relative tolerance zero, seeds42/43/44, forward/reverse order). Native latency: 0.116150s/query; CUDA graph: 0.083888s/query; peak reserved GPU memory during this model-only test: 3948MiB. This excludes simulator/rendering time. Simulator calibration and scored launch remain tracked in queue_status.json.

## Completed speed calibration

All 32 unscored simulator episodes completed and succeeded, with no failed calibration profiles. Native one-worker:235.66s/8 episodes. CUDA graph: one worker235.67s, two workers151.12s, four workers102.81s. The queue selected four workers, the fastest tested profile. These timings include startup and simulator overhead; they are not a direct full-benchmark runtime estimate.

The four-worker profile peaked at81,985MiB total GPU usage, including the pre-existing35,642MiB baseline. Incremental peak46,343MiB; observed total-memory headroom15,902MiB. Full scoring was admitted after three stable samples with at least60,000MiB free. All workers share one T5 server.

The final preflight passed all38 tests, nine training-source hashes, strict checkpoint loading, exact CUDA-graph action equality and all32 unscored episodes. See [preflight](preflight.json), [calibration](calibration.json), [graph validation](graph_gate.json), and [live report](report.md).
