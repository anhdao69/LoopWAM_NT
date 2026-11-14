# RoboTwin-10 epoch10 — continuation on aLinh

> **November 14 resume:** Dense-S12 is complete (77.25%). Only v0 1/4 epoch10 is resumed, with 800 verified complete episodes and 1,200 remaining. [Resume accounting and preserved stop history](resume_20261114.md).

The local queue was stopped at the user's request. Remaining order: **Dense-S12 → v0 1/4**, both epoch10, update90,890. Execution host is aLinh (`vm-perception-ws@10.148.183.189`, hostname `vmo-linhbn7-d`), with a48GB RTXPRO5000 Blackwell. The local RTXPRO6000 is left free. Existing 4/2 and2/2 results remain unchanged and audited:82.40% and78.90%.

## Protocol and hardware

Seed42 only;10tasks;clean andrandomized;100acceptedepisodes per task/setting;2,000/model. NFE10,CFG1,predict32/execute24,absolute14Dactions/proprioception,FP32weights/BF16compute. Same native model/evaluator, original FastWAM PIL/uint8 image handling, fixed CPU noise seed42 per query, unseen instructions, unseeded Python instruction-template RNG, expert admission and task horizons. RT32/depth8/OIDN2.3.3 rendering is preserved. Across independent runs, equal seed42 does not imply exactly paired instructions or trajectories.

The native repo is copied byte-for-byte from the frozen local evaluator;1,924 regular source/fixture files per model are bound by their original SHA256. Active env, checkpoint, simulator asset and policy symlinks are bound to the verified remote paths and checked before execution and at final audit. The remote inherited runtime has28,196 bound files. Of20,847 simulator asset files compared across hosts,20,841 are byte-identical; six Curobo YAMLs differ only by previously recorded absolute-path relocation, verified against original and relocated hashes. Model and deployed policy/harness source hashes match the local baseline. Release metadata, allcheckpoint bytes, normalization andVAE are independently verified remotely.

Two workers plus one sharedT5 service are used on this48GBGPU, matching the host's previously successful native runs. No precision, denoising-step, task or rendering-quality reduction is used. CUDA graphs require a fresh bit-exact native-versus-graph check on eight real observations. Eachmodel runs16unscored smoke episodes with two workers and must retain at least8,000MiB observed memory margin. These smoke episodes are excluded from SR.

## GPU admission and automatic queue

Before CUDA model initialization and immediately before starting the native evaluator's T5 GPU service, admission requires no compute processes, at most1,024MiB used and at most1% GPU utilization continuously for60seconds. Query failures and unparseable GPU state are treated as unavailable. Checks are repeated after expensive CPU/episode validation, so a stale earlier idle observation cannot authorize GPU startup. The unchanged native evaluator runs through a small deployment wrapper; model mathematics and simulator/scoring code are unchanged. Another user's job is never stopped.

Eachrun and the outer queue has a lock. Exact inherited ownership tokens andPID-start identities support cleanup even when a supervisor dies; stale owned descendants prevent duplicate launches. A model advances only after full final artifact audit. The queue runs independently of the chat connection. Reports mirror to the local evaluation_report folder every30seconds; temporary SSH failures are retried and do not stop remote evaluation.

## Dense-S12 completed-cell transfer

Four100-episode cells are reused:adjust_bottle/clean100successes,adjust_bottle/randomized99,grab_roller/clean100,grab_roller/randomized100. Thus400episodes and399successes were already complete locally. Every video, scene seed, instruction, query trace, checkpoint hash and counter was validated locally, then again against transferred bytes on aLinh. Only metadata paths for directories/videos change; original result metadata, local protocol/runtime manifest, validation evidence and exact relocation receipts are preserved under origin/ and carry_receipt.json. All17invariant protocol fields are checked, including epoch-update, instruction type, CUDA-graph and compile settings.

The native evaluator generates a new remote runtime manifest, revalidates these complete cells, then skips them and scores the remaining16cells (1,600episodes). The final Dense-S12 result combines400episodes generated on the local RTXPRO6000 with1,600generated on aLinh RTXPRO5000 under the same protocol. This mixed-host provenance is explicitly recorded in the final audit/report. It is not presented as2,000episodes generated entirely on one GPU.

Previously interrupted local attempts remain local, are preserved without deletion, and are excluded from the official score. Their metadata and logged counts are included in the remote final failed/interrupted-attempt report. Partial progress therefore starts from400, not the interrupted638-episode live total.

## Validation and reporting

Both checkpoint variants passed strict CPU loading, expected loop counts/epoch/update, finiteFP32weights, normalization andVAE checks on aLinh. The38native tests passed for eachmodel. Four supervisor/source tests per model and renderer guards passed. Migration regression tests cover partial/wrongcheckpoint rejection, path escape, unchanged episode fields, corrupttrace rejection, allprotocol-field mutations, symlink retargeting, strict idle parsing and service-admission ordering. Independent review findings were fixed before launch; fresh runtime/GPU/smoke validation remains an execution gate.

Official final outputs include final_results.json, final_report.md, task_results.csv, episodes.csv, protocol_comparison.json, failed_attempts.json, final_audit.json andCOMPLETE.json. Final audit covers20task/settings and2,000videos/episode traces per model, including imported artifacts. Raw remote timestamps are approximately203.46seconds behind local wall time at setup; clock_offset.json records the measured offset. Durations use each host's own clock; combined status refresh uses local time. Local stop/resume histories are retained, so remote elapsed time is not claimed as total end-to-end campaign wall time.

Interrupted local attempts: 8; excluded logged episodes: 266.

## Checkpoints

Pinned HF revision:b72f355505d98c473fb91e45d10e16d4cc7ce536.

- dense-s12 epoch10, video/actionloops 1/1; policySHA256 `a222704c10688bf42a531a4a4175fe45a1b172c437cf9b9efa435c4b4fae1ac8`.
- v0-video1-action4 epoch10, video/actionloops 1/4; policySHA256 `bd15ddb617cf78014582deb4787890d05c6519e2017a0c0204a9db553ec442d6`.

## Locations

- Remote campaign: `/media/newdisk/loopwam_robotwin10_epoch10_queue_20261113`
- Local migration evidence: `/data14t/anhdh35/loopwam_robotwin10_epoch10_queue_20261113/migration_alinh_20261113`
- Local reports: `/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_epoch10_remote189_20261113`
- [Live remote queue](progress.md)
- [Combined four-model status](combined_progress.json)
