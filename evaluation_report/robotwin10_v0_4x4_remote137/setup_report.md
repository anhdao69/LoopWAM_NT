# Remote RoboTwin-10 v0 4/4 evaluation — 10.148.182.137

User requested a separate v0 4/4 seed42 ten-task evaluation on this server while the local v0 4/2 job runs independently.

## Checkpoint and code

Exactly the original local RoboTwin-10 v0-video4-action4 checkpoint, epoch5/update45,445, 584,567,886 policy parameters, native video/action loops4/4, partition3/6/3, aligned late KV, no history/memory. HF revision `df3d575d3f3fde3cf1b8d5f609fab93dc6a95649`. Checkpoint SHA256 `bb1563f16bcd102b3f4ad4a1a14377ee49de7e461d862b7bbc5dba497f036127`. All ten release files are transferred unchanged and checked against pinned HF LFS SHA256/Git blob SHA1 and SHA256SUMS.

Frozen code commit: `7a2181d0ecc8e30d64806e94535b347bc9e1da2e`, copied by git bundle; model and harness source remain unchanged. Local validated Python3.10.22 environment was packed, archive-hash verified, unpacked and relocated, with the original RoboTwin venv overlaid. Dependency checking passed. All38 existing tests passed after relocating one test-only absolute path to a byte-identical FastWAM reference file. Assertions were unchanged; original failure log and test provenance are retained.

## Runtime portability

28,172 source/runtime input paths are checked by portable_inputs.json. The model weights, VAE, T5 weights/tokenizer, object meshes, textures, simulator/planner libraries and source match the originals. Six cuRobo YAML files contain absolute URDF/collision-sphere paths and received path-only relocation; original and relocated hashes are recorded in relocation.json. Interpreter and editable-package paths were adjusted for this dedicated workspace. No task horizon, scene acceptance, model computation or image preprocessing was changed.

GPU model matches the original server: RTX PRO6000 Blackwell Max-Q96GB. Hardware UUID and driver version are recorded in host_environment.json. Cross-server bitwise episode reproducibility is not assumed; native-versus-CUDA-graph equality is freshly checked on this server before scoring.

## Scored protocol

All10 release tasks: adjust_bottle, grab_roller, hanging_mug, move_stapler_pad, open_microwave, press_stapler, scan_object, stack_blocks_two, stamp_seal, turn_switch. Seed42 only; clean/randomized;100 accepted episodes/task/setting;2,000 total. NFE10, CFG1, predict32/execute24, fixed per-query CPU action noise seeded42, FP32 weights/BF16 compute, absolute14-Dqpos, checkpoint z-score normalization. Original FastWAM PIL uint8 three-camera packing384×320; original expert acceptance and task horizons; unseen instructions; RT32/depth8/OIDN2.3.3.

Candidate scene seeds start4,300,000 before expert skips. The released Python instruction-template sampler is not explicitly seeded; actual instructions and scene seeds are logged. Thus identical seed42 does not guarantee identical instruction strings across runs. Seed43 reruns and mixed-seed substitutions are excluded from all full-run comparison tables.

## Queue and memory admission

At setup, an unrelated user's VLLM process occupied about80GB, leaving about17GB free. It is left untouched. The queue checks memory before GPU phases. The model-only graph gate requires14,500MiB free, based on the original model's roughly4,062MiB peak reserved memory plus margin. Simulator gates require at least30,000MiB free for one worker,42,000 for two,60,000 for four. Three consecutive samples are required for mandatory admission. Shared T5 plus CUDA graphs is used after validation. The fastest successful graph profile with at least8,000MiB measured memory margin is selected; full-run admission additionally requires measured added peak plus10,000MiB or its profile threshold, whichever is larger.

Until enough memory is available, the queue remains waiting rather than stopping other jobs or reducing evaluation quality. Finish time depends on GPU availability and later calibration; no fixed ETA is promised.

Native and graph simulator calibration uses adjust_bottle/grab_roller, clean/randomized, two episodes per pair. Calibration episodes are excluded. Scored cells allow at most3 attempts with1,800-second no-progress timeout. Interrupted-attempt partial episodes are excluded; completed cells are artifact-validated before reuse. Videos, actual instructions, accepted scene seeds and query hashes are retained.

## Storage, monitoring and final reports

Remote workspace: `/media/vmo-perception/disk_2/anhdh35/loopwam_robotwin10_v0_4x4_20261104` on the secondary disk (6.7TB initially free). The nearly-full system disk is not used for bulk artifacts.

Remote tmux: `robotwin10_v0_4x4`. The session survives SSH disconnection, not host reboot. Validation failures stop progression and appear in queue_status.json.

Remote reports: `repo/evaluation_report/robotwin10_v0_4x4_remote137` beneath that workspace. Reports/status are mirrored to `/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_v0_4x4_remote137` on the original server; videos stay on the remote secondary disk.

Final auditing verifies all20 pairs and2,000 videos, counters, traces, checkpoint and bound hashes. It writes final_report.md, final_results.json, task_results.csv, episodes.csv, failed_attempts.json and final_audit.json. The report compares this independent remote v0 run against the original local v0 4/4 and dense-s30 full seed42 evaluations.

## Verified admission

All ten release files passed integrity checks. Strict native CPU load passed with584,567,886 parameters and4/4 loops. All28,172 portable source/runtime paths passed verification (six documented path-only YAML relocations). Fresh native/CUDA-graph comparison matched exactly; measured native0.1651s/query and graph0.1308s/query, peak model reserved memory4,062MiB. The persistent queue is alive and currently waits for30,000MiB free before its simulator gate. No scored episodes have started as of this admission record. See [live queue status](queue_status.json) and [launch verification](launch_verification.json).

Clock measurement at setup: this remote server was approximately23.49seconds ahead of the original server (about7ms SSH round trip). Reports retain their server timestamps; per-run durations use the same host clock. No system clock was changed.
