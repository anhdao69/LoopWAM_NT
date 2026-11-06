# RoboTwin-10 v0 4/4 evaluation setup

Prepared November 3, 2026 (Asia/Ho_Chi_Minh). This report describes the automated queue; it is not a claim that scored evaluation has finished. Read [queue status](queue_status.json), [live results](report.md), and, when available, [final audited results](final_report.md).

## Requested model and provenance

- Hugging Face: https://huggingface.co/anhdao69/LoopWAM_NT/tree/df3d575d3f3fde3cf1b8d5f609fab93dc6a95649/robotwin-10/v0-video4-action4
- Pinned revision: `df3d575d3f3fde3cf1b8d5f609fab93dc6a95649`.
- Native checkpoint `policy.pt`: 2,338,751,845 bytes; SHA256 `bb1563f16bcd102b3f4ad4a1a14377ee49de7e461d862b7bbc5dba497f036127`.
- Release metadata: epoch 5, update 45,445; 584,567,886 policy parameters; 12 stored video and action block pairs; partition 3/6/3; video/action core loops 4/4; late aligned action KV; no memory/history.
- Training: same ten tasks, 5,500 demonstrations, 1,163,351 windows per epoch, five epochs, 5,816,755 window presentations; seed 42.
- Frozen evaluator: `/storage/anhdh35/LoopWAM-robotwin10-dense-eval`, commit `7a2181d0ecc8e30d64806e94535b347bc9e1da2e`. All nine recorded training source hashes match. The worktree name contains dense, but the native loader selects v0 from the checkpoint. No dense architecture override is used.
- Shared runtime: `/data14t/anhdh35/loopwam_robotwin_v0_20261030/env` and verified existing RoboTwin assets, VAE, T5 and renderer libraries. No model/simulator source was changed for this run.

## Ordered queue and current-work handling

Dense-s30 completed all 2,000 episodes and passed its artifact audit before this queue was prepared. Initially a wait for another SpatialForcing-VLN PID was prepared. The user explicitly changed that to **start when GPU memory allows**. That process barrier is removed; other GPU jobs are left untouched. The queue downloads now and waits for GPU headroom before the relevant GPU phases. It does not infer completion from an elapsed one-hour timer.

The persistent tmux session is `robotwin10_v0_4x4`. Its sequence is:

1. Verify predecessor dense-s30 completion and audit.
2. Download the ten release files from the pinned revision, retrying transient failures up to five times per file; verify HF LFS SHA256 or Git blob SHA1 plus release SHA256SUMS. Partial files never become accepted checkpoints.
3. Strictly load the native v0 model on CPU; verify finite parameters, parameter count, 12 stored blocks, loops 4/4, epoch/update, metadata equality, VAE and normalization hashes. Require the same ten-task training recipe and parsed dataset statistics as dense-s30.
4. Recheck existing source/runtime hashes and freeze this run's scripts and metadata in runtime_manifest.json.
5. Run native versus CUDA-graph action comparison on eight saved real observations, forward and reverse input order, with seeds 42/43/44. Require exact equality. These are diagnostic inputs, not scored episodes.
6. Run unscored simulator calibration: adjust_bottle and grab_roller, clean/randomized, two episodes per pair. Native one-worker baseline; CUDA-graph profiles with one, two and four workers when memory allows. Choose the fastest successful graph profile with at least 8,000 MiB total GPU memory margin. Eight workers are excluded because previous long RoboTwin evaluation hit memory/renderer limits.
7. Run all 20 scored task/setting pairs with the selected concurrency, then verify all 2,000 episode artifacts and publish final tables plus a matched-task dense-s30 comparison.

The queue survives disconnection through tmux. It does not install a host-reboot startup service. Queue errors are written to queue_status.json; a validation failure stops progression instead of silently changing the protocol. The existing scorer supports validated completed-cell reuse on manual restart. Concurrency is fixed once calibrated for a scored run.

## Scored evaluation protocol

| Field | Value |
| --- | --- |
| Tasks | adjust_bottle, grab_roller, hanging_mug, move_stapler_pad, open_microwave, press_stapler, scan_object, stack_blocks_two, stamp_seal, turn_switch |
| Settings | demo_clean and demo_randomized |
| Episodes | 100 accepted episodes/task/setting; 2,000 total |
| Evaluation seed | 42 only |
| Candidate scene seed start | 4,300,000; expert acceptance may skip candidates |
| Instruction pool | unseen; pool size 100 per scored cell |
| NFE / CFG | 10 / 1 |
| Action horizon / execution | predict 32; execute 24 before replanning |
| Action and state | absolute 14-D joint positions; release z-score statistics |
| Action noise | fixed seed 42 for each query; original CPU noise generation |
| Weights / compute | FP32 weights; BF16 autocast |
| Observation preprocessing | original FastWAM PIL uint8 packing, three cameras at 384×320 |
| Renderer | RT32, depth 8, OIDN 2.3.3; same Blackwell-compatible setup as preceding evaluations |
| Scene acceptance / horizons | original expert plan/success acceptance and task horizons |
| Failure handling | at most 3 attempts per task/setting; 1,800-second no-progress timeout |
| Artifacts | per-episode video, actual instruction, accepted scene seed, action-query hashes, worker commands and logs |

Only completed 100-episode cells enter final scoring. Interrupted-attempt partial episodes and all smoke/calibration episodes are excluded. Live SR includes current partial cells and is explicitly provisional. The original instruction generator does not explicitly seed Python's random sampler, so identical evaluation seeds do not guarantee exactly paired instruction strings across models. Actual instructions are logged.

## Speed and GPU admission

A single shared T5 server serves all workers. CUDA graphs are allowed only after exact-output validation. Minimum free memory before simulator calibration is 30,000 MiB for one worker, 42,000 for two, and 60,000 for four; the native/graph model-only gate needs 18,000. Three consecutive memory samples, 15 seconds apart, are required for mandatory admission. Profiles above one worker are skipped if their memory threshold is unavailable. Full-evaluation admission requires the larger of the selected profile's threshold or measured added peak plus 10,000 MiB. These checks leave headroom but cannot reserve VRAM against unrelated jobs that allocate later.

Short profiles measure total startup and simulator time under the current competing workload. Final throughput varies by task and success/failure duration. There is no firm finish time until the GPU wait and calibration finish. Dense-s30's 2,000 episodes previously took about 4.90 hours using two workers; that is context, not a promised v0 runtime.

## Verification and report artifacts

The five existing evaluator/model test files passed: **38 tests**. An initial CPU-only attempt failed one preprocessing test because that test allocates a CUDA input; the unchanged suite passed with CUDA visible. Both logs are retained. Queue integrity checks exercised corrupted/truncated files, Git/LFS hash checks and PID identity reuse; queue source syntax passed.

Checkpoint strict-load, graph equivalence, simulator smoke and final artifact checks are asynchronous queue gates. Their JSON files are evidence only once generated and passing; their presence is not claimed in advance here.

Bulk directory: `/data14t/anhdh35/loopwam_robotwin10_v0_4x4_20261103`.

Reports: `/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_v0_4x4`.

Final outputs: final_report.md, final_results.json, task_results.csv, episodes.csv, failed_attempts.json, final_audit.json, protocol_comparison.json and COMPLETE.json. Source and runtime hashes, release manifest, download receipt, test logs, preflight and calibration evidence are retained. The final report compares this v0 with the completed dense-s30 on the same ten tasks and evaluation settings.

## Verified checkpoint admission

All ten release files downloaded and passed pinned HF hashes plus SHA256SUMS. Native CPU strict load passed at 2026-11-03T12:48:37.327043+07:00: 584,567,886 finite policy parameters, 12 stored blocks per expert, video/action loops 4/4, epoch 5/update 45,445, correct VAE and normalization. See [download receipt](download_receipt.json) and [CPU validation](cpu_validation.json). CUDA-graph equivalence subsequently passed; simulator gates remain pending.

## Verified CUDA-graph gate

Eight real observation inputs, seeds 42/43/44, forward and reverse order: native and graph actions matched exactly (maximum absolute difference 0). Under the current competing GPU workload, native inference measured 0.3815 seconds/query and graph inference 0.3132 seconds/query, approximately 17.9% lower latency. This is a model-only diagnostic, not an end-to-end throughput estimate. Peak model reserved memory was 4062.0 MiB. See [graph gate](graph_gate.json). The queue is waiting for at least 30,000 MiB free GPU memory before its first simulator gate.
