# RoboTwin-10 v0 4/1 — local rerun, 2026-11-05

## Requested rerun and release

The user requested repeating 4/1 on this local machine. This is a fresh independent run, separate from the completed vmo_aLinh 4/1 result (81.35% overall; 823 clean + 804 randomized successes). It uses the same checkpoint and scored protocol; the new result will not replace the old result based on which is higher.

Checkpoint: `anhdao69/LoopWAM_NT/robotwin-10/v0-video4-action1`, pinned HF revision `c0a56a65986af8c0f0beed4f17b4e75ad3cbad81`. SHA256 `b0fd9da51d9ff5ab512d5729de6565e85d240a10f078ddf9f803733a77aa757d`; 2,338,751,909 bytes. Epoch 5 / update 45,445; native 4 video / 1 action core loops; 12 stored block pairs, partition 3/6/3; 584,567,886 policy parameters; aligned late KV; no memory/history. Already downloaded release files are copied into the new directory and reverified using HF hashes and release SHA256SUMS.

## Protocol

Seed 42, NFE 10, ten tasks, clean/randomized, 100 accepted episodes per task/setting, 2,000 scored episodes. Tasks: adjust_bottle,grab_roller,hanging_mug,move_stapler_pad,open_microwave,press_stapler,scan_object,stack_blocks_two,stamp_seal,turn_switch.

CFG 1, predict 32 / execute 24, absolute 14-D qpos, checkpoint global z-score normalization, FP32 weights / BF16 compute, fixed CPU action noise seeded 42 for every query. Original FastWAM PIL uint8 three-camera packing 384×320,unseen instructions,task horizons and expert scene-plan/success acceptance. Rendering RT 32 / depth 8 / OIDN 2.3.3. Candidate scene seeds start 4,300,000 before expert-rejected scenes are skipped.

The instruction-template Python RNG remains as released, without an explicit seed. Actual instructions are logged; repeating seed 42 does not guarantee exactly paired instructions across runs. No seed mixing or best-seed substitution is applied.

## Local environment and storage

GPU: RTX PRO 6000 Blackwell Max-Q 96 GB. At setup,about 71,808 MiB was free while two independent SpatialForcing/VLN jobs occupied about 25 GB and substantial GPU compute. Those jobs are not modified. Runtime and latency measurements reflect this concurrent workload and are not directly comparable with an idle GPU.

Frozen evaluator: `/storage/anhdh35/LoopWAM-robotwin10-dense-eval`, commit `7a2181d0ecc8e30d64806e94535b347bc9e1da2e`. All nine training-source hashes match. Reuse the established Python 3.10.22 / PyTorch 2.7.1+cu128 RoboTwin environment and existing assets. Model and benchmark source are unchanged.

The root filesystem is full. New temporary files and caches are directed to `/data14t/anhdh35/loopwam_robotwin10_v0_4x1_local_20261105`: TMPDIR, XDG_CACHE_HOME, CUDA_CACHE_PATH, TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR, and __GL_SHADER_DISK_CACHE_PATH. The evaluator uses a Unix socket under /tmp; creation/cleanup of that socket was tested successfully without changing the evaluator. Other user files are not deleted. Bulk data has about 9.9 TB free; local reports remain on /storage.

## Fresh checks and speed selection

All 38 existing evaluator/model tests passed freshly, dependency check passed, and run configuration was checked against the remote 4/1 protocol. Strict loading must verify native 4/1 loops, finite parameters,parameter count, epoch/update,training contract,normalization and VAE. Bound source/runtime assets must match their established hashes before graph tests and before scoring.

A fresh native/CUDA-graph gate uses 8 saved real observations, seeds 42/43/44, and forward/reverse order with exact equality (atol=rtol=0). Unscored simulator profiles: native 1, graph 1, graph 2, graph 4; each 8 episodes on adjust_bottle/grab_roller in clean/randomized settings. Choose the fastest passing graph profile with at least 8,000 MiB measured memory headroom. Full admission also requires measured extra peak plus 10,000 MiB or the worker threshold, whichever is greater. A single shared T5 server serves workers. Short calibration timing includes startup, simulation and concurrent workload variability.

## Persistent queue and final evidence

Bulk: `/data14t/anhdh35/loopwam_robotwin10_v0_4x1_local_20261105`. Tmux: `robotwin10_v0_4x1_local_20261105`. It survives terminal disconnects, not a host reboot. Maximum 3 attempts per cell; 1,800-second no-progress timeout. Calibration and interrupted-attempt partial episodes are excluded. Only completed 100-episode cells enter final scoring.

Log every accepted scene seed, instruction, success, video and action-query hash. After all 20 cells complete, the auditor checks 2,000 videos, counters, checkpoint, source and runtime hashes. It writes final_report.md, final_results.json, final_audit.json, task_results.csv, episodes.csv, failed_attempts.json and baseline_comparison.json.

Comparisons retain the remote 4/1 result, all three completed 4/4 runs, dense-s30, 4/2 and 2/2 separately. All full results use seed 42. Unseeded instruction variation, cross-GPU differences and concurrent-load timing differences are disclosed.

[Queue status](queue_status.json) · [Live report](report.md) · [Execution plan](execution_plan.md) · [Final report](final_report.md) when generated.

## Verified model/runtime admission

All ten release files passed pinned HF hashes and SHA256SUMS; all 38 evaluator tests passed freshly in 4.68 seconds (one Transformers deprecation warning); pip check passed. Native 4/1 strict loading confirms correct loops, epoch/update, 584,567,886 finite parameters,normalization and VAE. Bound runtime/source integrity checks passed.

Eight real-input native/CUDA-graph checks passed exact equality with seeds 42/43/44 and forward/reverse order. Native model-only latency 0.136465 s/query; graph 0.149099 s/query; peak reserved 4,062 MiB. The graph timing was slower in this short measurement under heavy concurrent GPU load; this is not evidence of an isolated hardware performance difference. Worker selection uses full simulator profile wall times under the same active workload; no unsupported model-only speedup claim is made.

## Completed simulator calibration

All four profiles completed successfully. Their 32 calibration episodes are excluded from the scored evaluation. Each profile used the same checkpoint, seed 42, NFE 10, two tasks, both settings, and two accepted episodes per task/setting.

| Backend | Workers | Wall time | Successes | Peak GPU memory (all jobs) | Memory headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| native | 1 | 435.46 s | 8/8 | 48,026 MiB | 49,861 MiB |
| cuda-graph | 1 | 429.52 s | 8/8 | 47,952 MiB | 49,935 MiB |
| cuda-graph | 2 | 254.03 s | 8/8 | 57,332 MiB | 40,555 MiB |
| cuda-graph | 4 | 172.47 s | 8/8 | 67,841 MiB | 30,046 MiB |

Selected **four CUDA-graph workers**, with one shared T5 server. The measured profile was 2.49× faster than the one-worker graph profile. This is a short simulator-throughput measurement, including startup and competing workloads; it is not an estimate of the entire benchmark duration. Full admission requires at least 60,000 MiB free across three samples, followed by another checkpoint/runtime integrity check.

## Scored launch verification

The full evaluation started with four CUDA-graph workers. The launch manifest matches all 16 shared planned protocol fields and all 18 compared protocol/source fields of the completed local 4/2 baseline. The checkpoint SHA256 is the verified 4/1 release hash above. The first scored episodes completed successfully, with no failed attempts at launch verification. This is an in-progress evaluation, not a final SR. See [launch verification](launch_verification.json), [deployment snapshot](deployment_status.json), and the continuously updated [live report](report.md).
