# RoboTwin-10 v0 4/4 — fresh server137 run, 2026-11-05

## Request and checkpoint

The user requested another v0 4/4 evaluation on `anhdh35@10.148.182.137`. This is a fresh independent run. The earlier server137 queue was cancelled before scoring on2026-11-04 and remains preserved; it is not resumed or relabeled as this run. The separately requested4/1 evaluation continues on10.148.183.189.

Use the same original epoch5/update45,445 checkpoint as prior4/4 evaluations. HF release:`anhdao69/LoopWAM_NT/robotwin-10/v0-video4-action4`; pinned revision:`df3d575d3f3fde3cf1b8d5f609fab93dc6a95649`. SHA256:`bb1563f16bcd102b3f4ad4a1a14377ee49de7e461d862b7bbc5dba497f036127`;2,338,751,845 bytes. Native4video/4action core loops;12 stored block pairs, partition3/6/3;584,567,886 policy parameters;late aligned KV;no memory/history. The already-downloaded release files are copied into the new run directory and reverified against HF hashes and SHA256SUMS. No network download is needed if every file verifies.

## Scored protocol

Seed42;NFE10;all ten tasks;clean and randomized settings;100 accepted episodes per task/setting;2,000 scored episodes. Tasks:adjust_bottle,grab_roller,hanging_mug,move_stapler_pad,open_microwave,press_stapler,scan_object,stack_blocks_two,stamp_seal,turn_switch.

CFG1;predict32/execute24 actions;14-D absolute qpos;checkpoint z-score normalization;CPU action noise fixed to seed42 for every query;FP32 model weights/BF16 compute. Original FastWAM PIL uint8 three-camera packing384×320;unseen instructions;original task horizons and expert scene acceptance. Rendering RT32/depth8/OIDN2.3.3 matches previous evaluations. Candidate scene seeds start4,300,000 before rejected scenes are skipped.

Python instruction-template sampling follows the original unseeded implementation. Actual scene seeds and instruction strings are recorded; repeating seed42 does not guarantee exactly paired instruction strings across runs. This is a separate rerun, not a replacement selected by SR, and no seed43 or best-seed substitutions are used.

## Host and reproducibility

Server:`10.148.182.137`. GPU:RTX PRO6000 Blackwell Max-Q96GB, approximately97,002MiB free and no compute jobs at inspection. The prior deployment uses Python3.10.22,PyTorch2.7.1+cu128 and the established SAPIEN/mplib/cuRobo/OIDN environment. GPU driver/UUID/package evidence is saved freshly in host_environment.json.

Frozen evaluator commit:`7a2181d0ecc8e30d64806e94535b347bc9e1da2e`. All nine release training-source hashes must match. The previous validated environment, code and assets are reused without modification. Six planning YAML absolute asset paths were relocated during original setup; their recorded hashes are rechecked. The test-only FastWAM reference path relocation remains documented by the existing portable test provenance. No new relocations are required.

New bulk path:`/media/vmo-perception/disk_2/anhdh35/loopwam_robotwin10_v0_4x4_20261105`. Shared frozen runtime/code:`/media/vmo-perception/disk_2/anhdh35/loopwam_robotwin10_v0_4x4_20261104`. The bulk disk has about6.6TB free. New checkpoint copies, scripts, tests, profiles, videos, logs and results are isolated from previous runs.

## Admission and inference speed

Fresh38-test evaluator/model suite and dependency check are required. Strict checkpoint loading checks finite parameters,native4/4 loops,parameter count,epoch/update/training contract,normalization and VAE hashes. All bound runtime/source hashes are rechecked before launching scoring.

Fresh native/CUDA-graph action equality gate:eight real observations,seeds42/43/44,forward/reverse order,atol=rtol=0. Model-only latency and memory are recorded separately from simulator wall time.

Simulator profiles:native1,graph1,graph2,graph4 workers;each runs adjust_bottle/grab_roller,clean/randomized,two episodes per pair (8 episodes/profile;32 total). Select the fastest successful graph profile with at least8,000MiB measured memory margin. Admission requires stable free memory above each worker threshold and measured extra peak plus10,000MiB. A single shared T5 server serves all scored workers. The final report states fastest tested configuration, not an unmeasured global optimum. No NFE,precision,renderer or scoring changes are used for speed.

## Automatic queue and artifacts

Remote tmux:`robotwin10_v0_4x4_20261105`. It survives SSH disconnects,not host reboot. Maximum3 attempts per cell;1,800-second no-progress timeout. Only completed100-episode cells enter final scoring;calibration and partial failed attempts are excluded. Every scored episode keeps video,actual scene seed,instruction and action-query hashes.

After all20 task/settings finish,the auditor verifies2,000 videos,counters,checkpoint/source/runtime hashes and records excluded attempts. It generates final_report.md,final_results.json,final_audit.json,task_results.csv,episodes.csv and baseline_comparison.json. Comparisons include dense-s30,original4/4,server1894/4,4/2 and2/2 as separate full seed42 results. Exact per-episode instruction pairing or cross-GPU bitwise reproducibility is not assumed.

Local report mirror:`/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_v0_4x4_remote137_20261105`. Reports sync every60seconds;videos remain remotely. [Queue status](queue_status.json) · [Live report](report.md) · [Final report](final_report.md) when generated.

## Verified model/runtime admission

All38 tests passed freshly in21.53seconds and pip check found no broken requirements. All10 release files passed pinned hashes and SHA256SUMS verification. Strict native4/4 loading,parameter finiteness,normalization/VAE and all bound runtime/source checks passed. Fresh graph comparison passed exact action equality on8 observations with42/43/44 seeds and forward/reverse order. Native0.165613s/query;graph0.117280s/query;model-only peak reserved4,062MiB. Simulator calibration and scored launch are tracked separately.

## Completed simulator calibration

All32 unscored episodes passed with32 successes. Profiles (8episodes each):native1=235.55s;graph1=226.48s;graph2=151.04s;graph4=105.76s. Four workers were selected as the fastest tested configuration. Timings include startup and simulator overhead and are not a full-run ETA.

Selected profile peak45,954MiB,total baseline244MiB,incremental45,710MiB,observed total-memory headroom51,933MiB. Full-run admission requires60,000MiB free,checked over three samples15seconds apart. All38 tests,9training-source hashes,strict loading,runtime integrity,exact graph-action equivalence and32simulator checks passed. See [preflight](preflight.json) and [calibration](calibration.json).
