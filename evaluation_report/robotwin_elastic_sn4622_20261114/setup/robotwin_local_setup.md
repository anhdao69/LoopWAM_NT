# Local RoboTwin evaluation on sn4622128794 (E1 test weight)

Built 2026-11-07. Before this, the host had no RoboTwin evaluation setup: no assets, no simulator environment, no compiled
CuRobo. All earlier RoboTwin results were produced on the campaign host. This setup rebuilds the local RTX v0 protocol
(91.73% report, `evaluation_report/robotwin_v0`). Wherever the campaign recorded a hash, the files here are checked
against it. The test weight is `anhdao69/LoopWAM_NT` `robotwin/Elastic/E1/epoch_01.pt` (RT50-ep5-E1).

## Locations

| What | Where |
|---|---|
| Evaluator code (branch `eval/robotwin-local-e1-20261107`) | `/data/anh/code/LoopWAM-robotwin-eval` |
| Campaign-exact evaluator (commit `9cdceaa`, 777/777 protocol hashes) | `/data/anh/code/LoopWAM-robotwin-campaign-9cdceaa` (worktree) |
| Pristine elastic_robotwin training tree (`2c78ff6`) | `/data/anh/code/LoopWAM-elastic-robotwin-2c78ff6` (worktree) |
| Python 3.10.21 env, assets, CuRobo, CUDA 12.8 toolchain, checkpoints | `/data/anh/robotwin_eval/{env,assets,curobo-0.7.8,cuda128,checkpoints}` |
| Runtime environment variables | `source /data/anh/robotwin_eval/environment.sh` |
| Setup and launch scripts (copies in `scripts/robotwin_local_setup/`) | `/data/anh/robotwin_eval/setup/` |
| Verification records | `/data/anh/robotwin_eval/verification/`, `runtime_manifest.json` |

## Reconstruction and verification

| Item | Result |
|---|---|
| Evaluator source | 777/777 campaign protocol hashes (campaign-exact tree). The merged tree differs only in `loopwam_robot.py`, which gains an optional `payload=` argument; the gates show v0 output is bit-equal. |
| Python packages | 212/212 campaign pins (`uv`, Python 3.10.21; campaign used 3.10.22). `pip check` passes. torch 2.7.1+cu128 includes sm_120. |
| SAPIEN / mplib / OIDN | 109/109 campaign env files: the mplib planner patch, the original urdf_loader, OIDN 2.3.3 (official archive, sha-verified) and the 2.3.3 `_oidn_tricks.py`. |
| Assets (TianxingChen/RoboTwin2.0 @ `3dc3b79`) | 20,841 files byte-identical to the campaign manifest, none missing or extra. The 6 generated `curobo*.yml` files equal the campaign's after path mapping. |
| Runtime manifest | 21,529 files hashed; all 21,132 campaign entries present; 21,110 identical; 11 differ as expected (the 6 path-bearing yml files and 5 CuRobo extensions compiled here). |
| CuRobo v0.7.8 | 160/160 campaign source hashes. Built with NVIDIA's CUDA 12.8 redistributables (sha-verified), sm_120 SASS. Version 0.0.0, as in the campaign. |
| Wan VAE / UMT5 / tokenizer | Campaign hashes (VAE `38071ab5…`, UMT5 `7cace0da…`). |
| Tests | 26/26 campaign evaluator tests, 77/77 elastic_robotwin transport tests, 7/7 new E1 policy, probe and T5 tests. |
| Task modules | All 50 task modules and step limits import (same output as the campaign log). |
| Renderer | RT shader, 32 spp, depth 8, OIDN **2.3.3 core + CUDA device libraries loaded**, finite images, render GPU PCI equal to the CUDA GPU. |
| E1 export (CPU, static) | Release sha matches. 339/339 frozen tensors and the proprio encoder bit-equal to the FP32 v0 teacher. 338/338 student tensors trained and finite. Step 47462, T1, FLAT9 schedules, contract hashes (teacher, VAE, normalization) equal. |

### Scene acceptance replay (unmodified harness, model-free probe policy)

The harness walks candidate seeds from 4,300,000 with its own expert planning and success check. Results:

- adjust_bottle/demo_clean: the first 10 accepted seeds equal the campaign's (14 candidates, 4 rejections).
- adjust_bottle/demo_randomized: the first 3 equal the campaign's.
- Runs that later crashed on GPU-memory faults had also matched up to the crash: 4 seeds for adjust_bottle/demo_randomized and 5 for beat_block_hammer/demo_randomized.

Instructions come from the harness's unseeded Python `random`, so they are not reproducible. They are recorded, not compared.

### GPU gates on real simulator observations (`scripts/elastic_robotwin_gates.py`, `verification/gates/gates.json`)

Inputs were 4 first observations (2 clean, 2 randomized) captured by the harness. Text contexts came from block-streamed
UMT5 (`elastic_policy/t5_lean.py`). All 9 checks passed with a maximum absolute difference of 0:

1. The contexts are deterministic, finite and zero-padded.
2. The campaign-exact v0 policy is deterministic and finite.
3. v0 in the merged tree is bit-equal to the campaign tree, raw and denormalized.
4. E1's deployment input path plus teacher velocities is bit-equal to the deployed v0 `infer_action`. This repeats the training preflight on real observations.
5. Building the transport wrapper leaves v0 unchanged.
6. E1 policy inputs are bit-equal to v0's.
7. E1 rollouts for flat_k4_n10, k1_n2, k2_n5, k4_n5 and k1_n10 are finite, deterministic and consistent with `_query`.
8. On flat_k4_n10, the CUDA graph equals eager, including replays with new observations.
9. The pristine elastic_robotwin tree is bit-equal to the merged tree on all 20 schedule × observation rollouts.

The sanity check gives the same picture. On flat_k4_n10, E1's actions are within 0.8–2.4% relative L2 of the teacher's.
Actions for a mismatched observation differ by 135%. Peak GPU use is 3.4 GiB (E1) and 3.7 GiB (v0).

## Host constraints found during setup

* **GPU memory.** An RT worker needs up to about 5 GB on randomized scenes (textures, OIDN buffers) plus about 4 GB for
  E1. The shared UMT5 server needs about 12 GB. At setup time the user's GRPO run held about 80 GB on each GPU, and
  other users held more.
* **OIDN OOM is silent.** When GPU memory runs out, SAPIEN logs `OIDN Error: out of memory` and **keeps rendering without
  denoising**, before failing later with `cannot create buffer`. The campaign audit counted `cannot create buffer` and CUDA
  OOM, but not OIDN errors. `evaluate_elastic_robotwin.py` and the replay now reject any attempt whose log contains
  `OIDN Error`, `cannot create buffer`, `OutOfMemoryError` or `out of memory`.
* **Host RAM for UMT5.** The standard T5 loader builds umt5-xxl in FP32 on the CPU (about 23 GB) and then reads the 11 GB
  bf16 state dict, a peak of about 34 GB. That is unsafe on this shared 62 GB host. `--t5-loader lean`
  (`elastic_policy/t5_server_lean.py`) builds on the meta device and assigns the memory-mapped bf16 tensors. It runs the
  same `TextEncoder` and `serve()` code with identical parameters, because the standard path computes bf16(fp32(x)) = x.
  `scripts/check_t5_lean.py` checks the full-GPU lean encoder against the streamed gate contexts before the smoke run.

## Expert-gated scene acceptance is not reproducible for hard-planning tasks

The harness walks candidate seeds from 4,300,000 and accepts a scene only if its scripted expert succeeds. For hard-planning
tasks this check is not reproducible run to run. The expert's outcome depends on which trajectory CuRobo MotionGen finds.
MotionGen keeps its random state across candidates and stops a plan after a 10 s wall-clock timeout (`motion_gen.py`,
`plan_single`/`plan_batch`), so GPU load and planning history change which candidates pass. Four runs of
put_object_cabinet/demo_clean on the same code accepted four different seed sequences: the campaign, the E1 evaluation
worker, an expert-only replay, and an expert-only replay with the timeout removed (`verification/curobo_nondeterminism.json`).
Removing the timeout did not restore the campaign sequence, so the unmodified harness is kept. Easier tasks reproduce
exactly (adjust_bottle, beat_block_hammer, click_bell; put_bottles_dustbin mostly). A scene is fully determined by its seed,
and the policy never uses the planner. So each episode is evaluated correctly, but for some cells the scene set differs from
the campaign's. The audit (`scripts/robotwin_eval_audit.py`) reports, per cell, whether the seeds equal the campaign's, and an
E1-vs-v0 paired comparison on the seeds both runs accepted. It requires everything else: records, videos, no render faults,
the contract schedule and noise seed in every episode, and seeds walking upward from the start.

## Evaluation queue (2026-11-07): flat_k1_n2, flat_k4_n2, flat_k4_n5

`setup/run_e1_queue.sh` (unit `rtw-e1-queue`) runs several schedules from one orchestrator:

- **Shared T5 server:** one lean T5 server, placed on the GPU with the most free memory.
- **Evaluation order:** every cell of evaluation 1 starts before any cell of evaluation 2, so the tail of one overlaps the
  start of the next.
- **Cell order:** within an evaluation, cells start longest-first by v0 campaign duration.
- **Memory-gated workers:** a worker starts only while its GPU has `ADMIT_GPU` + `GPU_FLOOR` MiB free and the host has
  `ADMIT_RAM` + `RAM_FLOOR` MiB available. Workers started in the last `HOLD` s count as already using their share.
- **Never preempts:** at most 6 workers per GPU; nobody else's job is ever preempted.
- **Recovery:** the orchestrator is relaunched up to 3 times; resume reuses only verified complete cells.
- **Retries:** a cell is retried up to 8 times. A render fault, crash or 30-minute stall rejects the attempt.
- **Checks:** each evaluation gets the campaign seed check and the audit as soon as it completes.

Scheduling never changes how a cell is evaluated: same contract, eager inference (no CUDA graph). Measured peaks per worker are
9.3–10.3 GB GPU and 4.6–5.4 GB RSS; the T5 server uses 11.5 GB GPU and 1.8 GB RSS. GPU gates for these schedules are in
`verification/gates_eval3` (all 9 checks pass, bit-equal to the original gates on the shared schedules). A queue smoke
(2 schedules × 2 tasks × 2 settings × 2 episodes) is in `results/smoke_queue_20261107`.

```bash
journalctl --user -u rtw-e1-queue -f                       # progress; results in /data/anh/robotwin_eval/results/e1_<schedule>
cat /data/anh/robotwin_eval/results/e1_flat_k1_n2/summary.json
systemctl --user kill -s SIGINT rtw-e1-queue                # stop (complete cells are kept and reused on relaunch)
```

The host rebooted on 2026-11-09 at 09:20 UTC with flat_k4_n5 at 93/100 cells. Transient user units do not survive a reboot,
so the queue simply stopped. It was resumed on 2026-11-11 (unit `rtw-e1-resume-k4n5`, same script, `SCHEDULES=flat_k4_n5`).
Before reusing them, the orchestrator re-verified the 93 finished cells, including ffprobe on every video. The interrupted
attempts stay on disk without `failure.json`, and the audit lists them as interrupted.

## E0-epoch10-teacher-5ep (2026-11-11): flat_k1_n5, k1_n10, k2_n5, k2_n10, k4_n5, k4_n10

The export is `robotwin/Elastic/E0-epoch10-teacher-5ep/epoch_05.pt` (sha256 `07d24ff7…`, HF revision `b72f3555`). It is a
fresh RT50-E0 run with T0 transport, FLAT9 schedules and 5 RoboTwin-50 epochs (237,310 updates), distilled from the v0 4/4
**epoch-10** teacher `robotwin/v0-video4-action4/epoch_10/policy.pt` (sha256 `70cf5af7…`, step 474,620). The training code is
still `elastic_robotwin@2c78ff6`: the remote branch has no code change since then, and the deployment code here is
byte-identical to that tree. The data, normalization and VAE contract are the same as E1's. The reference is the v0 epoch-10
campaign `evaluation_report/robotwin_v0_epoch10` (91.87%), which used the same protocol as the epoch-5 campaign
(`epoch5_protocol_comparison.json`). Its `episodes.csv` is used for the paired comparison.

| Step | Record | Result |
|---|---|---|
| Download (pinned revision, sha256 of the export and the teacher, `SHA256SUMS` of both releases) | `setup/download_e0_ep10.sh` | passed |
| Static export check, `--expected-kind T0 --expected-step 237310 --expected-epoch 5` | `verification/e0ep10_static.json` | 22/22 checks |
| GPU gates against the epoch-10 teacher, every phase rerun | `verification/gates_e0ep10` | 10/10 checks |
| Queue smoke: flat_k1_n5 and flat_k4_n10 × adjust_bottle, click_bell × 2 settings × 2 episodes | `results/smoke_e0ep10_20261111` | passed (see below) |

Static check details: all 339 frozen tensors and the proprio encoder are bit-equal to the FP32 epoch-10 teacher, and all
338 student tensors were trained.

GPU gate details:
- The contexts are bit-equal to the E1 gates' contexts.
- On the epoch-10 teacher, the merged tree equals the campaign tree, and the teacher rollout through the elastic input path
  equals the deployed `infer_action`.
- The student's inputs are equal to the teacher's.
- All six schedules are finite, deterministic and consistent with `_query`.
- On flat_k4_n10, the CUDA graph equals eager.
- The pristine `2c78ff6` tree is bit-equal on all 24 rollouts.
- The gate record names the checkpoint, its sha256, the transport kind and the step.

Sanity check: the student's actions are within 0.7–7.1% relative L2 of the teacher's; 4 loops track the teacher most closely.
For a mismatched observation the difference is 135%.

Smoke results: 16/16 episodes succeeded; both audits and both reference seed checks passed; no attempts were rejected;
all 8 cells' seeds equal the epoch-10 run's. Each manifest records T0, step 237,310, eager inference, the RT shader and the
lean T5 server, and every episode header carries the scheduled depths. Policy query latency is 0.06 s for flat_k1_n5 and
0.18–0.44 s for flat_k4_n10, a few percent of the simulator time per episode.

Memory: at 19:17 the kernel OOM-killed a worker in put_object_cabinet/demo_randomized (exit −9, swap full). The attempt was
rejected and retried. The 6 GB `RAM_FLOOR` was too small, because workers grow from about 4.5 to 7 GB during a cell and
other users' memory swings by several GB. The queue was stopped with SIGINT, losing only in-progress attempts, and relaunched
at 19:20 with `ADMIT_RAM=7000 RAM_FLOOR=10000` and `-p OOMScoreAdjust=500`. With that adjustment, the kernel picks one of
our retryable workers before any other user's process if the host still runs out of memory.

Even with the larger floor, the kernel OOM-killed 5 more workers between 00:42 and 03:40 on 2026-11-12. All were randomized
cells, which grow to about 7–8 GB. The kernel picks the largest process, so it killed the cells furthest along: 270 episodes
were lost, including cells at 87 and 88 of 100. No other user's process was killed, and every killed attempt was rejected and
retried.

Since 05:17, `setup/oom_guard.sh` (unit `rtw-oom-guard`) sets every 10 s, without restarting anything:
- each worker's `oom_score_adj` to `1000 − 5 × episodes done`, so the least-advanced worker is the victim;
- the T5 server, orchestrator and queue scripts to 400.

Kernel score order is then: fresh workers, then advanced workers, then T5 and orchestrator, then everything belonging to other
users, which stays at 800 or less. It only changes which process the kernel kills under memory exhaustion; evaluation is
unchanged.

On 2026-11-13 at 18:24 UTC, the T5 server had 11 GB resident and 6.4 GB swapped. Its `TextEncoder.memo` keeps every prompt
it has ever encoded and is never evicted, so it grows with the number of distinct instructions over the server's lifetime;
this is also what had filled swap. The memo cannot change results: the encoder is deterministic, as the gates verify. But
the server code is part of the hashed evaluation contract, so it is not patched mid-run. Instead, the queue is restarted to
start a fresh server.

Each restart costs the in-progress attempts. It also re-verifies (ffprobe) every finished cell of every listed evaluation
before any worker starts, so completed evaluations should be dropped from `SCHEDULES` when relaunching. At 18:27 the queue was
relaunched with `SCHEDULES=flat_k2_n10,flat_k4_n5,flat_k4_n10` and, because the host was almost idle,
`ADMIT_RAM=6500 RAM_FLOOR=6000`. `oom_guard` keeps any shortfall on our freshest worker.

The full queue started on 2026-11-11 at 18:48 UTC as unit `rtw-e0ep10-queue` and was relaunched at 19:20 with the settings above. Results go to `results/e0ep10_<schedule>`,
and each evaluation is audited against the epoch-10 reference when it completes:

```bash
journalctl --user -u rtw-e0ep10-queue -f
python /data/anh/robotwin_eval/diag/partial_sr.py --ref /data/anh/code/LoopWAM_NT/evaluation_report/robotwin_v0_epoch10/episodes.csv \
    /data/anh/robotwin_eval/results/e0ep10_flat_k1_n5                     # read-only SR snapshot vs the epoch-10 teacher
systemctl --user kill -s SIGINT rtw-e0ep10-queue             # stop; relaunch the same systemd-run command to resume
```

`setup/run_elastic_queue.sh` is `run_e1_queue.sh` generalized to any export through `CKPT`, `DATA_DIR`, `KIND`, `STEP`,
`GATES`, `REF` and `ORDER_CSV`. It refuses to start unless a passed gate record covers every requested schedule for the
same checkpoint sha256, kind and step. Cells start longest-first by the measured E1 flat_k4_n2 cell times on this host
(`results/order_e1_flat_k4_n2_seconds.csv`); this only affects ordering. Workers run eager (no CUDA graph), as in the E1
evaluations. Allocator tuning was checked and rejected: loading the model leaves 0.44 GB of anonymous RSS, so a worker's
~4.5 GB comes from the simulator side.

## Running

```bash
# smoke test (queued as user unit rtw-e1-smoke; starts automatically when memory allows):
#   adjust_bottle + click_bell, clean + randomized, 3 episodes each, flat_k4_n10
journalctl --user -u rtw-e1-smoke -f
# full E1 evaluation, memory-gated (results in /data/anh/robotwin_eval/results/e1_<schedule>):
systemd-run --user --unit=rtw-e1-eval -p OOMPolicy=continue -E SCHEDULE=flat_k4_n10 -E GPUS=0 -E PER_GPU=1 -E T5_GPU=0 \
    /data/anh/robotwin_eval/setup/run_e1_eval.sh
```

Remaining differences from the campaign:

- Python 3.10.21 here vs 3.10.22.
- NVIDIA driver 580.105.08 vs 580.178.04.
- CuRobo binaries are compiled locally.
- setuptools is 80.9.0 here; the campaign version is unrecorded.
- The lean UMT5 loader (identical weights).
- Two shared GPUs here vs one exclusive GPU.
