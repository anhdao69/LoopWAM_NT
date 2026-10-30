# Original LoopWAM v0: automatic epoch-5 evaluation and epochs 6–10

## Requested sequence and checkpoint policy

The user corrected the last checkpoint number to **10**. After the current five-epoch run completes, evaluate its **epoch_05.pt** on the full RoboTwin benchmark, then continue the original v0 model for epochs **6, 7, 8, 9 and 10**. Save a weight snapshot at every one of those epochs. The continuation starts at **3e-5** and cosine-decays to **3e-7**, without a new warmup, while retaining epoch-5 model weights and Adam state. No epoch-10 evaluation was added.

The healthy original trainer was left running. Its source and output are unchanged. The isolated continuation source is `/mnt/data/vmo-ai-task/anhdh35/LoopWAM_robotwin_queue`, branch `robotwin_eval_continue_20261030`, based on `04ea0fa`. Runtime source is pinned at `0703c9db3f71c8347d1541751ca352c5901ca440`. The original five-epoch runtime remains `de52d2f`.

## Exact continuation contract

Original LoopWAM v0, 3/6/3 stored partition, video/action loops 4/4, aligned KV, zero memory tokens/history. The same 27,500 demonstrations, 6,075,103 windows per epoch, cached latents/text, published normalization and 14-D robot interfaces are used. Precision remains FP32 weights/Adam moments with BF16 compute.

Eight H100s use the previously measured fastest layout: microbatch 8 × accumulation 2 × 8 GPUs = **128 real windows**. The 95-window final batch of each epoch is averaged over its actual count. Every window appears once in each additional epoch; absolute shuffle seeds are 47, 48, 49, 50 and 51.

| Boundary | Optimizer updates, cumulative | Real windows, cumulative |
|---|---:|---:|
|Epoch 5 parent|237,310|30,375,515|
|Epoch 10 final|474,620|60,751,030|
|Additional five epochs|237,310|30,375,515|

The first additional update uses LR 3e-5, and the last uses 3e-7. AdamW betas (0.9, 0.95), epsilon 1e-8, weight decay 0.01 and clip 1.0 are preserved. Recovery checkpoints are saved every 1,000 updates and at epoch ends, separately from the five retained epoch snapshots.

Before handoff, the queue compares every policy/proprio tensor between epoch_05.pt and the final optimizer checkpoint, verifies exact epoch/data cursor and immutable training contract, checks complete FP32 finite Adam moments, and records both SHA256 hashes. The trainer checks the parent hash again before continuation.

## Evaluation contract and hardware adaptation

Full evaluation means **50 tasks × 2 settings × 100 accepted episodes = 10,000 rollouts**: 5,000 clean and 5,000 randomized. Seed 42, unseen instructions, task-specific step limits, expert scene checks, 32-action prediction, replan every 24 actions, ten denoising steps, CFG 1, and FastWAM's fixed seed-42 action noise on each query. All episode outcomes and videos stay on the server. The original `LoopMoT` is constructed; only the existing T5 transport helper is shared with older experiment code.

The evaluator schedules two independent simulator workers per GPU, 16 workers across eight GPUs, with one shared T5 server. A cell is complete only after a zero worker exit, exactly 100 distinct scene records, and validation of every saved video. Completed cells can resume only under the identical checkpoint/protocol/runtime manifest. Failed cells retry at most three times; incomplete evaluation never releases continuation.

**Rendering deviation:** H100 cannot execute RoboTwin's requested ray-tracing shader. These evaluations explicitly use SAPIEN 3.0.0b1's `default` raster shader. This is recorded in manifests and affects visual comparability with ray-traced RoboTwin results. Task definitions, physics version, randomization, camera geometry and policy preprocessing were preserved. The shader override is opt-in; existing code still defaults to `rt`.

A second runtime issue was isolated to the old Vulkan renderer's PCI-device selection: its device-priority routine throws for nonmatching physical GPUs. A private Mesa device-select layer exposes the assigned PCI GPU to each worker, avoiding this bug without changing SAPIEN or physics. CUDA and Vulkan assignments are derived from the same GPU PCI address, and scene construction asserts the renderer's actual address. Two distinct H100 PCI addresses passed camera-render probes. Matching NVIDIA 570.158.01 userspace graphics libraries are isolated under `/mnt/data`; no system driver installation or reboot occurred. See [Mesa's device-selection documentation](https://docs.mesa3d.org/envvars.html) and the [SAPIEN H100 ray-tracing discussion](https://github.com/haosulab/SAPIEN/issues/219).

## Verification and automatic recovery

Final source regression results: **337 passed / 24 skipped on CPU**, **344 passed / 17 skipped on GPU**. Twelve pipeline tests cover continuation budget/LR/cursor, immutable parent rejection, log reconciliation on recovery, complete evaluation coverage, duplicate child locks, child failures, committed smoke recovery, artifact/runtime changes, environment activation failure, and CUDA/Vulkan mapping. Policy tests cover original-v0 inputs, normalization, fixed noise, replanning, protocol rejection and VAE provenance.

The single-GPU epoch-4 smoke completed one clean and one randomized `adjust_bottle` episode; both succeeded and both videos passed ffprobe validation. This is a functionality check, **not a benchmark success-rate estimate**. Cell times were 100.27 and 95.25 seconds, including separate model/planner startup and contention with the running trainer. A frame was visually checked. The concurrent repeat also passed both episodes on two distinct GPUs, in 110.29 and 110.35 seconds per cell. All 50 task modules imported successfully. These checks validate execution; two workers per GPU have not been separately benchmarked. Concurrent multi-GPU evidence is recorded in the accompanying JSON.

After the full epoch-5 evaluation, the queue performs a native eight-GPU continuation smoke: two updates, save/reload, then two more updates. It checks exact optimizer restoration, resumed window cursor and finite gradients. The full continuation then starts again from the verified epoch-5 parent in a fresh output directory, so the smoke adds no updates to the production model. **This native continuation smoke is scheduled, not claimed as already passed.**

The controller uses an exclusive lock and durable status files, runs independently of the SSH connection, and can adopt an already running child Slurm step. Its supervisor retries a failed controller up to three times. Training recovers from committed `latest.pt`; post-checkpoint metric records are preserved separately, and incomplete outputs without a checkpoint are quarantined. Source/runtime hashes prevent silent changes after accepted gates. A failed gate stops the sequence.

## Location, status and timing

Parent allocation: **5489**, worker-2, eight H100s; original training step **5489.10**. Queue controller step **5489.54**, named `test_training_queue`, is CPU-only and waits for verified parent completion. It does not reserve extra GPUs while waiting.

Run root: `/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin_full_v0_20261028/posttrain_20261030/`:

- `config.json`: immutable launch contract and source hashes.
- `queue/status.json`: current stage and heartbeat.
- `eval_epoch05/`: 10,000-rollout evaluation, episode records, videos and manifest.
- `epochs06_10/checkpoints/epoch_06.pt` through `epoch_10.pt`: retained continuation snapshots.
- `epochs06_10/latest.pt`: optimizer recovery state.
- `queue/COMPLETE.json`: written only after the full sequence passes all checks.

At roughly 03:53 UTC on October 30 the original run was around update 217,551/237,310, approximately 91.7% complete. Its recent update time was about 0.70–0.75 seconds. Epoch 5 is projected around **07:45–08:15 UTC**, allowing checkpoint I/O. Five additional epochs need approximately **48–52 hours**, based on the existing measured run. Full-evaluation time cannot be estimated tightly from two easy-task smoke episodes: a preliminary allowance of **6–24 hours** gives about **54–76 hours after epoch 5**, plus smoke/startup time. Re-estimate from completed production task cells. These are forecasts, not measured completion times.

The controller is persistent, but allocation 5489 must remain active. Node failure, allocation cancellation, or persistent storage/simulator failure can stop the sequence; no literal 100% availability guarantee is claimed. All persistent runtime, evaluation, training and recovery artifacts are under `/mnt/data`. About 17 TiB was free during setup.

Evidence: [machine-readable setup status](LoopWAM_RoboTwin_Epoch5_Eval_Continue_20261030.json) and [raw evidence directory](../evidence/robotwin_eval_continue_20261030/).
