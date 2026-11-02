# RoboTwin-10 seven-model campaign (H100 interactive allocations)

Trains seven LoopWAM architectures on RoboTwin-10 (the LiLa-WAM 10-task subset: adjust_bottle, grab_roller,
hanging_mug, move_stapler_pad, open_microwave, press_stapler, scan_object, stack_blocks_two, stamp_seal,
turn_switch) with the full-RoboTwin v0 recipe, and publishes each to
`anhdao69/LoopWAM_NT/robotwin-10/<variant>/` on Hugging Face.

Variants: `v0-video4-action4`, `v0-video4-action2`, `v0-video4-action1`, `v0-video2-action2`,
`v0-video1-action4`, `dense-s12`, `dense-s30` (see `VARIANTS` in `src/fastwam/models/wan22/loopwam_robot.py`).

## Pieces

| File | Role |
|---|---|
| `scripts/train_loopwam_robotwin10.py` | Trainer: 1,163,351 windows, 5 epochs = 45,445 updates of exactly 128 windows, seed 42. Data order, noise and loss normalization are independent of the GPU count, so runs resume on any world size. Stops cleanly on `--stop-file` / `--stop-at-unix`. |
| `scripts/prepare_robotwin10.py` | Writes/checks `shared/data/data_manifest.json` and `dataset_stats.json`; checks every prompt has a T5 cache entry. |
| `scripts/smoke_robotwin10_variants.py` | Per-variant GPU smoke: parameter count, identical robot interfaces, finite gradients, exact save/load round trip. |
| `scripts/publish_robotwin10.py` | Verifies the epoch-5 checkpoint (payload, cursor, finite FP32, equal to the final `latest.pt`), uploads it with README/config/provenance/metrics/SHA256SUMS, checks Hub LFS SHA256; `--index` updates the root README and `checkpoint_index.json`. |
| `build_cache.sh` | Builds the frozen-VAE latent cache (214 GB) with `scripts/chronoloop_robot_precompute.py`. |
| `finalize_cache.sh` | After the build: verifies all windows valid, marks the local cache staged, copies it to the striped Lustre master (`COMPLETE`). |
| `stage_cache.sh`, `pcopy.py` | Copies the master cache to node-local `/tmp` on each allocation (parallel ranged copy, sampled verification). |
| `node_agent.py` | One per allocation (Slurm step via `srun --overlap`): stages the cache, starts/resumes/stops trainers per `control/agent_<job>.json`, restarts crashes (cooldown after three), kills stalled runs, runs publish tasks. |
| `dispatcher.py` | Runs on the login node: polls `squeue` for the allocation pool, keeps an agent in every running allocation, plans placements, queues HF publishes and the final index update. Writes `control/STATUS.txt`. |
| `config.json` | Allocation pool, frozen code path, per-variant cost estimates; `gpus_per_run: 4` (one variant per whole 4-GPU allocation, non-preemptive, most remaining work first). |
| `env.sh`, `watch_events.sh` | Environment; event watcher used for monitoring. |

Work directory on the H100 cluster: `/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_work`
(these files live there; runs in `/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10/<variant>/train`).
