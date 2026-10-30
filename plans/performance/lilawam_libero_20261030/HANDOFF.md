# LiLa-WAM trained on OUR LIBERO data (paper recipe): state on 2026-10-30 10:30

User request (2026-10-30): "train the lilawam for me and eval as our protocol".

## Done
- **Data.** OUR LIBERO data was converted into the authors' LIBERO HDF5 layout:
  - source: `/groups/yshang/an221229/data/FastWAM/libero_mujoco3.3.2/*_no_noops_lerobot`;
  - output: `/groups/yshang/an221229/data/LiLaWAM_libero_ours_hdf5` (36 GB; 1,712 episodes, 277,713 frames, 40 tasks);
  - converter: `../convert_ours_to_lilawam_hdf5.py`.
  - Images are 256x256 PNG, standard colour, upright, the same orientation as our LeRobot frames.
  - `action[:, 6]` is raw LIBERO: -1 open, +1 close (= 1 - 2*ours). `proprio` is OUR 8-D `observation.state`.
- **Adapter.** `/groups/yshang/an221229/code/lilawam_libero/`:
  - `libero_data.py`: the dataset subclass, plus `composite()`, which places [agentview | wrist] side by side as one 512x256 RGB image;
  - `train_libero.py`: runs the public `train.py` unchanged, with only `create_dataset` swapped;
  - `stats_libero.py`, which wrote `/groups/yshang/an221229/data/LiLaWAM_libero_ours_stats.json`;
  - `task_cond_libero.py`, which wrote `/groups/yshang/an221229/data/LiLaWAM_libero_ours_taskcond/<suite>/<task>/task_cond.npy`;
  - `libero_ours.yaml`, the base config.
  - `libero_data.libero_obs_to_inputs(obs)`: raw LIBERO env obs -> (composite image, 8-D state), exactly as in training. Its `quat2axisangle` matches FastWAM's (max diff 0.0 over 500 observations). **Use this in the eval adapter.**
  - `openloop_check.py`: held-in open-loop check through the public sampling procedure.
- **Data verification.** `../verify_conversion.py` checked 24 random demonstrations (3,876 frames): actions, gripper map, state and every pixel match the source exactly (`../conversion_verification.json`).
- **Hugging Face.** `hf_upload_watcher.py` runs on evuser2 and does not depend on the job or a session. It uploads each finished suite to `anhdao69/LoopWAM_NT` under `baselines/LiLa-WAM_LIBERO_ours/<suite>/`: `stage2_epoch2_final.pt`, `stage1_epoch5.pt`, configs, norm stats, logs and task vectors. It then verifies the HF LFS sha256 against the local file.
  - Records: `uploaded_<suite>.json`, `manifest.json`. Log: `hf_upload_watcher.log`.
  - If job 1094056 ends before a suite finishes, that suite's latest checkpoint is uploaded as `INCOMPLETE_<...>.pt`.
- **Open-loop checks.** `openloop_watcher.sh` runs on evc103 inside the job and writes `openloop/<suite>_stage{1,2}.json` per finished suite.
- **Latency of the trained policy.** Use `latency/measure_lilawam_libero_latency.py`. It must run with the GPU idle, after training.
- **Report.** `latency/make_report.py` regenerates `plans/performance/Public_WAM_Latency_20261030.md` from the result files. `latency/finalize_to_repo.sh "<msg>"` copies everything into the LoopWAM_elastic worktree and commits; the push is separate.
- **Paper recipe** (user, 2026-10-30: "Check the paper LilaWAM and follow their training recipe"; arXiv 2608.03701, Sec. 4.1-4.2):
  - one policy per suite;
  - third-person + wrist views at 256x256;
  - chunk 12;
  - AdamW(0.9, 0.99, wd 0.01), batch 128, clip 1.0, lambda_ff 0.5, logit-normal t;
  - stage-wise cosine LR, peak 2e-4 then peak 5e-5.

  The paper does NOT state, so these were chosen here:
  - epochs per stage: 5 + 2, sized to fit job 1094056;
  - LR floors: stage 1 5e-5 (public config), stage 2 5e-6;
  - two-view fusion: side by side;
  - data cleaning: none.
- **Norm stats per suite:** `/groups/yshang/an221229/data/LiLaWAM_libero_ours_stats_<suite>.json`.
- **Training.** Runs in the user's interactive job 1094056 (evc103, step 7). The srun client is on evuser2, so it does not depend on any Claude session. Script: `train_job.sh`. Progress: `train_job.log`. Per-run logs: `logs/<suite>_stage{1,2}.log`.
  - Order: libero_10, goal, object, spatial. For each suite, stage 1 runs `configs/<suite>_stage1.yaml` for 5 epochs; then stage 2 runs `--init_from` stage-1 epoch 5 with `configs/<suite>_stage2.yaml` for 2 epochs.
  - Final checkpoint per suite: `<suite>/stage2/sft_*/checkpoint_epoch_2.pt`.
  - Speed: about 0.625 s/step at batch 128, and all 4 suites total 277,713 frames. That is about 166 min including startups, so the expected finish is about 12:25-12:30; the job ends 12:51:40.
  - The joint 4-suite run in `../libero_ours_train` was cancelled (SUPERSEDED.txt).

## Remaining (needs a new GPU allocation)
1. Check `train_job.log`: every suite should show "stage2 rc=0" with a checkpoint. If the job ended early, resume the unfinished stage with `train_libero.py --config <that stage yaml> --norm_stats_path <suite stats> --save_dir <...> --resume <last ckpt>`.
2. **LIBERO eval adapter.** Write one for our matched protocol: 4 suites x 10 tasks x 50 trials x seeds 42/43/44 = 6,000 episodes, the same runner and settings as the FastWAM/LoopWAM evals. Requirements:
   - Model build: follow `robotwin_infer.RobotWinInference`, using each suite's stage-2 config and checkpoint and that suite's stats JSON. Every episode of a suite uses that suite's policy. Also use `task_cond` = `<taskcond dir>/<suite>/<task dir>/task_cond.npy`. Task dir = task text lowercased, with non-alphanumerics replaced by `_`.
   - Inputs: `libero_data.libero_obs_to_inputs(obs)` gives the composite (both views flipped `[::-1, ::-1]`, 256x256 INTER_AREA, `[agentview | wrist]`) and the 8-D state. Then ImageNet-normalize, as `measure_lilawam_libero_latency.py` does.
   - Output: the action is already in the raw env convention. Do NOT apply FastWAM's gripper flip.
   - Smoothing: the public B-spline smoothing (on by default) acts on all 7 dims, but barely changes predicted chunks (see the open-loop results). The execution horizon must still be chosen: The paper gives none for LIBERO; RoboTwin executes 16 of 32. Pick before the full run, and record the choice.
   - Open-loop check: done per suite (`openloop/`). Still smoke-test a few closed-loop episodes before launching 6,000.
3. Report with our matched tables, as done for the other models. Paper reference (per-suite policies, 2 views, 50 rollouts per task): Spatial 98.0, Object 98.8, Goal 97.2, Long 94.2, avg 97.1.

## Matched LIBERO eval (added 10-30 11:15; user: "Monitor the new interactive slurm ... run eval ... maximize the evaluation speed")
- **Allocation:** the user's interactive job **1102933** (2 H100, 64 CPUs, 512 GB, 2 h). It was pending, expected start 12:51:40. The user cancelled 1102308/1102309. Do NOT use 990547.
- **Automatic launch.** `eval_login_watcher.sh` runs on evuser2 under setsid; log `eval_login_watcher.log`.
  - It polls squeue. When the job is RUNNING it calls `srun --jobid=1102933 --overlap --cpus-per-task=64 --gres=gpu:2 --mem=0 bash eval_lilawam_step.sh`, logging to `eval_step_1102933.log`.
  - The watcher holds the user's GitHub PAT only in its process environment, passed through ssh stdin and never written to disk. The step copies it to a shell variable, unsets it before any Python starts, and uses it only for the final `latency/push_repo.sh`.
- **The step:**
  - waits for training "all done";
  - evaluates every suite that has `stage2/sft_*/checkpoint_epoch_2.pt` (up to 3 attempts; resumable through group files);
  - then runs `update_hf_readme.py` (HF model card + eval JSONs), `latency/finalize_to_repo.sh` (report regenerated with the eval table, local commit) and `push_repo.sh`.
- **Runner:** `/groups/yshang/an221229/code/lilawam_libero/evaluate_lilawam_matched.py`.
  - Protocol constants, claims, heartbeat and `summarize()` are imported unchanged from the LoopWAM runner `evaluate_v0_matched.py`.
  - Speed: one batched bf16 inference server per GPU (DINOv3 once per batch, then each suite's action model). libero_10 and goal are served by GPU 0, spatial and object by GPU 1. 56 CPU-only simulator workers (nproc − 8) run one env each, with EGL spread over both GPUs.
  - LiLa-WAM choices: execute 6 of 12 actions per query; gripper sign(g), already raw LIBERO; every query of a group starts from the same noise (CPU generator seeded with the group seed); public B-spline smoothing on.
  - Output: `eval_matched/` (manifest.json, groups/*.json, runs/, summary.json).
- **Pre-checks on evc103** (sharing the GPU with training), in `../eval_smoke/`:
  - the runner works end to end and summarize() passes;
  - task-name to task-vector mapping verified for all 40 tasks;
  - eval image and state equal the training format (`../eval_debug/compare_eval_top_train_bottom.png`);
  - sampling equals the public `_predict_chunk`.
  - **Closed-loop libero_10: 0/20** (10 tasks × 2 episodes, seed 42). The robot reaches toward the objects but rarely grasps (`../eval_debug/closed_loop_montage.png`).
  - The likely cause is training budget: 5+2 epochs, about 5.7k updates for libero_10, with loss still falling every epoch. Open-loop MAE is only about 2× better than the mean-action baseline.
- **If the step failed or the job ended early:** rerun the same srun command inside any allocation the user assigns. Finished groups are kept.
- **Goal closed-loop pre-check (11:10): 1/20.** Task 4 "put the bowl on top of the cabinet" succeeded in 101 steps; all others hit 400 steps.
  - So the pipeline can complete tasks; the policies are weak. Goal had about 3k updates in total.
  - Goal uploaded to HF at commit 595b21b8 (sha256 match).
  - If the user wants paper-level numbers, the suites need far longer training. Resume from the stage-1/stage-2 checkpoints.
