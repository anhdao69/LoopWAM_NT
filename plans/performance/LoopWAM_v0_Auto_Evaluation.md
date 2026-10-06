# v0 automatic evaluation and allocation release

## Requested behavior

After the current fresh v0 run finishes, evaluate its final checkpoint on LIBERO-Long and automatically cancel interactive allocation **4689**. Evaluation uses the existing protocol: 100 episodes (10 per task), videos, two GPUs, 700 policy steps, 30 settling steps, ten denoising steps, four recurrent loops, and replanning every ten actions.

## Implementation

- `scripts/watch_v0_evaluate.py` runs as a detached process on the login node, outside Slurm. It polls every 30 seconds and survives cancellation of the target allocation.
- Wait for training step **4689.61** to exit and for complete training status: 7,250 updates, ten epochs, 926,780 real windows, and the final checkpoint. The watcher understands v0's original zero-based `micro` field in `trainer_state.json`; the evaluator separately validates the checkpoint's `next_micro` field.
- Launch `scripts/evaluate_v0_in_allocation.sh` in the same allocation, using its two GPUs after training releases them.
- Verify all 100 unique task/episode results, final v0 checkpoint metadata/hash, and all saved videos before issuing `scancel 4689`.
- Bind cancellation to the allocation ID, owner and start time. Refuse evaluation/cancellation if an additional Slurm step appears. Training/evaluation errors retain the allocation and write `status.json` with the error.
- A file lock prevents duplicate watchers; existing status/evaluation outputs prevent implicit reruns or overwrites. No running training source is changed.

Training directory:
`/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_scratch_fast_bs128_20261005`

Watcher/evaluation directory:
`/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_auto_eval_job4689`

Expected files: `status.json`, `watcher.log`, `evaluation.log`, `inference/summary.json`, `inference/videos/`.

## Verification

Focused cancellation tests cover successful ordering and refusal after incomplete/resumed/wrong-job training, failed or partial evaluation, missing videos, duplicate episodes, additional steps, or changed allocation identity. The success path uses synthetic evaluation files and a fake Slurm interface; tests never cancel a real allocation.

Local watcher tests passed. Three related rollout tests could not run in the local shell because its Python environment lacks `pyarrow`; verification uses the existing server environment instead. Full server test results and live activation evidence are recorded below after verification.

Server verification: **165 passed, 7 CUDA-only skips**, with three upstream robosuite deprecation warnings. Shell syntax and Python compilation checks passed. Read-only live preflight matched allocation4689/owner/start time and found only the interactive shell and training step. The existing v0 checkpoint at update4200 passed evaluator architecture and normalization/contract checks (v0, four loops, two training ranks, global128). Training is still running, so final evaluation and cancellation have not occurred.

## Active watcher

Activated on login-0 as detached PID **321486**, parent PID1, pinned source **17ef2c6c6fad86d726b66b362a618b9e7b1b564c**. Verified stage `waiting_for_training`, current v0 update4260/7250, and no evaluation directory or evaluation GPU step yet. Existing training step4689.61 remains running. Source is synced locally and pushed to `LoopWAM_NT`.

The watcher is armed to execute the future evaluation and cancellation; neither has happened at activation. [Activation snapshot](../evidence/v0_auto_eval/activation_status.json).
