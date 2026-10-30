#!/usr/bin/env bash
# Runs on evuser2 (setsid nohup). Waits for the user's interactive job 1102933 to start, then launches the eval step
# in it with srun --overlap. The srun client lives here, so it does not depend on any Claude session.
RUN=/groups/yshang/an221229/checkpoints/wam_baselines/lilawam/libero_ours_paper_recipe
J=1102933
echo "[$(date -Is)] watcher start for job $J"
while true; do
  st=$(squeue -h -j $J -o %T 2>/dev/null)
  if [ "$st" = "RUNNING" ]; then break; fi
  if [ -z "$st" ]; then echo "[$(date -Is)] job $J is gone (not pending/running); exiting"; exit 1; fi
  sleep 20
done
echo "[$(date -Is)] job $J RUNNING on $(squeue -h -j $J -o %N); launching eval step"
sleep 20
srun --jobid=$J --overlap --nodes=1 --ntasks=1 --cpus-per-task=$(squeue -h -j $J -o %C) --cpu-bind=none \
  --gres=gpu:$(scontrol show job $J | grep -oE 'gres/gpu=[0-9]+' | head -n 1 | cut -d= -f2) --mem=0 \
  bash $RUN/eval_lilawam_step.sh > $RUN/eval_step_$J.log 2>&1
echo "[$(date -Is)] srun rc=$?"
