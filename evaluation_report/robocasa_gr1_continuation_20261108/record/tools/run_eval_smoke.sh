#!/bin/bash
# Eval smoke (2026-11-06): production epoch_05.pt, first 2 tasks x 1 episode, exact eval_epoch.sh command on GPUs 4-7.
G=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101; CR=$G/continuation_20261106
TASKS=$(cd /mnt/data/vmo-ai-task/anhdh35/LoopWAM_gr1_eval_code && PYTHONPATH=src /mnt/data/vmo-ai-task/anhdh35/FastWAM/.venv/bin/python -c 'from fastwam.datasets.robocasa_gr1 import GR1_ENV_NAMES as N; print(",".join(N[:2]))')
echo "tasks $TASKS"
cd /tmp; exec srun --jobid=5758 --overlap -n1 -c120 --cpu-bind=none --gres=gpu:8 -J training_vln   --export=ALL,EVAL_CK=$G/train/production/checkpoints/epoch_05.pt,EVAL_STEP=117580,EVAL_OUT=$CR/gates/eval_smoke,EVAL_TASKS=$TASKS,EVAL_EPISODES=1,EVAL_PORT=6790,EVAL_ALLOW_BUSY=1   bash $CR/tools/eval_epoch.sh 6 1
