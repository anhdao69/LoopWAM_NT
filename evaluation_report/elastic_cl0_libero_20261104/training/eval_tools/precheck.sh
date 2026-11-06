#!/usr/bin/env bash
# Pre-launch check inside job 5623: GPUs idle, no stale eval processes/MPS, output dir state.
E=/mnt/data/vmo-ai-task/anhdh35/elastic_cl0/eval
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "--- compute apps"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "--- eval/mps processes"; pgrep -af "evaluate_rt_matched|nvidia-cuda-mps|preflight_rt_graph" | grep -v pgrep || echo none
echo "--- /dev/shm"; ls /dev/shm | grep elastic_cl0 || echo none
echo "--- final_rollout"; ls $E/final_rollout 2>/dev/null | wc -l
echo "--- gates"; ls $E/gates
echo "--- free mem"; free -g | head -2; nproc
