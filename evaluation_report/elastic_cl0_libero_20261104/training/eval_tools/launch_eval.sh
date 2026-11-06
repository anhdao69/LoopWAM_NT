#!/usr/bin/env bash
# Launch the CL-0 production eval lane as an overlapping step of job 5623 (GPUs 0-3, all 60 CPUs), detached.
E=/mnt/data/vmo-ai-task/anhdh35/elastic_cl0/eval
cd /mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_cl0 || exit 1
if squeue -s -j 5623 -h -o %j | grep -q cl0-eval; then echo "lane already running"; exit 1; fi
setsid nohup srun --jobid=5623 --overlap -n1 -c60 --mem=0 -J cl0-eval \
  bash plans/elastic_cl0/eval/eval_lane.sh cl0eval 0,1,2,3 12 > $E/logs/lane_cl0eval.log 2>&1 < /dev/null &
echo "launched srun pid $! on $(hostname) at $(date -u)"
