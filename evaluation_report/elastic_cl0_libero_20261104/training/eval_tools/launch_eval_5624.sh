#!/usr/bin/env bash
# Second CL-0 eval lane "cl0eval_b" on job 5624 (GPUs 4-7 of worker-2, idle since the GR1 eval finished), same
# runner, conditions, workers/GPU and MPS as lane cl0eval (job 5623). The lanes share the queue through the runner's
# atomic per-group claims (O_EXCL claim files owned by a runner token; taken over only when the owner's 60 s
# heartbeat is >10 min stale) and skip groups whose record exists, so no group is evaluated twice.
E=/mnt/data/vmo-ai-task/anhdh35/elastic_cl0/eval
cd /mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_cl0 || exit 1
if squeue -s -j 5624 -h -o %j | grep -qE "cl0-eval|gr1"; then echo "job 5624 already has an eval step"; exit 1; fi
setsid nohup srun --jobid=5624 --overlap -n1 -c60 --mem=0 --gres=gpu:4 -J cl0-eval-b \
  bash plans/elastic_cl0/eval/eval_lane.sh cl0eval_b 0,1,2,3 12 > $E/logs/lane_cl0eval_b.log 2>&1 < /dev/null &
echo "launched srun pid $! on $(hostname) at $(date -u)"
