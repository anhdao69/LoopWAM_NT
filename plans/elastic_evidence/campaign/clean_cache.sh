#!/usr/bin/env bash
# Runs inside an overlapping step of job $1 once all training is complete: removes the node-local latent cache when
# no training lane of this job is still running (evaluation lanes are left alone).
set -uo pipefail
JOB=$1
WORK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work
for pid in $(pgrep -u "$USER" -f "$WORK/lane.sh|scripts/run_elastic.sh"); do
  tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -qx "SLURM_JOB_ID=$JOB" && { echo "job $JOB: training lane $pid still running; cache kept"; exit 0; }
done
rm -rf "/dev/shm/elastic_latents_$JOB" && echo "job $JOB: removed /dev/shm/elastic_latents_$JOB"
