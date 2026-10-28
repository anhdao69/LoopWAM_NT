#!/usr/bin/env bash
# Runs inside an overlapping step of job $1 (launched by dispatcher.py shortly before the job ends, or at campaign
# end). Gracefully stops this job's elastic lanes/runs (recovery state is saved), then frees the node-local cache.
set -uo pipefail
JOB=$1
WORK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work
mine() { for pid in $(pgrep -u "$USER" -f "$WORK/lane.sh|scripts/run_elastic.sh"); do
           tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -qx "SLURM_JOB_ID=$JOB" && echo "$pid"; done; }
pids=$(mine)
echo "[$(date -Is)] job $JOB: stopping ${pids:-nothing}"
[ -n "$pids" ] && kill -TERM $pids 2>/dev/null
for i in $(seq 90); do [ -z "$(mine)" ] && break; sleep 10; done
[ -n "$(mine)" ] && echo "[$(date -Is)] job $JOB: processes still running after 15 min: $(mine)"
if [ -z "$(mine)" ] && [ -d "/dev/shm/elastic_latents_$JOB" ]; then
  rm -rf "/dev/shm/elastic_latents_$JOB" && echo "[$(date -Is)] job $JOB: removed /dev/shm/elastic_latents_$JOB"
fi
