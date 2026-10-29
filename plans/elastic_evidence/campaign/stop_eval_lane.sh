#!/usr/bin/env bash
# Runs inside an overlapping step of job $1: stops evaluation lane $2 of that job (dispatcher preemption so that a
# training run can take its GPUs). The lane's runner is killed; its in-flight task/seed groups are re-run elsewhere.
set -uo pipefail
JOB=$1; LANE=$2
WORK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work
mine() { for pid in $(pgrep -u "$USER" -f "$WORK/eval_lane.sh $LANE "); do
           tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -qx "SLURM_JOB_ID=$JOB" && echo "$pid"; done; }
pids=$(mine)
echo "[$(date -Is)] job $JOB: stopping eval lane $LANE: ${pids:-nothing}"
[ -n "$pids" ] && kill -TERM $pids 2>/dev/null
for i in $(seq 36); do [ -z "$(mine)" ] && break; sleep 5; done
[ -n "$(mine)" ] && echo "[$(date -Is)] job $JOB: eval lane $LANE still running after 3 min"
echo "[$(date -Is)] job $JOB: eval lane $LANE stopped"
