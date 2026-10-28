#!/usr/bin/env bash
# Runs on login node evuser2 (nohup setsid; log logs/elastic_watcher.log). User request 2026-10-28: train every
# LoopWAM_elastic.md variant in the user's interactive job 1080499 (evc102, 4xH100). Holds the srun client of each
# 2-GPU lane (GPUs 0,1 and 2,3) so training outlives any session. Never touches any other job.
set -uo pipefail
WORK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work
RUNS=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic
JOB=1080499
log() { echo "[$(date -Is)] $*"; }
state() { squeue -h -j "$1" -o %T 2>/dev/null; }
complete() { [ -f "$RUNS/$1/status.json" ] && grep -q '"status": "complete"' "$RUNS/$1/status.json"; }
all_complete() { for r in E0 E2 E1 SP11 F0-PB; do complete "$r" || return 1; done; }
lane() {  # NAME GPUS
  for attempt in 1 2 3 4 5 6; do
    [ "$(state $JOB)" = RUNNING ] || { log "job $JOB is not running; lane $1 stops"; return; }
    all_complete && { log "all runs complete; lane $1 done"; return; }
    log "lane $1 (GPUs $2) step attempt $attempt on $(squeue -h -j $JOB -o %N)"
    srun --jobid=$JOB --overlap --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:4 --mem=0 \
         bash "$WORK/lane.sh" "$1" "$2" >> "$WORK/logs/lane_$1.log" 2>&1
    log "lane $1 step exited rc=$?"
    sleep 300
  done
}
log "elastic watcher pid $$ on $(hostname); job $JOB"
lane A 0,1 &
sleep 20
lane B 2,3 &
wait
log "watcher done"
