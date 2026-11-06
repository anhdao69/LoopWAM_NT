#!/bin/bash
# Login-node keeper of queue_ep5.sh (setsid nohup). Every 10 min: unless the queue is done, relaunch the orchestrator when
# its heartbeat is older than 20 min (its step died); the orchestrator is idempotent and locked. At most 5 relaunches.
# Usage: keeper_ep5.sh CLONE REVISION
CLONE=$1; REV=$2
B=/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin/ep5; N=0
log() { echo "[$(date -u +%FT%TZ)] keeper: $*" >> "$B/keeper.log"; }
launch() {
  cd /tmp && setsid nohup srun --jobid=5758 --overlap -n1 -c2 --gres=none -J training_vln bash "$CLONE/plans/elastic_robotwin/queue_ep5.sh" "$REV" \
    >> "$B/logs/queue.stdout" 2>&1 < /dev/null &
  log "launched queue (srun pid $!)"
}
log "keeper start: host $(hostname) boot $(cat /proc/sys/kernel/random/boot_id) pid $$"
while sleep 600; do
  phase=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "$B/STATUS.json" 2> /dev/null)
  case "$phase" in done|done_with_problems) log "queue $phase: keeper exits"; exit 0 ;; esac
  squeue -j 5758 -h -o %T 2> /dev/null | grep -q RUNNING || { log "job 5758 not running: keeper exits"; exit 0; }
  age=$(( $(date +%s) - $(stat -c %Y "$B/heartbeat" 2> /dev/null || echo 0) ))
  if [ "$age" -gt 1200 ]; then
    [ "$N" -ge 5 ] && { log "heartbeat ${age}s old and 5 relaunches used: keeper gives up"; exit 1; }
    N=$((N + 1)); log "heartbeat ${age}s old (phase '$phase'): relaunch $N"; launch; sleep 600
  fi
done
