#!/bin/bash
# Login-node keeper for the GR1 epochs 6-10 orchestrator (setsid nohup on the login node). Every 10 min: if the pipeline
# is neither done nor failed and the orchestrator heartbeat is older than 20 min (its srun step died), relaunch it; the
# orchestrator is idempotent and holds a node-local lock, so a relaunch next to a live one exits at once. At most 5.
CR=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101/continuation_20261106
N=0
log() { echo "[$(date -u +%FT%TZ)] keeper: $*" >> "$CR/keeper.log"; }
launch() {
  cd /tmp && setsid nohup srun --jobid=5758 --overlap -n1 -c2 --gres=none -J training_vln bash "$CR/tools/orchestrate.sh" \
    >> "$CR/logs/orchestrator.stdout" 2>&1 < /dev/null &
  log "launched orchestrator (srun pid $!)"
}
log "keeper start: host $(hostname) pid $$"
while sleep 600; do
  phase=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "$CR/STATUS.json" 2> /dev/null)
  case "$phase" in done|failed) log "pipeline $phase: keeper exits"; exit 0 ;; esac
  squeue -j 5758 -h -o %T 2> /dev/null | grep -q RUNNING || { log "job 5758 not running: keeper exits"; exit 0; }
  age=$(( $(date +%s) - $(stat -c %Y "$CR/heartbeat" 2> /dev/null || echo 0) ))
  if [ "$age" -gt 1200 ]; then
    [ "$N" -ge 5 ] && { log "heartbeat ${age}s old and 5 relaunches used: keeper gives up"; exit 1; }
    N=$((N + 1)); log "heartbeat ${age}s old (phase '$phase'): relaunch $N"; launch; sleep 600
  fi
done
