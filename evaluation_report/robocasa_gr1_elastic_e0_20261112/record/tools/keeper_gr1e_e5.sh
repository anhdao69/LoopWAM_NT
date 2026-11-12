#!/bin/bash
# keeper_gr1e_e5.sh: keeper_gr1e.sh for launcher_gr1e_e5.sh / orchestrate_gr1e_e5.sh (2026-11-10); otherwise identical.
# Login-node keeper of the GR1-E0 x5 orchestrator (setsid nohup). Every 10 min: unless the orchestrator reached a final
# phase, relaunch it when its heartbeat is older than 20 min (its step died); the orchestrator is idempotent and locked.
# It relaunches launcher_gr1e.sh, which waits for the GR1 continuation if needed and then starts the orchestrator with
# the revision pinned in $B/REVISION. At most 5 relaunches. Usage: keeper_gr1e.sh
B=/mnt/data/vmo-ai-task/anhdh35/elastic_gr1; T=$B/tools; N=0
log() { echo "[$(date -u +%FT%TZ)] keeper: $*" >> "$B/keeper.log"; }
launch() {
  cd /tmp && setsid nohup srun --jobid=5758 --overlap -n1 -c2 --gres=none -J training_vln bash "$T/launcher_gr1e_e5.sh" \
    >> "$B/logs/orchestrator.stdout" 2>&1 < /dev/null &
  log "launched launcher/orchestrator (srun pid $!)"
}
log "keeper (e5) start: host $(hostname) boot $(cat /proc/sys/kernel/random/boot_id) pid $$"
while sleep 600; do
  phase=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "$B/STATUS.json" 2> /dev/null)
  case "$phase" in done|done_with_problems|failed|failed_validation) log "orchestrator $phase: keeper exits"; exit 0 ;; esac
  squeue -j 5758 -h -o %T 2> /dev/null | grep -q RUNNING || { log "job 5758 not running: keeper exits"; exit 0; }
  age=$(( $(date +%s) - $(stat -c %Y "$B/heartbeat" 2> /dev/null || echo 0) ))
  if [ "$age" -gt 1200 ]; then
    [ "$N" -ge 5 ] && { log "heartbeat ${age}s old and 5 relaunches used: keeper gives up"; exit 1; }
    N=$((N + 1)); log "heartbeat ${age}s old (phase '$phase'): relaunch $N"; launch; sleep 600
  fi
done
