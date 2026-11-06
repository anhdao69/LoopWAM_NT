#!/bin/bash
# Login-node chain (setsid nohup), 2026-11-06: the RT50 E0/E1/E2 timing campaign as one 8-GPU step of job 5758, then
# (user: "Auto-restart GR1" after the timing tests) relaunch the GR1 epochs 6-10 orchestrator and its keeper. The
# orchestrator is idempotent: gates PASSED and the RAM cache are kept, epoch 6 starts fresh from the parent.
W=/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin
CR=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101/continuation_20261106
log() { echo "[$(date -u +%FT%TZ)] chain: $*" >> "$W/logs/chain.log"; }
log "start host $(hostname) boot $(cat /proc/sys/kernel/random/boot_id) pid $$"
cd /tmp
srun --jobid=5758 --overlap -n1 -c120 --cpu-bind=none --gres=gpu:8 -J training_vln \
  bash /mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_rt/plans/elastic_robotwin/timing_campaign.sh \
  > "$W/logs/campaign.stdout" 2>&1 < /dev/null
rc=$?
log "campaign step rc=$rc; $(grep -c 'rc=0' "$W/timing/campaign.log" 2>/dev/null) runs rc=0; $(grep -q CAMPAIGN_DONE "$W/timing/campaign.log" && echo done || echo NOT-done)"
echo "[$(date -u +%FT%TZ)] relaunch after the RT50 timing tests (user: auto-restart GR1 when they finish)" >> "$CR/orchestrator.log"
setsid nohup srun --jobid=5758 --overlap -n1 -c2 --gres=none -J training_vln bash "$CR/tools/orchestrate.sh" \
  >> "$CR/logs/orchestrator.stdout" 2>&1 < /dev/null &
log "GR1 orchestrator relaunched (srun pid $!)"
sleep 90
setsid nohup bash "$CR/tools/keeper.sh" > /dev/null 2>&1 < /dev/null &
log "GR1 keeper started (pid $!); STATUS $(tr -d '\n ' < "$CR/STATUS.json")"
