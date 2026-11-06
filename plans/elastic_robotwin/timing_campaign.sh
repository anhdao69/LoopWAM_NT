#!/usr/bin/env bash
# Timing campaign of RT50-E0/E1/E2 with the real trainer, inside one 8-GPU step (-c120 --cpu-bind=none --gres=gpu:8):
#   1) 8 GPUs, microbatch 16 (accumulation 1), E0 -> E1 -> E2
#   2) 4 GPUs (0-3), microbatch 32 (accumulation 1), E0 -> E1 -> E2
#   3) two concurrent 4-GPU lanes (E0 on 0-3, E1 on 4-7), microbatch 32: interference of a 2-lane layout
# 108 updates per run = 12 FLAT9 blocks (block 1 is warm-up) and one recovery save at update 100. Then summary.json/md.
set -uo pipefail
D=$(cd "$(dirname "$0")" && pwd); source "$D/env.sh"; T=$ERT_WORK/timing; mkdir -p "$T"
U=${ERT_TIMING_UPDATES:-108}
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$T/campaign.log"; }
wait_free() {
  for i in $(seq 120); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)" = 0 ] && return 0; sleep 5
  done
  log "GPUs still busy after 10 min"; return 1
}
log "campaign start host $(hostname) job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?} updates $U"
for RUN in RT50-E0 RT50-E1 RT50-E2; do wait_free || exit 1; bash "$D/time_run.sh" "$RUN" 0,1,2,3,4,5,6,7 16 "$U" w8_mb16 >> "$T/campaign.log" 2>&1; done
for RUN in RT50-E0 RT50-E1 RT50-E2; do wait_free || exit 1; bash "$D/time_run.sh" "$RUN" 0,1,2,3 32 "$U" w4_mb32 >> "$T/campaign.log" 2>&1; done
wait_free || exit 1
bash "$D/time_run.sh" RT50-E0 0,1,2,3 32 "$U" w4x2_mb32 >> "$T/campaign.log" 2>&1 &
bash "$D/time_run.sh" RT50-E1 4,5,6,7 32 "$U" w4x2_mb32 >> "$T/campaign.log" 2>&1 &
wait
python "$D/summarize_timing.py" "$T" "$T/summary.json" "$T/summary.md" >> "$T/campaign.log" 2>&1
log CAMPAIGN_DONE
