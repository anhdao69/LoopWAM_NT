#!/usr/bin/env bash
# Autoscaler for the 2026-11-01 elastic matched lanes on the shared host (host RAM, not GPU, is the binding limit:
# one worker holds ~3.1 GB, 1.8 GB of it the MuJoCo/EGL env; the lane's runner process ~0.7 GB).
# Every 60 s: when MemAvailable stayed >= UP2_GB for 3 checks, start a 2-worker lane; else when it stayed >= UP1_GB,
# a 1-worker lane. A lane is its own systemd unit wam-elastic-l<N> running run_until_done.sh on one GPU (the GPU with
# fewer of our lanes that has room). After any start, or after the RAM guard stopped a lane, wait SETTLE seconds.
# Scale-down is ram_guard.sh (it stops the newest wam-elastic-* unit). Heartbeat files of our dead runners are removed
# so their claimed groups are retaken at once (a group interrupted mid-way reruns from episode 0, same seed).
# No lane is added once fewer than 2 groups are unclaimed (720 groups = 6 conditions x 120). Exits when every condition
# has its summary.
set -uo pipefail
ROOT=/data/anh/code/wam_baselines
OUT=$ROOT/runs/elastic_matched_20261101
LOG=$ROOT/logs/elastic_autoscale_20261101.log
UP2_GB=${UP2_GB:-11}; UP1_GB=${UP1_GB:-7}
STEP_GB=${STEP_GB:-0}  # extra GB required per worker already running beyond the second
MAX_WORKERS=${MAX_WORKERS:-16}; GPU_MIB_PER_WORKER=${GPU_MIB_PER_WORKER:-4500}; SETTLE=${SETTLE:-600}
MPS_ENV=(-E CUDA_MPS_PIPE_DIRECTORY=/dev/shm/anh_wam_mps/pipe -E CUDA_MPS_LOG_DIRECTORY=/dev/shm/anh_wam_mps/log)
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
avail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
lanes() { systemctl --user list-units 'wam-elastic-l*' --state=active --no-legend --plain | awk '{print $1}'; }
workers_of() { systemctl --user show "$1" -p Description --value | awk '{print $NF}'; }
gpu_of() { systemctl --user show "$1" -p Description --value | awk '{print $(NF-1)}'; }
n_summaries() { ls "$OUT"/final_rollout/*/summary.json 2>/dev/null | wc -l; }
release_dead_claims() {
  local f pid
  for f in "$OUT"/runners/"$(hostname)"-nojob-*.json; do
    [ -e "$f" ] || continue
    pid=$(basename "$f" .json | awk -F- '{print $(NF-1)}')
    kill -0 "$pid" 2>/dev/null || { rm -f "$f"; say "released heartbeat of dead runner $(basename "$f" .json)"; }
  done
}
pick_gpu() {  # GPU with fewer of our lanes among those with room for $1 workers
  local need=$1 g free best= bn=999 n u
  for g in 0 1; do
    free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits) || continue
    [ "$free" -ge $((need * GPU_MIB_PER_WORKER + 4000)) ] || continue
    n=0; for u in $(lanes); do [ "$(gpu_of "$u")" = "$g" ] && n=$((n + $(workers_of "$u"))); done
    [ "$n" -lt "$bn" ] && { bn=$n; best=$g; }
  done
  echo "$best"
}
next=$(( $(cat "$OUT/.lane_counter" 2>/dev/null || echo 0) + 1 ))
say "autoscaler started (2-worker lane at >= ${UP2_GB} GB, 1-worker at >= ${UP1_GB} GB, +${STEP_GB} GB per running worker beyond 2, settle ${SETTLE} s, max ${MAX_WORKERS} workers)"
hi2=0; hi1=0; last=0; prev_lanes=$(lanes | wc -l)
while [ "$(n_summaries)" -lt 6 ]; do
  release_dead_claims
  cur=$(lanes | wc -l)
  [ "$cur" -lt "$prev_lanes" ] && { last=$(date +%s); say "lanes ${prev_lanes} -> ${cur}; settling"; }
  prev_lanes=$cur
  a=$(avail_gb)
  total=0; for u in $(lanes); do total=$((total + $(workers_of "$u"))); done
  extra=$(( total > 2 ? (total - 2) * STEP_GB : 0 ))
  [ "$a" -ge $((UP2_GB + extra)) ] && hi2=$((hi2 + 1)) || hi2=0
  [ "$a" -ge $((UP1_GB + extra)) ] && hi1=$((hi1 + 1)) || hi1=0
  # A lane only helps while unclaimed groups remain; stop adding lanes in the tail.
  done_g=$(ls "$OUT"/final_rollout/*/groups/*.json 2>/dev/null | wc -l)
  claimed=$(ls "$OUT"/final_rollout/*/claims/*.claim 2>/dev/null | wc -l)
  open=$((720 - done_g - claimed))
  if [ $(( $(date +%s) - last )) -ge "$SETTLE" ] && [ "$hi1" -ge 3 ] && [ "$total" -lt "$MAX_WORKERS" ] && [ "$open" -ge 2 ]; then
    w=1; [ "$hi2" -ge 3 ] && [ $((total + 2)) -le "$MAX_WORKERS" ] && w=2
    g=$(pick_gpu "$w")
    if [ -n "$g" ]; then
      unit=wam-elastic-l$next
      if systemd-run --user --unit="$unit" --description="elastic matched lane l$next gpu $g $w" \
          -p OOMPolicy=continue -p OOMScoreAdjust=500 -E MIN_RAM_GB=6 -E BUILD_MIN_RAM_GB=4 "${MPS_ENV[@]}" \
          "$ROOT/matched_eval_elastic/run_until_done.sh" "l$next" "$g" "$w" >/dev/null 2>&1; then
        say "started $unit on GPU $g with $w worker(s) (MemAvailable ${a} GB, our workers before: $total)"
        echo "$next" > "$OUT/.lane_counter"; next=$((next + 1)); last=$(date +%s); hi1=0; hi2=0
        prev_lanes=$(lanes | wc -l)
      else
        say "failed to start $unit"
      fi
    fi
  fi
  sleep 60
done
say "all six conditions have summaries; autoscaler exits"
