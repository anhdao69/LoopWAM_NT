#!/usr/bin/env bash
# Usage: run_until_done.sh <lane> <gpu list e.g. 0 or 0,1> <workers per gpu>
# Resumable elastic-stack matched evaluation (LoopWAM-elastic @ eval/matched-pro6000-20261101): the six conditions
# requested 2026-11-01, E0 at K_a 3/5/6 x NFE10 (inference-only test-time depths) and the unmodified v0 4/4 parent
# (plan Z10, no distillation) at [4,4], [4]x5 and [1,1]. Several lanes (one per GPU) share one output root: groups are
# claimed atomically, so lanes never change per-group seeding and a group is written by exactly one record file.
# Repeats attempts until every condition has its summary. Meant to run as its own systemd user service.
set -uo pipefail
LANE=$1; GPUS=$2; WPG=$3
ROOT=/data/anh/code/wam_baselines
OUT=$ROOT/runs/elastic_matched_20261101
LOG=$ROOT/logs/elastic_matched_20261101_$LANE.log
CK=$ROOT/checkpoints/loopwam_hf
MIN_RAM_GB=${MIN_RAM_GB:-8}
GPU_MIB_PER_WORKER=${GPU_MIB_PER_WORKER:-4500}
# Heaviest first (fewer stragglers at the end); the runner orders groups condition-major, Long first.
CONDITIONS=${CONDITIONS:-E0:flat_k6_n10,E0:flat_k5_n10,E0:flat_k3_n10,Z10:parent5,Z10:flat_k1_n2,Z10:parent2}
EXPECTED=E0=5569c47fa6c91e136a611d2b819a053c94f6b2956a7f602bbed4db23c69f60e8,Z10=1c659a7c2f6970c3ea5972a8fba4abd9db35857c83a67a053c0ee35555fb602b
mkdir -p "$OUT"
say() { echo "{\"event\": \"$1\", \"lane\": \"$LANE\", \"time\": $(date +%s)${2:+, $2}}" >> "$LOG"; }
done_all() {
  local c run s n=0 k
  for c in ${CONDITIONS//,/ }; do
    run=${c%%:*}; s=${c#*:}
    k=$(ls -d "$OUT"/final_rollout/matched6000pro__"$run"__"$s"__* 2>/dev/null | head -1)
    [ -n "$k" ] && [ -f "$k/summary.json" ] && n=$((n + 1))
  done
  [ "$n" -eq "$(echo "${CONDITIONS//,/ }" | wc -w)" ]
}
gpu_ready() {
  local g free
  for g in ${GPUS//,/ }; do
    free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits) || return 1
    [ "$free" -ge $((WPG * GPU_MIB_PER_WORKER)) ] || return 1
  done
}
release_dead_claims() {  # heartbeat files of our own runners whose process is gone: their claims become takeable now
  local f pid
  for f in "$OUT"/runners/"$(hostname)"-nojob-*.json; do
    [ -e "$f" ] || continue
    pid=$(basename "$f" .json | awk -F- '{print $(NF-1)}')
    kill -0 "$pid" 2>/dev/null || { rm -f "$f"; say released "\"token\": \"$(basename "$f" .json)\""; }
  done
}
for attempt in $(seq 1 200); do
  done_all && { say complete; exit 0; }
  until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge "$MIN_RAM_GB" ] && gpu_ready; do sleep 60; done
  release_dead_claims
  say attempt "\"attempt\": $attempt, \"gpus\": \"$GPUS\", \"workers_per_gpu\": $WPG"
  before=$(grep -c '"event": "group"' "$LOG" 2>/dev/null); before=${before:-0}
  CUDA_VISIBLE_DEVICES=${GPUS%%,*} "$ROOT/matched_eval_elastic/launch.sh" scripts/evaluate_rt_matched.py \
      --tag final_rollout --key-prefix matched6000pro --conditions "$CONDITIONS" --expected-sha "$EXPECTED" \
      --test-time-schedules flat_k3_n10,flat_k5_n10,flat_k6_n10 --zero-shot-parent "Z10=$CK/v0-video4-action4/policy.pt" \
      --weights-root "$CK/Elastic" --output-root "$OUT" --vae-path "$ROOT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" \
      --text-cache-dir "$ROOT/text_cache/loopwam" --text-provenance "$ROOT/text_cache/loopwam/provenance.json" \
      --gpus "$GPUS" --workers-per-gpu "$WPG" --threads 1 --no-videos \
      --build-lock "$ROOT/runs/build.lock" --build-min-ram-gb "${BUILD_MIN_RAM_GB:-6}" >> "$LOG" 2>&1
  status=$?
  after=$(grep -c '"event": "group"' "$LOG" 2>/dev/null); after=${after:-0}
  say exited "\"attempt\": $attempt, \"status\": $status, \"groups_scored\": $((after - before))"
  done_all && { say complete; exit 0; }
  # Nothing scored: the remaining groups are claimed by another live lane; wait for it instead of spinning.
  [ $((after - before)) -eq 0 ] && sleep 300 || sleep 60
done
exit 1
