#!/usr/bin/env bash
# CL-0 elastic evaluation lane (inside an overlapping srun step of job 5623). Adapted from the reference lane
# plans/elastic_evidence/campaign/eval_lane.sh (v3) with this campaign's paths; the logic is unchanged except that
# the expected checkpoint sha256 come from eval_env.sh (pinned by the pre-evaluation audit) instead of HF receipts.
# Usage: eval_lane.sh <lane-name> <GPU list, e.g. 0,1,2,3> <workers per GPU>
# Evaluates EVAL_CONDITIONS (RUN:SCHEDULE:GATE in queue order) of every run whose training is complete. Per gate
# name, a CUDA-graph gate on the run's epoch_10.pt first (gate_one.sh: actions of all of the gate's schedules must be
# bit-exact against eager on real simulator observations, otherwise those conditions use the eager backend). Then
# the resumable matched runner (scripts/evaluate_rt_matched.py), condition-major in the listed order: task/seed
# groups are claimed atomically, a group record is written only when all 50 episodes finish, and the queue order
# never changes per-group seeding. The runner's main process is pinned to the lane's first GPU; everything runs at
# nice 10. Exits when no pending group of a ready run is left. SIGTERM stops the runner (claims expire after 10 min).
# With EVAL_MPS=1 the lane starts a private CUDA MPS daemon (node-local pipe directory given only to this lane's
# processes): workers then share each GPU concurrently instead of time-slicing.
set -uo pipefail
LANE=$1; GPUS=$2; WPG=$3
TOOLS=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_cl0/plans/elastic_cl0/eval
source $TOOLS/eval_env.sh
cd "$ELASTIC_ROOT"
mkdir -p "$EVAL_OUT/gates" "$EVAL_OUT/logs"
log() { echo "[$(date -Is)] eval lane $LANE: $*"; }
first=${GPUS%%,*}
child=""; stop=0
trap 'stop=1; [ -n "$child" ] && kill -TERM -- -"$child" 2>/dev/null' TERM
ready() {
  [ -f "$EVAL_RUNS/$1/epoch_10.pt" ] && python -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['status']=='complete' and s['update']==21700 else 1)" "$EVAL_RUNS/$1/status.json" 2>/dev/null
}
gate_backend() {  # prints cuda-graph or eager for gate name $1 of run $2; returns 1 if the gate could not be run
  local name=$1 run=$2 out=$EVAL_OUT/gates/$1.json lock=$EVAL_OUT/gates/$1.lock schedules="" c
  for c in $EVAL_CONDITIONS; do [ "${c##*:}" = "$name" ] && schedules="$schedules${schedules:+,}$(echo "$c" | cut -d: -f2)"; done
  for i in $(seq 40); do
    if [ -f "$out" ]; then
      python -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d['all_bit_exact'] is True and set(sys.argv[2].split(',')) <= set(d['schedules']) else 1)" "$out" "$schedules" \
        && echo cuda-graph || echo eager
      return 0
    fi
    log "gating $name ($run: $schedules) on GPU $first" >&2
    bash $TOOLS/gate_one.sh "$name" "$run" "$schedules" "$first" >&2; rc=$?
    [ $rc = 0 ] && continue
    [ $rc = 1 ] && { log "gate $name failed to run (see logs/gate_${name}.log)" >&2; return 1; }
    [ $(( $(date +%s) - $(stat -c %Y "$lock" 2>/dev/null || date +%s) )) -gt 1800 ] && rmdir "$lock" 2>/dev/null
    sleep 30
  done
  return 1
}
log "start on $(hostname) job ${SLURM_JOB_ID:-?} gpus $GPUS workers/gpu $WPG"
mps_dir=""
if [ "${EVAL_MPS:-0}" = 1 ]; then
  mps_dir=/dev/shm/elastic_cl0_mps_${SLURM_JOB_ID:-x}_$LANE
  rm -rf "$mps_dir"; mkdir -p "$mps_dir/pipe" "$mps_dir/log"
  export CUDA_MPS_PIPE_DIRECTORY=$mps_dir/pipe CUDA_MPS_LOG_DIRECTORY=$mps_dir/log
  if ( unset CUDA_VISIBLE_DEVICES; nvidia-cuda-mps-control -d ); then
    log "MPS daemon started ($mps_dir)"
  else
    log "MPS daemon failed to start; running without MPS"
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; rm -rf "$mps_dir"; mps_dir=""
  fi
fi
stop_mps() {
  [ -n "$mps_dir" ] || return 0
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; sleep 3; rm -rf "$mps_dir"; mps_dir=""
}
trap stop_mps EXIT
for attempt in 1 2 3 4 5; do
  [ "$stop" = 1 ] && break
  conds=""; declare -A backend_of=()
  for c in $EVAL_CONDITIONS; do
    run=${c%%:*}; rest=${c#*:}; s=${rest%%:*}; name=${rest#*:}
    ready "$run" || continue
    if [ -z "${backend_of[$name]+x}" ]; then
      backend_of[$name]=$(gate_backend "$name" "$run") || backend_of[$name]=failed
      [ "${backend_of[$name]}" = eager ] && log "$name: CUDA graph not bit-exact; using the eager backend"
    fi
    [ "$stop" = 1 ] && break
    [ "${backend_of[$name]}" = failed ] && continue
    conds="$conds${conds:+,}$run:$s:${backend_of[$name]}"
  done
  [ "$stop" = 1 ] && break
  [ -z "$conds" ] && { log "no run is ready for evaluation"; break; }
  log "attempt $attempt: $conds"
  started=$(date +%s); runlog=$EVAL_OUT/logs/runner_${LANE}.log
  before=$(grep -c '"event": "group"' "$runlog" 2>/dev/null); before=${before:-0}
  CUDA_VISIBLE_DEVICES=$first setsid nice -n 10 python scripts/evaluate_rt_matched.py --tag final_rollout \
    --key-prefix elastic --conditions "$conds" --expected-sha "$EVAL_EXPECTED_SHA" --output-root "$EVAL_OUT" \
    --weights-root "$EVAL_RUNS" --vae-path "$ELASTIC_VAE" --text-cache-dir "$EVAL_TEXT" --verify-text-hash \
    --gpus "$GPUS" --workers-per-gpu "$WPG" --no-videos >> "$runlog" 2>&1 &
  child=$!; wait "$child"; rc=$?
  while kill -0 "$child" 2>/dev/null; do sleep 5; done
  child=""
  after=$(grep -c '"event": "group"' "$runlog" 2>/dev/null); after=${after:-0}
  log "runner exited rc=$rc after $(( $(date +%s) - started ))s; $((after - before)) groups scored"
  [ "$stop" = 1 ] && break
  if [ "$((after - before))" -eq 0 ]; then
    newly=0; for c in $EVAL_CONDITIONS; do run=${c%%:*}; ready "$run" && [[ ",$conds," != *",$run:"* ]] && newly=1; done
    [ "$newly" = 1 ] || break
  fi
  sleep 30
done
[ "$stop" = 1 ] && log "stopped by request" || log "finished"
