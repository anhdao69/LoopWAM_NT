#!/usr/bin/env bash
# Elastic evaluation lane (inside an overlapping srun step; srun client on evuser2, launched by dispatcher.py).
# Usage: eval_lane.sh <lane-name> <physical GPU list, e.g. 2,3> <workers per GPU>
# Evaluates the decisive conditions (EVAL_ORDER x EVAL_SCHEDULES, plan section 12.2) of every run whose training is
# complete. Per run, a CUDA-graph gate on its epoch_10.pt first (actions must be bit-exact against eager on real
# simulator observations, otherwise that run is evaluated with the eager backend). Then the resumable matched runner
# (scripts/evaluate_rt_matched.py): task/seed groups are claimed atomically, so lanes on any job share the work.
# The runner's workers receive physical GPU indices (the step sees all four GPUs); its main process is pinned to the
# lane's first GPU so that nothing touches a training GPU. Everything runs at nice 10 so training keeps its CPUs.
# Exits when this lane finds no pending group of a ready run. SIGTERM stops the runner (claims expire after 10 min).
# With EVAL_MPS=1 the lane starts a private CUDA MPS daemon (node-local pipe directory given only to this lane's
# processes, so training never connects): workers then share each GPU concurrently instead of time-slicing, which
# removes the GPU bottleneck at several workers per GPU. Under MPS the stack reproduced 20/20 stored RT-A2 episodes
# exactly (parity_rt.sh via mps_test.sh, 2026-10-29).
set -uo pipefail
LANE=$1; GPUS=$2; WPG=$3
source /groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work/eval_env.sh
cd "$ELASTIC_ROOT"
mkdir -p "$EVAL_OUT/gates" "$EVAL_OUT/logs"
log() { echo "[$(date -Is)] eval lane $LANE: $*"; }
first=${GPUS%%,*}
child=""; stop=0
trap 'stop=1; [ -n "$child" ] && kill -TERM -- -"$child" 2>/dev/null' TERM
ready() {
  [ -f "$EVAL_RUNS/$1/epoch_10.pt" ] && python -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['status']=='complete' and s['update']==21700 else 1)" "$EVAL_RUNS/$1/status.json" 2>/dev/null
}
gate_backend() {  # prints cuda-graph or eager for run $1; returns 1 if the gate could not be run
  local run=$1 out=$EVAL_OUT/gates/$1.json lock=$EVAL_OUT/gates/$1.lock
  for i in $(seq 40); do
    if [ -f "$out" ]; then
      python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['all_bit_exact'] is True else 1)" "$out" \
        && echo cuda-graph || echo eager
      return 0
    fi
    if mkdir "$lock" 2>/dev/null; then
      log "gating $run on GPU $first" >&2
      CUDA_VISIBLE_DEVICES=$first MUJOCO_EGL_DEVICE_ID=$first nice -n 10 python scripts/preflight_rt_graph.py \
        --run "$run" --schedules "$(echo $EVAL_SCHEDULES | tr ' ' ,)" --weights-root "$EVAL_RUNS" \
        --vae-path "$ELASTIC_VAE" --text-cache-dir "$EVAL_TEXT" --queries 8 --output "$out.tmp.$$" \
        >> "$EVAL_OUT/logs/gate_${run}.log" 2>&1 && mv "$out.tmp.$$" "$out"
      rmdir "$lock"
      [ -f "$out" ] || { log "gate for $run failed to run (see logs/gate_${run}.log)" >&2; return 1; }
      continue
    fi
    # Another lane is gating this run; a lock older than 30 min is stale.
    [ $(( $(date +%s) - $(stat -c %Y "$lock" 2>/dev/null || date +%s) )) -gt 1800 ] && rmdir "$lock" 2>/dev/null
    sleep 30
  done
  return 1
}
log "start on $(hostname) job ${SLURM_JOB_ID:-?} gpus $GPUS workers/gpu $WPG"
mps_dir=""
if [ "${EVAL_MPS:-0}" = 1 ]; then
  mps_dir=/dev/shm/elastic_mps_${SLURM_JOB_ID:-x}_$LANE
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
  conds=""; shas=""
  for run in $EVAL_ORDER; do
    ready "$run" || continue
    backend=$(gate_backend "$run") || continue
    [ "$backend" = eager ] && log "$run: CUDA graph not bit-exact; using the eager backend"
    for s in $EVAL_SCHEDULES; do conds="$conds${conds:+,}$run:$s:$backend"; done
    sha=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['sha256'])" "$EVAL_RUNS/$run/epoch_10.pt.uploaded.json" 2>/dev/null) \
      && shas="$shas${shas:+,}$run=$sha"
  done
  [ "$stop" = 1 ] && break
  [ -z "$conds" ] && { log "no run is ready for evaluation"; break; }
  log "attempt $attempt: $conds"
  started=$(date +%s); runlog=$EVAL_OUT/logs/runner_${LANE}.log
  before=$(grep -c '"event": "group"' "$runlog" 2>/dev/null); before=${before:-0}
  CUDA_VISIBLE_DEVICES=$first setsid nice -n 10 python scripts/evaluate_rt_matched.py --tag final_rollout \
    --key-prefix elastic --interleave --conditions "$conds" --expected-sha "$shas" --output-root "$EVAL_OUT" \
    --weights-root "$EVAL_RUNS" --vae-path "$ELASTIC_VAE" --text-cache-dir "$EVAL_TEXT" --verify-text-hash \
    --gpus "$GPUS" --workers-per-gpu "$WPG" --no-videos >> "$runlog" 2>&1 &
  child=$!; wait "$child"; rc=$?
  while kill -0 "$child" 2>/dev/null; do sleep 5; done
  child=""
  after=$(grep -c '"event": "group"' "$runlog" 2>/dev/null); after=${after:-0}
  log "runner exited rc=$rc after $(( $(date +%s) - started ))s; $((after - before)) groups scored"
  [ "$stop" = 1 ] && break
  # Nothing left for this lane (all groups done or claimed elsewhere) unless another run became ready meanwhile.
  if [ "$((after - before))" -eq 0 ]; then
    newly=0; for run in $EVAL_ORDER; do ready "$run" && [[ ",$conds," != *",$run:"* ]] && newly=1; done
    [ "$newly" = 1 ] || break
  fi
  sleep 30
done
[ "$stop" = 1 ] && log "stopped by request" || log "finished"
