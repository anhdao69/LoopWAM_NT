#!/usr/bin/env bash
# RT50-ep5 E0 -> E1 -> E2, one RoboTwin-50 epoch (47,462 updates) each, on GPUs 0-3 of job 5758 (GR1 owns GPUs 4-7).
# Runs as a 2-CPU step (-c2 --gres=none), idempotent: every decision comes from files, so a relaunch continues.
#   per run: up to 6 attempts of train_run.sh (resume from latest.pt); an attempt whose train.log/metrics.jsonl stop
#   changing for 30 min is cancelled (scancel of its step) and retried; a run still incomplete after 6 attempts is
#   marked failed and the queue moves on. Each completed run is verified (verify_rt50_export.py on GPU 0:
#   epoch_01.pt vs latest.pt, step_00002000.pt vs step_00002000_recovery.pt) before the next one starts.
# Usage: queue_ep5.sh REVISION
set -u
REV=$1
D=$(cd "$(dirname "$0")" && pwd); source "$D/env_ep5.sh"
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
Q=${ERT_QUEUE:-"RT50-ep5-E0 RT50-ep5-E1 RT50-ep5-E2"}
B=$ERT_EP5; RUNS=$B/runs; V=$B/validation; L=$B/logs; mkdir -p "$RUNS" "$V" "$L"
JOB=5758; UPDATES=47462; MAX_ATTEMPTS=6; HANG=1800
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$B/queue.log"; }
status() {
  python3 -I -c 'import json,sys,time; json.dump(dict(phase=sys.argv[1], detail=sys.argv[2], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(sys.argv[3], "w"), indent=1)' \
    "$1" "$2" "$B/STATUS.json.tmp" && mv "$B/STATUS.json.tmp" "$B/STATUS.json"
  log "STATUS $1: $2"
}
exec 8>/tmp/ert_ep5_queue.lock
flock -n 8 || { echo "[$(date -u +%FT%TZ)] another queue_ep5 is running on $(hostname)" >> "$B/queue.log"; exit 0; }
( exec 8>&-; while true; do touch "$B/heartbeat"; sleep 60; done ) & BEAT=$!
trap 'kill $BEAT 2>/dev/null' EXIT
log "queue start: host $(hostname) job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?} pid $$ revision $REV queue '$Q'"
complete() {
  python3 -I - "$RUNS/$1" "$UPDATES" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
try: s = json.loads((out / "status.json").read_text())
except Exception: sys.exit(1)
ok = s.get("status") == "complete" and s.get("update") == int(sys.argv[2]) and s.get("epoch") == 1 and s.get("next_micro") == 0
sys.exit(0 if ok and all((out / f).is_file() for f in ("latest.pt", "epoch_01.pt", "step_00002000.pt", "step_00002000_recovery.pt")) else 1)
PY
}
verified() { python3 -I -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("passed") is True else 1)' "$V/verify_$1.json" 2>/dev/null; }
alive() { [ -d "$RUNS/$1" ] || return 1; ( flock -n 9 || exit 0; exit 1 ) 9>>"$RUNS/$1/run.lock"; }
last_write() { stat -c %Y "$RUNS/$1/metrics.jsonl" "$RUNS/$1/train.log" 2>/dev/null | sort -n | tail -1; }
attempts() { cat "$B/attempts_$1" 2>/dev/null || echo 0; }
watch() {  # RUN PID(or empty when monitoring an orphan): wait for the attempt to end; cancel it on a hang
  local run=$1 pid=$2 start; start=$(date +%s)
  while { [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; } || { [ -z "$pid" ] && alive "$run"; }; do
    sleep 60
    local now last; now=$(date +%s); last=$(last_write "$run"); last=${last:-$start}
    [ "$last" -lt "$start" ] && last=$start
    if [ $((now - last)) -gt "$HANG" ]; then
      local s; s=$(cat "$RUNS/$run/step_id" 2>/dev/null)
      log "$run: no training output for $((now - last)) s: cancelling step $JOB.${s:-?}"
      [ -n "$s" ] && scancel "$JOB.$s"
      sleep 120; start=$(date +%s)
    fi
  done
  [ -n "$pid" ] && wait "$pid"
}
verify() {
  local run=$1
  status verifying "$run: export verification on GPU 0"
  # -c120: a smaller step's cpuset need not contain the pinned CPUs (taskset EINVAL); GPU 0 and its CPUs are idle here.
  srun --jobid=$JOB --overlap -n1 -c120 --cpu-bind=none --gres=gpu:8 -J training_vln bash -c "
    source '$D/env_ep5.sh'; cd \"\$ELASTIC_ROOT\"; export CUDA_VISIBLE_DEVICES=0
    taskset -c 0-7,64-71 python -u scripts/verify_rt50_export.py --run-dir '$RUNS/$run' \
      --pairs epoch_01.pt:latest.pt,step_00002000.pt:step_00002000_recovery.pt --out '$V/verify_$run.json'" \
    > "$L/verify_$run.log" 2>&1
  local rc=$?
  if verified "$run"; then log "$run: exports VERIFIED"; else log "$run: export verification FAILED rc=$rc (see $L/verify_$run.log)"; fi
}
summary=""
for run in $Q; do
  if verified "$run"; then log "$run already complete and verified"; summary="$summary $run:verified"; continue; fi
  if alive "$run"; then
    status training "$run: monitoring an attempt started by an earlier queue process"
    watch "$run" ""
  fi
  while ! complete "$run"; do
    n=$(attempts "$run")
    [ "$n" -ge "$MAX_ATTEMPTS" ] && break
    n=$((n + 1)); echo "$n" > "$B/attempts_$run"
    status training "$run: attempt $n/$MAX_ATTEMPTS ($( [ -f "$RUNS/$run/latest.pt" ] && echo resume || echo fresh ))"
    srun --jobid=$JOB --overlap -n1 -c120 --cpu-bind=none --gres=gpu:8 -J training_vln \
      bash "$D/train_run.sh" "$run" "$REV" >> "$L/train_${run}.log" 2>&1 &
    watch "$run" $!; rc=$?
    log "$run: attempt $n ended rc=$rc; status $(tr -d '\n ' < "$RUNS/$run/status.json" 2>/dev/null | cut -c1-200)"
    complete "$run" || sleep 60
  done
  if complete "$run"; then
    log "$run: training COMPLETE"
    verify "$run"
    verified "$run" && summary="$summary $run:verified" || summary="$summary $run:complete-unverified"
  else
    log "$run: FAILED after $(attempts "$run") attempts; moving on"; summary="$summary $run:failed"
  fi
done
case "$summary" in *failed*|*unverified*) status done_with_problems "${summary# }" ;; *) status done "${summary# }" ;; esac
