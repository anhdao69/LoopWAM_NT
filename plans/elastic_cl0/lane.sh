#!/usr/bin/env bash
# One CL-0 elastic lane inside an overlapping srun step: runs the queued runs in order on the given GPUs.
# Usage: lane.sh <lane> <gpus e.g. 0,1,2,3> <world> <microbatch> "<queue e.g. CL0-E0 CL0-E1 CL0-E2>"
# Optional env: CL0_CKPT_MIN_BLOCKS (adaptive block checkpointing), ELASTIC_SPEED_ARGS (validated speed flags).
# run_elastic.sh refuses (exit 75) a run another lane owns; completed runs are skipped and then finalized
# (manifest + records uploaded next to the checkpoints). SIGTERM stops the current run gracefully.
set -uo pipefail
LANE=$1; GPUS=$2; WORLD=$3; MB=$4; QUEUE=$5
source "$(dirname "$0")/env.sh"
RUNS=$CL0_WORK/runs; LOGS=$CL0_WORK/logs; mkdir -p "$RUNS" "$LOGS"
export CUDA_VISIBLE_DEVICES=$GPUS ELASTIC_LANE=$LANE
log() { echo "[$(date -Is)] lane $LANE: $*"; }
complete() { [ -f "$RUNS/$1/status.json" ] && python -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['status']=='complete' and s['update']==21700 else 1)" "$RUNS/$1/status.json"; }
child=""; stop=0
trap 'stop=1; if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null; fi' TERM
[ -f "$ELASTIC_LATENTS/STAGED.json" ] || { log "latents are not staged at $ELASTIC_LATENTS"; exit 1; }
log "start on $(hostname) job ${SLURM_JOB_ID:-?} gpus $GPUS world $WORLD mb $MB speed '${ELASTIC_SPEED_ARGS:-}' queue '$QUEUE'"
for run in $QUEUE; do
  [ "$stop" = 1 ] && break
  complete "$run" && { log "$run already complete"; continue; }
  log "starting $run"
  ELASTIC_RUN=$run ELASTIC_OUTPUT=$RUNS/$run ELASTIC_WORLD=$WORLD ELASTIC_MICROBATCH=$MB \
    ELASTIC_CKPT_MIN_BLOCKS=${CL0_CKPT_MIN_BLOCKS:-} ELASTIC_WORKERS=${CL0_WORKERS:-3} ELASTIC_SAVE_EVERY=100 \
    bash "$ELASTIC_ROOT/scripts/run_elastic.sh" >> "$LOGS/run_${run}.log" 2>&1 &
  child=$!; code=0
  wait "$child" || code=$?
  if kill -0 "$child" 2>/dev/null; then wait "$child" || code=$?; fi
  child=""
  if [ "$code" = 75 ]; then log "$run is owned elsewhere"; continue; fi
  log "$run exited rc=$code"
  [ "$stop" = 1 ] && break
  if complete "$run"; then
    ( cd "$ELASTIC_ROOT" && python scripts/elastic_manifest.py --output-dir "$RUNS/$run" && \
      { [ -z "$ELASTIC_HF_TOKEN_FILE" ] || python scripts/upload_elastic.py --run-name "$run" --output-dir "$RUNS/$run" --token-file "$ELASTIC_HF_TOKEN_FILE" \
        --extra manifest.json metrics.jsonl status.json launches.jsonl schedule_sequence.json \
        step_00002000.pt.uploaded.json epoch_08.pt.uploaded.json epoch_09.pt.uploaded.json epoch_10.pt.uploaded.json; } ) \
      >> "$LOGS/finalize_${run}.log" 2>&1 && log "$run finalized" || log "$run finalize failed (see finalize_${run}.log)"
  else
    log "$run did not complete; stopping lane"; break
  fi
done
log "lane done (stop=$stop)"
