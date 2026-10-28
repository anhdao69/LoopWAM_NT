#!/usr/bin/env bash
# Elastic campaign lane (runs inside an overlapping srun step of an interactive job; srun client on evuser2).
# Usage: lane.sh <lane-name> <gpu-pair>. All lanes share one ordered queue; run_elastic.sh refuses (exit 75) a run
# that another lane owns (node-local flock + cross-node mkdir claim). Completed runs are skipped; finished runs get
# a manifest and their records uploaded next to the checkpoints. SIGTERM stops the current run gracefully
# (recovery state saved) and ends the lane; no new run starts in the last hour of the job.
set -uo pipefail
LANE=$1; GPUS=$2
WORK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work
RUNS=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic
QUEUE="E0 E2 E1 SP11 F0-PB"
source $WORK/env.sh
export CUDA_VISIBLE_DEVICES=$GPUS ELASTIC_HF_TOKEN_FILE=/groups/yshang/an221229/cache/huggingface/token ELASTIC_LANE=$LANE
log() { echo "[$(date -Is)] lane $LANE: $*"; }
complete() { [ -f "$RUNS/$1/status.json" ] && python -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['status']=='complete' and s['update']==21700 else 1)" "$RUNS/$1/status.json"; }
seconds_left() { echo $(( $(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s) - $(date +%s) )); }
child=""; stop=0
trap 'stop=1; if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null; fi' TERM
# The node-local cache is per allocation; stage it if this job does not have it yet.
if [ ! -f "$ELASTIC_LATENTS/STAGED.json" ]; then
  exec 8>"$WORK/stage.lock"; flock 8
  [ -f "$ELASTIC_LATENTS/STAGED.json" ] || python $WORK/stage_latents.py "$ELASTIC_LUSTRE_LATENTS" "$ELASTIC_LATENTS" >> $WORK/logs/stage_latents_${SLURM_JOB_ID}.log 2>&1
  flock -u 8; exec 8>&-
fi
[ -f "$ELASTIC_LATENTS/STAGED.json" ] || { log "latent staging failed"; exit 1; }
log "start on $(hostname) job ${SLURM_JOB_ID:-?} gpus $GPUS"
for pass in 1 2 3; do
  progressed=0
  for run in $QUEUE; do
    [ "$stop" = 1 ] && break 2
    complete "$run" && continue
    left=$(seconds_left)
    if [ "$left" -lt 3600 ]; then log "job ends in ${left}s; not starting $run"; break 2; fi
    init=$WORK/init/elastic_common_init_v1.pt
    [ "$run" = F0-PB ] && init=$WORK/init/rt_a2_legacy_init_v1.pt
    log "trying $run (pass $pass)"
    ELASTIC_RUN=$run ELASTIC_OUTPUT=$RUNS/$run ELASTIC_INIT=$init ELASTIC_WORLD=2 ELASTIC_MICROBATCH=64 \
      ELASTIC_CKPT_MIN_BLOCKS=180 ELASTIC_WORKERS=2 ELASTIC_SAVE_EVERY=100 \
      bash $ELASTIC_ROOT/scripts/run_elastic.sh >> $WORK/logs/run_${run}.log 2>&1 &
    child=$!; code=0
    wait "$child" || code=$?
    if kill -0 "$child" 2>/dev/null; then wait "$child" || code=$?; fi
    child=""
    if [ "$code" = 75 ]; then log "$run is owned elsewhere"; continue; fi
    log "$run exited rc=$code"
    [ "$stop" = 1 ] && break 2
    if complete "$run"; then
      progressed=1
      ( cd $ELASTIC_ROOT && python scripts/elastic_manifest.py --output-dir $RUNS/$run && \
        python scripts/upload_elastic.py --run-name $run --output-dir $RUNS/$run --token-file $ELASTIC_HF_TOKEN_FILE \
          --extra manifest.json metrics.jsonl status.json launches.jsonl schedule_sequence.json \
          step_00002000.pt.uploaded.json epoch_08.pt.uploaded.json epoch_09.pt.uploaded.json epoch_10.pt.uploaded.json ) \
        >> $WORK/logs/finalize_${run}.log 2>&1 && log "$run finalized" || log "$run finalize failed (see finalize_${run}.log)"
    fi
  done
  [ "$progressed" = 1 ] || break
done
[ "$stop" = 1 ] && log "stopped by request" || log "queue exhausted"
