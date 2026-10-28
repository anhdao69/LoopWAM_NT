#!/usr/bin/env bash
# Elastic campaign lane (runs inside an overlapping srun step of the interactive job; srun client on evuser2).
# Usage: lane.sh <lane-name> <gpu-pair>. Both lanes share one ordered queue; run_elastic.sh's per-run flock
# makes a lane skip a run the other lane owns (exit 75). Completed runs are skipped; finished runs get a
# manifest and their records uploaded next to the checkpoints.
set -uo pipefail
LANE=$1; GPUS=$2
WORK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work
RUNS=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic
QUEUE="E0 E2 E1 SP11 F0-PB"
source $WORK/env.sh
export CUDA_VISIBLE_DEVICES=$GPUS ELASTIC_HF_TOKEN_FILE=/groups/yshang/an221229/cache/huggingface/token
log() { echo "[$(date -Is)] lane $LANE: $*"; }
complete() { [ -f "$RUNS/$1/status.json" ] && python -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['status']=='complete' and s['update']==21700 else 1)" "$RUNS/$1/status.json"; }
# The node-local cache is per allocation; restage it if this job does not have it yet.
if [ ! -f "$ELASTIC_LATENTS/STAGED.json" ]; then
  exec 8>"$WORK/stage.lock"; flock 8
  [ -f "$ELASTIC_LATENTS/STAGED.json" ] || python $WORK/stage_latents.py "$ELASTIC_LUSTRE_LATENTS" "$ELASTIC_LATENTS" >> $WORK/logs/stage_latents_${SLURM_JOB_ID}.log 2>&1
  flock -u 8
fi
log "start on $(hostname) job ${SLURM_JOB_ID:-?} gpus $GPUS"
for pass in 1 2 3; do
  progressed=0
  for run in $QUEUE; do
    complete "$run" && continue
    init=$WORK/init/elastic_common_init_v1.pt
    [ "$run" = F0-PB ] && init=$WORK/init/rt_a2_legacy_init_v1.pt
    log "trying $run (pass $pass)"
    code=0
    ELASTIC_RUN=$run ELASTIC_OUTPUT=$RUNS/$run ELASTIC_INIT=$init ELASTIC_WORLD=2 ELASTIC_MICROBATCH=64 \
      ELASTIC_CKPT_MIN_BLOCKS=180 ELASTIC_WORKERS=2 ELASTIC_SAVE_EVERY=100 \
      bash $ELASTIC_ROOT/scripts/run_elastic.sh >> $WORK/logs/run_${run}.log 2>&1 || code=$?
    if [ "$code" = 75 ]; then log "$run is owned by the other lane"; continue; fi
    log "$run exited rc=$code"
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
log "queue exhausted"
