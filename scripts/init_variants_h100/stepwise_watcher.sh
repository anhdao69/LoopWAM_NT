#!/usr/bin/env bash
# Runs on login node evuser2 (nohup setsid; log logs/stepwise_watcher.log). User request 2026-10-28: train the v0 init
# variant `stepwise` on all 4 GPUs of the user's NEW interactive job (whichever of NEW_JOBS starts running first); if that
# job ends before training is complete, finish it in the remaining time of interactive job 990547 (once its two variant
# runs are done and its GPUs are free). The srun client of each step lives here, so the step outlives any session.
set -uo pipefail
WORK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work
ROOT=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants
RUN=$ROOT/stepwise
NEW_JOBS="1080498 1080499 1080516"
FALLBACK=990547
log() { echo "[$(date -Is)] $*"; }
state() { squeue -h -j "$1" -o %T 2>/dev/null; }
complete() { [ -f "$1/train/timing.json" ] && grep -q '"status": "complete"' "$1/train/timing.json"; }
step() {  # JOB: run step_variant4.sh in an overlapping step that sees all 4 GPUs; returns when the step ends
  log "launching stepwise on job $1 ($(squeue -h -j "$1" -o %N))"
  srun --jobid="$1" --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:4 --mem=0 \
       env VARIANT=stepwise bash "$WORK/step_variant4.sh" >> "$WORK/logs/stepwise_step_$1.log" 2>&1
  log "step on job $1 exited rc=$?"
}
log "watching for the first of $NEW_JOBS to run; fallback $FALLBACK; pid $$ on $(hostname)"
CHOSEN=
while [ -z "$CHOSEN" ]; do
  complete "$RUN" && { log "stepwise already complete"; exit 0; }
  ALIVE=
  for j in $NEW_JOBS; do
    S=$(state "$j")
    [ "$S" = RUNNING ] && { CHOSEN=$j; break; }
    [ -n "$S" ] && ALIVE=1
  done
  [ -n "$CHOSEN" ] && break
  [ -n "$ALIVE" ] || { log "none of $NEW_JOBS is pending or running any more; not starting stepwise"; exit 0; }
  sleep 30
done
log "job $CHOSEN is running on $(squeue -h -j "$CHOSEN" -o %N)"
for attempt in 1 2 3 4; do
  [ "$(state "$CHOSEN")" = RUNNING ] || break
  step "$CHOSEN"
  complete "$RUN" && { log "stepwise training complete on job $CHOSEN"; exit 0; }
  sleep 120   # step refused to start (GPUs busy, lock still held) or crashed; retry while the job runs
done
log "job $CHOSEN is $(state "$CHOSEN" || true); stepwise incomplete (update $(tail -n 50 "$RUN/train/metrics.jsonl" 2>/dev/null | grep -o '"update": [0-9]*' | tail -n 1))"
for attempt in 1 2 3 4; do
  [ "$(state "$FALLBACK")" = RUNNING ] || { log "fallback job $FALLBACK is not running; stepwise stays incomplete"; exit 0; }
  until complete "$ROOT/core_random" && complete "$ROOT/ffn_magnitude"; do
    [ "$(state "$FALLBACK")" = RUNNING ] || { log "fallback job $FALLBACK ended; stepwise stays incomplete"; exit 0; }
    sleep 60
  done
  step "$FALLBACK"
  complete "$RUN" && { log "stepwise training complete on job $FALLBACK"; exit 0; }
  sleep 120
done
log "stepwise incomplete after fallback attempts"
