#!/usr/bin/env bash
#SBATCH --job-name=loopwam_iv
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work/logs/slurm_%j.out
# Runs one v0 init-variant training (train_variant.sh) to completion: one trainer per run directory (run lock),
# retries from the last checkpoint after a crash, deletes its node-local cache on exit. Nothing is uploaded.
# Interactive allocation (from the login node, so the srun client outlives any session):
#   srun --jobid=J --overlap --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:4 --mem=0 \
#        env VARIANT=core_random GPUS=0,1 bash job_variant.sh
# WAIT_FOR_RUN=<run dir> first waits until that run's training is complete, its trainer has exited and GPUS are free.
# Batch fallback: sbatch --export=ALL,VARIANT=core_random job_variant.sh
set -uo pipefail
VARIANT=${VARIANT:?set VARIANT}
WORK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work
RUN=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants/$VARIANT
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}
OWNER="${SLURM_JOB_ID:-local}.${SLURM_STEP_ID:-batch}"
log() { echo "[$(date -Is)] $*"; }
complete() { [ -f "$RUN/train/timing.json" ] && grep -q '"status": "complete"' "$RUN/train/timing.json"; }
last_update() { local u; u=$(tail -n 100 "$RUN/train/metrics.jsonl" 2>/dev/null | grep -o '"update": [0-9]*' | tail -n 1 | grep -o '[0-9]*$' || true); echo "${u:-0}"; }
mkdir -p "$RUN"
log "variant $VARIANT, job step $OWNER on $(hostname), GPUs $GPUS"

LOCK="$RUN/.run_lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  HOLDER=$(cat "$LOCK/job" 2>/dev/null || echo none)
  if [[ "$HOLDER" != "$OWNER" ]] && squeue -h -j "${HOLDER%%.*}" -t R,CG 2>/dev/null | grep -q .; then
    log "run locked by $HOLDER (job still running); exiting"; exit 0
  fi
  rm -rf "$LOCK"; mkdir "$LOCK" || { log "lock race lost"; exit 0; }
fi
echo "$OWNER" > "$LOCK/job"
TAG=loopwam_iv_${VARIANT}_${SLURM_JOB_ID:-local}
cleanup() {
  rm -rf "/tmp/$TAG" "/tmp/${TAG}_triton" "$RUN/latents_${SLURM_JOB_ID:-local}"
  [[ "$(cat "$LOCK/job" 2>/dev/null)" == "$OWNER" ]] && rm -rf "$LOCK"
}
trap cleanup EXIT

if [ -n "${WAIT_FOR_RUN:-}" ]; then
  log "waiting for $WAIT_FOR_RUN to complete and GPUs $GPUS to be free"
  until [ -f "$WAIT_FOR_RUN/train/timing.json" ] && grep -q '"status": "complete"' "$WAIT_FOR_RUN/train/timing.json"; do sleep 60; done
  log "$WAIT_FOR_RUN training complete"
  while ps -u "$USER" -o args= | awk -v d="--output-dir $WAIT_FOR_RUN/train" '$1 != "awk" && index($0, d)' | grep -q .; do sleep 15; done
  for i in $(seq 1 120); do
    USED=$(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    [ "$USED" -lt 2000 ] && break; sleep 15
  done
  [ "$USED" -lt 2000 ] || { log "GPUs $GPUS still busy ($USED MiB); not starting"; exit 3; }
  log "trainer exited and GPUs $GPUS free ($USED MiB); starting $VARIANT"
fi

STOP=0; TRAIN_PID=
trap 'log "USR1: walltime near, stopping trainer"; STOP=1; [ -n "$TRAIN_PID" ] && kill -TERM -- -"$TRAIN_PID" 2>/dev/null' USR1
for attempt in 1 2 3 4; do
  complete && break
  BEFORE=$(last_update)
  VARIANT=$VARIANT GPUS=$GPUS setsid bash "$WORK/train_variant.sh" &
  TRAIN_PID=$!
  wait "$TRAIN_PID"; RC=$?
  while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 5; done
  log "attempt $attempt: trainer exited rc=$RC, updates $BEFORE -> $(last_update)"
  [ "$STOP" = 1 ] && break
  [ "$RC" = 0 ] && break
  sleep 60   # e.g. intermittent CUDA init error 802, or a crash: resume from the last checkpoint
done
complete && log "training complete: $RUN/train" || log "training NOT complete: $RUN/train"
