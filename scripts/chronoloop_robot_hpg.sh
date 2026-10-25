#!/usr/bin/env bash
#SBATCH --partition=hpg-b200
#SBATCH --account=yu954774.ucf
#SBATCH --qos=yu954774.ucf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@600
# ChronoLoop robot training on UF HiPerGator B200s (group QOS: 2 GPUs, 16 CPUs, 125 GB in total).
#   interactive:  bash scripts/chronoloop_robot_hpg.sh DATASET RUN [extra trainer args]
#   batch:        sbatch --job-name=chrono-DATASET-RUN scripts/chronoloop_robot_hpg.sh DATASET RUN [extra trainer args]
# CHRONO_EPOCHS (default 10) sets the schedule length; continuations inherit it through --export=ALL.
# Runs from the directory holding this script (use a frozen snapshot), resumes from <out>/latest.pt, stages the latent
# cache on node-local /tmp (/dev/shm would count against the 120 GB job memory), and stops cleanly 15 min before the
# allocation ends. A batch job that ends unfinished resubmits itself once (same arguments).
set -uo pipefail
DATASET="$1"; RUN="$2"; shift 2
CODE="${CHRONO_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$CODE"; source scripts/chrono_env_hpg.sh
EPOCHS=${CHRONO_EPOCHS:-10}   # length of the cosine schedule; every epoch end is saved
COMMON="--epochs $EPOCHS --text-threads 4 --save-epochs $(seq -s, 1 "$EPOCHS") --checkpoint-every 500"
case "$RUN" in
  CL-0) FLAGS="--memory-tokens 0 --action-loops 4 $COMMON"; DIR="CL-0_no-memory_a4";;
  CL-A) FLAGS="--memory-tokens 16 --mem-source learned --mem-write loop --action-loops 4 $COMMON"; DIR="CL-A_mem16-learned-loopwrite_a4";;
  *) echo "Unknown run $RUN (CL-0 | CL-A)" >&2; exit 1;;
esac
OUT=$CHRONO_ROBOT_SHARED/runs/$DATASET/$DIR
mkdir -p "$OUT"
LOG="$OUT/job_${SLURM_JOB_ID}.log"
exec >>"$LOG" 2>&1
NGPU=$(nvidia-smi -L | wc -l)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TRITON_CACHE_DIR="/tmp/chrono_${SLURM_JOB_ID}_${RUN}_triton"
STAGED="/tmp/chronoloop_${DATASET}_latents_${SLURM_JOB_ID}"
echo "[$(date -Is)] job $SLURM_JOB_ID on $(hostname): $RUN $DATASET, $NGPU GPUs, code=$CODE ($(cat REVISION 2>/dev/null || git rev-parse HEAD)) extra=$*"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

# Run lock (mkdir is atomic); a lock held by a job that is no longer running is stale.
LOCK="$OUT/.run_lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  OWNER=$(cat "$LOCK/job" 2>/dev/null || echo none)
  if [[ "$OWNER" != "$SLURM_JOB_ID" ]] && squeue -h -j "$OWNER" -t R,CG 2>/dev/null | grep -q .; then
    echo "Run locked by running job $OWNER; exiting without work."; exit 0
  fi
  echo "Taking over stale lock from $OWNER"; rm -rf "$LOCK"; mkdir "$LOCK" || { echo "lock race lost"; exit 0; }
fi
echo "$SLURM_JOB_ID" > "$LOCK/job"
trap 'rm -rf "$STAGED"; [[ "$(cat "$LOCK/job" 2>/dev/null)" == "$SLURM_JOB_ID" ]] && rm -rf "$LOCK"' EXIT

STATUS=0
if [[ ! -f "$OUT/COMPLETE" ]]; then
  DEADLINE=${SLURM_JOB_END_TIME:-$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)}
  echo "[$(date -Is)] training; allocation ends $(date -d "@$DEADLINE" -Is)"
  torchrun --standalone --nproc_per_node="$NGPU" scripts/train_chronoloop_robot.py \
    --dataset "$DATASET" --run-name "$RUN" $FLAGS --shared-dir "$CHRONO_ROBOT_SHARED/$DATASET" \
    --stream-micro $((32 / NGPU)) --stage-latents "$STAGED" --output-dir "$OUT" --deadline-unix "$DEADLINE" "$@" &
  TRAIN_PID=$!
  GOT_USR1=0
  trap 'echo "USR1: walltime near"; GOT_USR1=1' USR1
  # A trapped signal interrupts wait (status > 128) while the trainer runs on: wait again, or exiting here
  # would kill the trainer mid-checkpoint.
  while :; do
    wait $TRAIN_PID; STATUS=$?
    if [[ $GOT_USR1 -eq 1 && $STATUS -gt 128 ]]; then GOT_USR1=0; continue; fi
    break
  done
  echo "[$(date -Is)] trainer exit $STATUS"
fi
if [[ -f "$OUT/COMPLETE" ]]; then echo "[$(date -Is)] run complete"; exit 0; fi

# Batch jobs only: continue in a new job (at most one pending), unless the trainer failed twice in a row.
if [[ "${SLURM_JOB_NAME:-}" == chrono-* ]]; then
  if [[ $STATUS -ne 0 ]]; then echo "$SLURM_JOB_ID" >> "$OUT/failures"; else rm -f "$OUT/failures"; fi
  if [[ $(cat "$OUT/failures" 2>/dev/null | wc -l) -ge 2 ]]; then echo "2 consecutive failures; not resubmitting"; exit $STATUS; fi
  PENDING=$(squeue -h -u "$USER" -n "$SLURM_JOB_NAME" -t PD -o %i | head -1)
  if [[ -z "$PENDING" ]]; then
    NEW=$(sbatch --parsable --job-name="$SLURM_JOB_NAME" --output="$OUT/slurm_%j.out" --export=ALL \
          "$CODE/scripts/chronoloop_robot_hpg.sh" "$DATASET" "$RUN" "$@")
    echo "[$(date -Is)] continuation submitted: $NEW"; echo "$NEW" >> "$OUT/continuations.txt"
  else
    echo "continuation already pending: $PENDING"
  fi
fi
exit $STATUS
