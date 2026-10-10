#!/usr/bin/env bash
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --time=22:00:00
#SBATCH --signal=B:USR1@600
# ChronoLoop training job on 2 or 4 H100s (the --gres given at submission; default 4).
# Submit through scripts/chronoloop_submit.sh, which may queue a 2-GPU and a 4-GPU twin per run:
# the first to start takes the run lock and cancels its pending twin.
# Required env: CHRONO_RUN (registry key), CHRONO_CODE (frozen code snapshot), CHRONO_OUT (run directory).
# Behaviour: resume from <out>/latest.pt if present; exit cleanly before walltime; if unfinished,
# resubmit itself (one pending continuation at most); when complete, upload epochs 8/9/10 to HF
# with retries and verification. A live heartbeat from another job blocks duplicate training.
set -uo pipefail
: "${CHRONO_RUN:?}" "${CHRONO_CODE:?}" "${CHRONO_OUT:?}"
cd "$CHRONO_CODE"
source scripts/chrono_env.sh
export TRITON_CACHE_DIR="/tmp/chrono_${SLURM_JOB_ID}_triton"
export NCCL_DEBUG=WARN
mkdir -p "$CHRONO_OUT"
JOB_NAME="$(python scripts/chronoloop_experiments.py "$CHRONO_RUN" --field job)"
HF_FOLDER="$(python scripts/chronoloop_experiments.py "$CHRONO_RUN" --field hf)"
LOG="$CHRONO_OUT/job_${SLURM_JOB_ID}.log"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] job $SLURM_JOB_ID on $(hostname) run=$CHRONO_RUN code=$CHRONO_CODE"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

NGPU=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
if [[ $NGPU -ne 2 && $NGPU -ne 4 ]]; then echo "Unsupported GPU count $NGPU"; exit 1; fi

walltime_hours() {  # remaining allocation hours from Slurm
  local end; end=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
  python - "$end" <<'EOF'
import sys, datetime
end = datetime.datetime.fromisoformat(sys.argv[1])
print(max(0.0, (end - datetime.datetime.now()).total_seconds() / 3600))
EOF
}

# Duplicate guard: another live trainer for this run directory?
if [[ -f "$CHRONO_OUT/timing.json" && ! -f "$CHRONO_OUT/COMPLETE" ]]; then
  read -r OTHER AGE < <(python - "$CHRONO_OUT/timing.json" <<'EOF'
import json, sys, time
t = json.load(open(sys.argv[1]))
print(t.get('slurm_job_id') or 'none', int(time.time() - t.get('heartbeat_unix', 0)))
EOF
)
  if [[ "$OTHER" != "$SLURM_JOB_ID" && "$OTHER" != "none" && "$AGE" -lt 1200 ]] && squeue -h -j "$OTHER" -t R 2>/dev/null | grep -q .; then
    echo "Another job ($OTHER) is training this run (heartbeat ${AGE}s ago); exiting without work."; exit 0
  fi
fi

# Run lock (mkdir is atomic): a lock held by a job that is no longer running is stale.
LOCK="$CHRONO_OUT/.run_lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  OWNER=$(cat "$LOCK/job" 2>/dev/null || echo none)
  if [[ "$OWNER" != "$SLURM_JOB_ID" ]] && squeue -h -j "$OWNER" -t R,CG 2>/dev/null | grep -q .; then
    echo "Run locked by running job $OWNER; exiting without work."; exit 0
  fi
  echo "Taking over stale lock from $OWNER"; rm -rf "$LOCK"; mkdir "$LOCK" || { echo "lock race lost"; exit 0; }
fi
echo "$SLURM_JOB_ID" > "$LOCK/job"
# Cancel the pending twin (same run, other GPU count) now that this job owns the run.
for TWIN in $(squeue -h -u "$USER" -n "$JOB_NAME" -t PD -o %i); do
  [[ "$TWIN" != "$SLURM_JOB_ID" ]] && scancel "$TWIN" && echo "cancelled pending twin $TWIN"
done
# Node-local latent copy made by the trainer (--stage-latents auto) and the lock go when this job exits.
trap 'rm -rf "/dev/shm/chronoloop_latents_${SLURM_JOB_ID}"; [[ "$(cat "$LOCK/job" 2>/dev/null)" == "$SLURM_JOB_ID" ]] && rm -rf "$LOCK"' EXIT
STATUS=0
if [[ ! -f "$CHRONO_OUT/COMPLETE" ]]; then
  LIMIT=$(walltime_hours)
  echo "[$(date -Is)] training on $NGPU GPUs; remaining allocation ${LIMIT} h"
  torchrun --standalone --nproc_per_node=$NGPU scripts/train_chronoloop.py \
    $(python scripts/chronoloop_experiments.py "$CHRONO_RUN") \
    --stream-micro $((32 / NGPU)) --output-dir "$CHRONO_OUT" --time-limit-hours "$LIMIT" &
  TRAIN_PID=$!
  trap 'echo "USR1: walltime near"; ' USR1
  wait $TRAIN_PID; STATUS=$?
  echo "[$(date -Is)] trainer exit $STATUS"
  if [[ $STATUS -eq 0 ]]; then rm -f "$CHRONO_OUT/failures"; else echo "$SLURM_JOB_ID" >> "$CHRONO_OUT/failures"; fi
fi

if [[ -f "$CHRONO_OUT/COMPLETE" ]]; then
  for attempt in 1 2 3; do
    python scripts/chronoloop_hf_upload.py --run-dir "$CHRONO_OUT" --folder "$HF_FOLDER" --require-all && break
    echo "upload attempt $attempt failed"; sleep 300
  done
  if ! python - "$CHRONO_OUT/hf_upload.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1])); sys.exit(0 if r.get('verified') and len(r.get('epochs_verified', [])) == 3 else 1)
EOF
  then
    echo "Upload not verified; queueing a CPU-only retry job"
    sbatch --partition=normal --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=04:00:00 \
      --job-name="${JOB_NAME}-upload" --output="$CHRONO_OUT/upload_%j.log" \
      --wrap="cd $CHRONO_CODE && source scripts/chrono_env.sh && python scripts/chronoloop_hf_upload.py --run-dir $CHRONO_OUT --folder $HF_FOLDER --require-all"
  fi
  exit 0
fi

# Unfinished: upload any finished epoch checkpoints now (idempotent), then continue in a new job.
ls "$CHRONO_OUT"/checkpoints/epoch_*.pt >/dev/null 2>&1 && \
  python scripts/chronoloop_hf_upload.py --run-dir "$CHRONO_OUT" --folder "$HF_FOLDER" || true
FAILS=$(cat "$CHRONO_OUT/failures" 2>/dev/null | wc -l)
if [[ $FAILS -ge 3 ]]; then
  echo "3 consecutive training failures; not resubmitting (see job logs)"; exit $STATUS
fi
if [[ -f "$CHRONO_OUT/latest.pt" || $STATUS -eq 0 || $FAILS -lt 3 ]]; then
  PENDING=$(squeue -h -u "$USER" -n "$JOB_NAME" -t PD -o %i | grep -v "^$SLURM_JOB_ID$" | head -1)
  if [[ -z "$PENDING" ]]; then
    NEW=$(sbatch --parsable --job-name="$JOB_NAME" --output="$CHRONO_OUT/slurm_%j.out" \
          --gres=gpu:nvidia_h100_80gb_hbm3:$NGPU --cpus-per-task=$((4 * NGPU)) --mem=$((64 * NGPU))G \
          --export=ALL,CHRONO_RUN="$CHRONO_RUN",CHRONO_CODE="$CHRONO_CODE",CHRONO_OUT="$CHRONO_OUT" \
          "$CHRONO_CODE/scripts/chronoloop_job.sbatch")
    echo "[$(date -Is)] continuation submitted: $NEW"
    echo "$NEW" >> "$CHRONO_OUT/continuations.txt"
  else
    echo "continuation already pending: $PENDING"
  fi
fi
exit $STATUS
