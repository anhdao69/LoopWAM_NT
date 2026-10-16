#!/usr/bin/env bash
#SBATCH --partition=highgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@600
# ChronoLoop robot training job on 2 or 4 H100s (the --gres given at submission).
# Submit through scripts/chronoloop_robot_submit.sh, which may queue a 2-GPU and a 4-GPU twin per run:
# the first to start takes the run lock and cancels its pending twin.
# Required env: CHRONO_RUN (CL-0 | CL-A), CHRONO_DATASET, CHRONO_FLAGS (run flags), CHRONO_CODE (frozen code
# snapshot), CHRONO_OUT (run directory), CHRONO_SHARED_DS (dataset shared dir with schedule + latents).
# Behaviour: finish any missing latent-cache windows under a shared lock; resume from <out>/latest.pt;
# exit cleanly before walltime; if unfinished, resubmit itself (at most one pending continuation).
# A live heartbeat from another job blocks duplicate training.
set -uo pipefail
: "${CHRONO_RUN:?}" "${CHRONO_DATASET:?}" "${CHRONO_FLAGS:?}" "${CHRONO_CODE:?}" "${CHRONO_OUT:?}" "${CHRONO_SHARED_DS:?}"
cd "$CHRONO_CODE"
source scripts/chrono_env.sh
export TRITON_CACHE_DIR="/tmp/chrono_${SLURM_JOB_ID}_triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
mkdir -p "$CHRONO_OUT"
JOB_NAME="${SLURM_JOB_NAME}"
LOG="$CHRONO_OUT/job_${SLURM_JOB_ID}.log"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] job $SLURM_JOB_ID on $(hostname) run=$CHRONO_RUN dataset=$CHRONO_DATASET code=$CHRONO_CODE"
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
STAGED="/dev/shm/chronoloop_${CHRONO_DATASET}_latents_${SLURM_JOB_ID}"
trap 'rm -rf "$STAGED"; [[ "$(cat "$LOCK/job" 2>/dev/null)" == "$SLURM_JOB_ID" ]] && rm -rf "$LOCK"' EXIT

# Finish the shared latent cache if it is incomplete (one builder at a time; skips valid windows).
VALID=$(python - "$CHRONO_SHARED_DS/latents/valid.uint8" <<'EOF'
import sys, numpy as np, os
p = sys.argv[1]
print('missing' if not os.path.exists(p) else ('complete' if (np.memmap(p, mode='r', dtype=np.uint8) == 1).all() else 'partial'))
EOF
)
if [[ "$VALID" != complete ]]; then
  echo "[$(date -Is)] latent cache $VALID; building missing windows (waits for any other builder)"
  flock "$CHRONO_SHARED_DS/.build_lock" torchrun --standalone --nproc_per_node=$NGPU scripts/chronoloop_robot_precompute.py \
    --dataset "$CHRONO_DATASET" --shared-dir "$CHRONO_SHARED_DS" --workers 3 --decode-threads 1 --batch 16 \
    || { echo "latent build failed"; echo "$SLURM_JOB_ID" >> "$CHRONO_OUT/failures"; }
fi

STATUS=0
if [[ ! -f "$CHRONO_OUT/COMPLETE" ]]; then
  LIMIT=$(walltime_hours)
  echo "[$(date -Is)] training on $NGPU GPUs; remaining allocation ${LIMIT} h"
  torchrun --standalone --nproc_per_node=$NGPU scripts/train_chronoloop_robot.py \
    --dataset "$CHRONO_DATASET" --run-name "$CHRONO_RUN" $CHRONO_FLAGS --shared-dir "$CHRONO_SHARED_DS" \
    --stream-micro $((32 / NGPU)) --stage-latents "$STAGED" --output-dir "$CHRONO_OUT" --time-limit-hours "$LIMIT" &
  TRAIN_PID=$!
  trap 'echo "USR1: walltime near"; ' USR1
  wait $TRAIN_PID; STATUS=$?
  echo "[$(date -Is)] trainer exit $STATUS"
  if [[ $STATUS -eq 0 ]]; then rm -f "$CHRONO_OUT/failures"; else echo "$SLURM_JOB_ID" >> "$CHRONO_OUT/failures"; fi
fi

if [[ -f "$CHRONO_OUT/COMPLETE" ]]; then echo "[$(date -Is)] run complete"; exit 0; fi

# Unfinished: continue in a new job with the same GPU count.
FAILS=$(cat "$CHRONO_OUT/failures" 2>/dev/null | wc -l)
if [[ $FAILS -ge 3 ]]; then
  echo "3 consecutive failures; not resubmitting (see job logs)"; exit $STATUS
fi
PENDING=$(squeue -h -u "$USER" -n "$JOB_NAME" -t PD -o %i | grep -v "^$SLURM_JOB_ID$" | head -1)
if [[ -z "$PENDING" ]]; then
  NEW=$(sbatch --parsable --job-name="$JOB_NAME" --output="$CHRONO_OUT/slurm_%j.out" --time="${CHRONO_TIME:-24:00:00}" \
        --gres=gpu:nvidia_h100_80gb_hbm3:$NGPU --cpus-per-task=$((4 * NGPU)) --mem="${CHRONO_MEM:-256G}" \
        --export=ALL "$CHRONO_CODE/scripts/chronoloop_robot_job.sbatch")
  echo "[$(date -Is)] continuation submitted: $NEW"
  echo "$NEW" >> "$CHRONO_OUT/continuations.txt"
else
  echo "continuation already pending: $PENDING"
fi
exit $STATUS
