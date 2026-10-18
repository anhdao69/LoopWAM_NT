#!/usr/bin/env bash
# Train one ChronoLoop robot run (CL-0 | CL-A) inside the current interactive allocation on a GPU subset,
# from a frozen code snapshot. Resumes from <out>/latest.pt, stops cleanly 15 min before the allocation
# ends, and takes the same run lock as chronoloop_robot_job.sbatch, so a batch job cannot train it too.
#   bash scripts/chronoloop_robot_interactive.sh CODE_DIR DATASET RUN GPU_LIST [extra trainer args]
# e.g. ... robotwin5hard CL-A 2,3 --max-updates 35111   (stop at the end of epoch 5)
# Runs of one allocation share one node-local latent copy, removed when the last of them exits.
set -uo pipefail
CODE="$1"; DATASET="$2"; RUN="$3"; GPUS="$4"; shift 4
cd "$CODE"; source scripts/chrono_env.sh
ROBOT=/groups/yshang/an221229/checkpoints/ChronoLoop/robot
COMMON="--epochs 10 --text-threads 4 --save-epochs 5,8,9,10 --checkpoint-every 500"   # as chronoloop_robot_submit.sh
case "$RUN" in
  CL-0) FLAGS="--memory-tokens 0 --action-loops 4 $COMMON"; DIR="CL-0_no-memory_a4";;
  CL-A) FLAGS="--memory-tokens 16 --mem-source learned --mem-write loop --action-loops 4 $COMMON"; DIR="CL-A_mem16-learned-loopwrite_a4";;
  *) echo "Unknown run $RUN (CL-0 | CL-A)" >&2; exit 1;;
esac
OUT=$ROBOT/runs/$DATASET/$DIR
LOG="$OUT/interactive_${SLURM_JOB_ID}.log"
mkdir -p "$OUT"
exec >>"$LOG" 2>&1
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES="$GPUS"
export TRITON_CACHE_DIR="/tmp/chrono_${SLURM_JOB_ID}_${RUN}_triton"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
STAGED="/dev/shm/chronoloop_${DATASET}_latents_${SLURM_JOB_ID}"

LOCK="$OUT/.run_lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  OWNER=$(cat "$LOCK/job" 2>/dev/null || echo none)
  if [[ "$OWNER" != "$SLURM_JOB_ID" ]] && squeue -h -j "$OWNER" -t R,CG 2>/dev/null | grep -q .; then
    echo "Run locked by running job $OWNER; exiting without work."; exit 0
  fi
  echo "Taking over stale lock from $OWNER"; rm -rf "$LOCK"; mkdir "$LOCK" || { echo "lock race lost"; exit 0; }
fi
echo "$SLURM_JOB_ID" > "$LOCK/job"
trap 'rm -rf "$LOCK"; ps -u "$USER" -o args= | grep -q "[t]rain_chronoloop_robot.py --dataset $DATASET" || rm -rf "$STAGED"' EXIT

# Some nodes (evc101) intermittently fail CUDA init with error 802; retry such attempts.
for attempt in 1 2 3 4 5; do
  END=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
  LIMIT=$(python -c "import datetime;print(max(0,(datetime.datetime.fromisoformat('$END')-datetime.datetime.now()).total_seconds()/3600))")
  echo "[$(date -Is)] interactive $RUN on GPUs $GPUS (job $SLURM_JOB_ID, ${LIMIT} h left) code=$CODE extra=$*"
  before=$(wc -l < "$LOG")
  # Snapshots before the --deadline-unix option (54b7c80 and older) take the relative --time-limit-hours.
  if grep -q -- "--deadline-unix" scripts/train_chronoloop_robot.py; then STOP=(--deadline-unix "$(date -d "$END" +%s)")
  else STOP=(--time-limit-hours "$LIMIT"); fi
  torchrun --standalone --nproc_per_node="$NPROC" scripts/train_chronoloop_robot.py \
    --dataset "$DATASET" --run-name "$RUN" $FLAGS --shared-dir "$ROBOT/$DATASET" \
    --stream-micro $((32 / NPROC)) --stage-latents "$STAGED" --output-dir "$OUT" "${STOP[@]}" "$@"
  STATUS=$?
  echo "[$(date -Is)] trainer exit $STATUS"
  [[ $STATUS -ne 0 ]] && tail -n +$((before + 1)) "$LOG" | grep -q "Error 802: system not yet initialized" || break
  echo "CUDA init failure (error 802), retry $attempt"; sleep 20
done
exit $STATUS
