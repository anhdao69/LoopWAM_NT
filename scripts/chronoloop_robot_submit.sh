#!/usr/bin/env bash
# Submit ChronoLoop robot runs (CL-0, CL-A) as H100 Slurm jobs from a frozen code snapshot of HEAD.
#   bash scripts/chronoloop_robot_submit.sh --dataset robotwin5hard [--gpus 2,4] [--time HH:MM:SS] [--mem 256G] RUN ...
# --gpus 2,4 queues a 2-GPU and a 4-GPU twin per run; whichever starts first trains and cancels the other
# (run lock in chronoloop_robot_job.sbatch). Skips COMPLETE runs and GPU counts already queued.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/chrono_env.sh
DATASET=""; TIME="24:00:00"; GPUS="2,4"; MEM="256G"; EPOCHS=10; RUNS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    *) RUNS+=("$1"); shift;;
  esac
done
[[ -n "$DATASET" ]] || { echo "--dataset is required" >&2; exit 1; }
if [[ -n "$(git status --porcelain -- src scripts)" ]]; then
  echo "Commit src/ and scripts/ first: jobs run from a snapshot of HEAD" >&2; exit 1
fi
ROBOT=/groups/yshang/an221229/checkpoints/ChronoLoop/robot
SHARED_DS=$ROBOT/$DATASET
[[ -f "$SHARED_DS/schedule_e${EPOCHS}_s42_n32_t4_r24.json" ]] || { echo "Missing schedule in $SHARED_DS (run --prepare-only)" >&2; exit 1; }
REV=$(git rev-parse --short=12 HEAD)
CODE=/groups/yshang/an221229/checkpoints/ChronoLoop/code/$REV
if [[ ! -d "$CODE" ]]; then
  mkdir -p "$CODE.tmp"
  git archive HEAD | tar -x -C "$CODE.tmp"
  ln -s "$CHRONO_ROOT/.venv" "$CODE.tmp/.venv"
  mkdir -p "$CODE.tmp/checkpoints"
  ln -s "$(readlink -f checkpoints/Wan-AI)" "$CODE.tmp/checkpoints/Wan-AI"
  ln -s "$(readlink -f checkpoints/LoopWAM)" "$CODE.tmp/checkpoints/LoopWAM"
  echo "$(git rev-parse HEAD)" > "$CODE.tmp/REVISION"
  mv "$CODE.tmp" "$CODE"
fi
COMMON="--epochs $EPOCHS --text-threads 4 --save-epochs 5,8,9,10 --checkpoint-every 500"
for RUN in "${RUNS[@]}"; do
  case "$RUN" in
    CL-0) FLAGS="--memory-tokens 0 --action-loops 4 $COMMON"; DIR="CL-0_no-memory_a4";;
    CL-A) FLAGS="--memory-tokens 16 --mem-source learned --mem-write loop --action-loops 4 $COMMON"; DIR="CL-A_mem16-learned-loopwrite_a4";;
    *) echo "Unknown run $RUN (CL-0 | CL-A)" >&2; exit 1;;
  esac
  JOB="chrono-${DATASET}-${RUN}"
  OUT=$ROBOT/runs/$DATASET/$DIR
  mkdir -p "$OUT"
  if [[ -f "$OUT/COMPLETE" ]]; then echo "$RUN: complete, skipped"; continue; fi
  for G in ${GPUS//,/ }; do
    if squeue -h -u "$USER" -n "$JOB" -o "%b" | grep -q ":$G$"; then
      echo "$RUN: a ${G}-GPU job is already queued/running, skipped"; continue
    fi
    # Values go through the environment (--export=ALL): --export=VAR=... would split CHRONO_FLAGS at commas.
    ID=$(CHRONO_RUN="$RUN" CHRONO_DATASET="$DATASET" CHRONO_FLAGS="$FLAGS" CHRONO_CODE="$CODE" CHRONO_OUT="$OUT" \
         CHRONO_SHARED_DS="$SHARED_DS" CHRONO_TIME="$TIME" CHRONO_MEM="$MEM" \
         sbatch --parsable --job-name="$JOB" --time="$TIME" --output="$OUT/slurm_%j.out" \
         --gres=gpu:nvidia_h100_80gb_hbm3:$G --cpus-per-task=$((4 * G)) --mem="$MEM" --export=ALL \
         "$CODE/scripts/chronoloop_robot_job.sbatch")
    echo "$ID $RUN $JOB ${G}xH100 $TIME $MEM $CODE $OUT" | tee -a "$ROBOT/runs/submissions.txt"
  done
done
