#!/usr/bin/env bash
# Submit ChronoLoop runs as H100 Slurm jobs from a frozen code snapshot of HEAD.
#   bash scripts/chronoloop_submit.sh [--after JOBID] [--gpus 4|2|2,4] [--time HH:MM:SS] [--time2 HH:MM:SS] RUN ...
# --gpus 2,4 queues a 2-GPU and a 4-GPU twin per run; whichever starts first trains and cancels the
# other (run lock in chronoloop_job.sbatch). Skips COMPLETE runs and GPU counts already queued.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/chrono_env.sh
AFTER=""; TIME="22:00:00"; TIME2=""; GPUS="4"; RUNS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --after) AFTER="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
    --time2) TIME2="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    *) RUNS+=("$1"); shift;;
  esac
done
if [[ -n "$(git status --porcelain -- src scripts)" ]]; then
  echo "Commit src/ and scripts/ first: jobs run from a snapshot of HEAD" >&2; exit 1
fi
REV=$(git rev-parse --short=12 HEAD)
CODE=/groups/yshang/an221229/checkpoints/ChronoLoop/code/$REV
if [[ ! -d "$CODE" ]]; then
  mkdir -p "$CODE.tmp"
  git archive HEAD | tar -x -C "$CODE.tmp"
  ln -s "$CHRONO_ROOT/.venv" "$CODE.tmp/.venv"
  mkdir -p "$CODE.tmp/data" "$CODE.tmp/checkpoints"
  ln -s "$(readlink -f data/lerobot_v30)" "$CODE.tmp/data/lerobot_v30"
  ln -s "$(readlink -f data/text_embeds_cache)" "$CODE.tmp/data/text_embeds_cache"
  ln -s "$(readlink -f checkpoints/Wan-AI)" "$CODE.tmp/checkpoints/Wan-AI"
  ln -s "$(readlink -f checkpoints/LoopWAM)" "$CODE.tmp/checkpoints/LoopWAM"
  echo "$(git rev-parse HEAD)" > "$CODE.tmp/REVISION"
  mv "$CODE.tmp" "$CODE"
fi
for RUN in "${RUNS[@]}"; do
  JOB=$(python scripts/chronoloop_experiments.py "$RUN" --field job)
  OUT=$CHRONO_RUNS/$(python scripts/chronoloop_experiments.py "$RUN" --field dir)
  mkdir -p "$OUT"
  if [[ -f "$OUT/COMPLETE" ]]; then echo "$RUN: complete, skipped"; continue; fi
  for G in ${GPUS//,/ }; do
    if squeue -h -u "$USER" -n "$JOB" -o "%i %b" | grep -q ":$G\b"; then
      echo "$RUN: a ${G}-GPU job is already queued/running, skipped"; continue
    fi
    T="$TIME"; [[ "$G" == 2 && -n "$TIME2" ]] && T="$TIME2"
    DEP=(); [[ -n "$AFTER" ]] && DEP=(--dependency="afterany:$AFTER")
    ID=$(sbatch --parsable --job-name="$JOB" --time="$T" "${DEP[@]}" --output="$OUT/slurm_%j.out" \
         --gres=gpu:nvidia_h100_80gb_hbm3:$G --cpus-per-task=$((4 * G)) --mem=$((64 * G))G \
         --export=ALL,CHRONO_RUN="$RUN",CHRONO_CODE="$CODE",CHRONO_OUT="$OUT" "$CODE/scripts/chronoloop_job.sbatch")
    echo "$ID $RUN $JOB ${G}xH100 $T $CODE $OUT ${AFTER:+after:$AFTER}" | tee -a "$CHRONO_RUNS/submissions.txt"
  done
done
