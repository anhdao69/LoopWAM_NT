#!/usr/bin/env bash
# Submit ChronoLoop runs as 4 x H100 Slurm jobs from a frozen code snapshot of HEAD.
#   bash scripts/chronoloop_submit.sh [--after JOBID] [--time HH:MM:SS] RUN [RUN ...]
# Skips a run that is COMPLETE or already has a queued/running job with its name.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/chrono_env.sh
AFTER=""; TIME="22:00:00"; RUNS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --after) AFTER="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
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
  if squeue -h -u "$USER" -n "$JOB" -o %i | grep -q .; then
    echo "$RUN: job already queued/running ($(squeue -h -u "$USER" -n "$JOB" -o '%i %T' | tr '\n' ' ')), skipped"; continue
  fi
  DEP=(); [[ -n "$AFTER" ]] && DEP=(--dependency="afterany:$AFTER")
  ID=$(sbatch --parsable --job-name="$JOB" --time="$TIME" "${DEP[@]}" --output="$OUT/slurm_%j.out" \
       --export=ALL,CHRONO_RUN="$RUN",CHRONO_CODE="$CODE",CHRONO_OUT="$OUT" "$CODE/scripts/chronoloop_job.sbatch")
  echo "$ID $RUN $JOB $CODE $OUT ${AFTER:+after:$AFTER}" | tee -a "$CHRONO_RUNS/submissions.txt"
done
