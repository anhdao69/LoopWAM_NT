#!/usr/bin/env bash
# Freeze the current working tree as a code snapshot and submit the ChronoLoop robot pipeline from it.
#   bash scripts/chronoloop_robot_pipeline_submit.sh [--dry-run] [extra sbatch args, e.g. --dependency=afterany:JOB]
# The snapshot (git HEAD + uncommitted changes, recorded in REVISION and UNCOMMITTED.patch) is reused when the same
# content was frozen before. CHRONO_DATASET / CHRONO_EPOCHS / CHRONO_ORDER pass through to the job.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/chrono_env_hpg.sh
DRY=0; [[ "${1:-}" == --dry-run ]] && { DRY=1; shift; }
DATASET=${CHRONO_DATASET:-robotwin5cov}; EPOCHS=${CHRONO_EPOCHS:-5}
SHARED=$CHRONO_ROBOT_SHARED/$DATASET
[[ -f "$SHARED/schedule_e${EPOCHS}_s42_n32_t4_r24.json" ]] || { echo "Missing schedule in $SHARED (run --prepare-only --epochs $EPOCHS)" >&2; exit 1; }
DIGEST=$( (git rev-parse HEAD; git diff HEAD; git ls-files -o --exclude-standard -z | xargs -0 -r sha256sum) | sha256sum | cut -c1-8)
CODE=$CHRONO_BLUE/checkpoints/ChronoLoop/code/$(git rev-parse --short=12 HEAD)-$DIGEST
if [[ ! -d "$CODE" ]]; then
  mkdir -p "$CODE.tmp"
  git ls-files -co --exclude-standard -z | rsync -a --from0 --files-from=- ./ "$CODE.tmp/"
  ln -s "$CHRONO_BLUE/envs/chrono" "$CODE.tmp/.venv"
  ln -s "$CHRONO_BLUE/checkpoints" "$CODE.tmp/checkpoints"
  echo "$(git rev-parse HEAD)+$DIGEST" > "$CODE.tmp/REVISION"
  (git diff HEAD; git ls-files -o --exclude-standard) > "$CODE.tmp/UNCOMMITTED.patch"
  mv "$CODE.tmp" "$CODE"
fi
PIPE=$CHRONO_ROBOT_SHARED/runs/$DATASET/pipeline; mkdir -p "$PIPE"
CMD=(sbatch --parsable --job-name="chrono-$DATASET" --output="$PIPE/slurm_%j.out"
     --export=ALL,CHRONO_CODE="$CODE",CHRONO_DATASET="$DATASET",CHRONO_EPOCHS="$EPOCHS" "$@"
     "$CODE/scripts/chronoloop_robot_pipeline.sbatch")
echo "snapshot: $CODE"
if [[ $DRY -eq 1 ]]; then echo "dry run: ${CMD[*]}"; exit 0; fi
for v in $(compgen -e | grep '^SLURM_'); do unset "$v"; done   # submitting from inside an allocation: don't leak its values
ID=$("${CMD[@]}")
echo "$ID $(date -Is) $DATASET epochs=$EPOCHS $CODE" | tee -a "$PIPE/submissions.txt"
