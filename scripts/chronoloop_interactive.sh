#!/usr/bin/env bash
# Train one ChronoLoop run inside the current interactive allocation on a GPU subset, from a
# frozen snapshot, stopping cleanly 15 min before the allocation ends (resumable).
#   bash scripts/chronoloop_interactive.sh CODE_DIR RUN GPU_LIST [extra trainer args]
set -euo pipefail
CODE="$1"; RUN="$2"; GPUS="$3"; shift 3
cd "$CODE"; source scripts/chrono_env.sh
OUT=$CHRONO_RUNS/$(python scripts/chronoloop_experiments.py "$RUN" --field dir)
mkdir -p "$OUT"
END=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
LIMIT=$(python -c "import datetime,sys;print(max(0,(datetime.datetime.fromisoformat('$END')-datetime.datetime.now()).total_seconds()/3600))")
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
export CUDA_VISIBLE_DEVICES="$GPUS"
export TRITON_CACHE_DIR="/tmp/chrono_${SLURM_JOB_ID}_${RUN}_triton"
echo "[$(date -Is)] interactive $RUN on GPUs $GPUS (job $SLURM_JOB_ID, ${LIMIT} h left) code=$CODE" >> "$OUT/interactive_${SLURM_JOB_ID}.log"
exec torchrun --standalone --nproc_per_node="$NPROC" scripts/train_chronoloop.py \
  $(python scripts/chronoloop_experiments.py "$RUN") --output-dir "$OUT" --time-limit-hours "$LIMIT" "$@" \
  >> "$OUT/interactive_${SLURM_JOB_ID}.log" 2>&1
