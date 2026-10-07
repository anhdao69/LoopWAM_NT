#!/usr/bin/env bash
set -euo pipefail
SOURCE=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/source_dense_s30_20261006
OUT=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_multiseed_job4728_worker1_20261007
cd "$SOURCE"
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR="/tmp/anhdh35_v0_multiseed_${SLURM_JOB_ID:?}_triton"
python "$OUT/runner.py" --output-root "$OUT" \
  --checkpoint /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_scratch_fast_bs128_20261005/latest.pt
