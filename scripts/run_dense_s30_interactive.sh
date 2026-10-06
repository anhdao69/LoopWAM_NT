#!/usr/bin/env bash
# Run inside the user's existing two-H100 allocation.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR="/tmp/anhdh35_dense_s30_${SLURM_JOB_ID:?Slurm allocation required}_triton"
python scripts/run_dense_s30_job.py "$@"
