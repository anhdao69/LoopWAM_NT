#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR="/tmp/anhdh35_full_v0_${SLURM_JOB_ID:?Slurm allocation required}_triton"
python scripts/run_full_libero_v0_job.py "$@"
