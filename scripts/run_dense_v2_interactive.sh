#!/usr/bin/env bash
# Launch inside an already allocated four-GPU Slurm step, from a pinned checkout.
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: $0 OUTPUT_ROOT prepare|run" >&2
    exit 2
fi
cd "$(dirname "$0")/.."
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR="/tmp/anhdh35_dense_v2_${SLURM_JOB_ID:?Slurm allocation required}_triton"
python scripts/run_dense_v2_job.py --output-root "$1" --phase "$2"
