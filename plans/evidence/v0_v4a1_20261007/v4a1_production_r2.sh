#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/vmo-ai-task/anhdh35/FastWAM
cd "$ROOT/runs/loopwam_nt/source_v0_v4a1_prod_r2_20261007"
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=4
export LP_NUM_THREADS=4
python "$ROOT/runs/loopwam_nt/v4a1_recover.py"
