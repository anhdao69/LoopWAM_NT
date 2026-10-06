#!/usr/bin/env bash
#SBATCH --job-name=LoopWAM-v1
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
# Submit from a pinned checkout using --chdir; pass absolute output root.
if [[ $# -ne 1 ]]; then echo 'Usage: sbatch --chdir=PINNED_CHECKOUT --output=LOG scripts/submit_loopwam_v1.sbatch OUTPUT_PARENT' >&2; exit 2; fi
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/loopwam_triton_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"
python scripts/run_loopwam_v1_job.py --output-root "$1/v1_bs128_job${SLURM_JOB_ID}" --episodes-per-task 10
