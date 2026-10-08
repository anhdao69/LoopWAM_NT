#!/usr/bin/env bash
#SBATCH --job-name=test_training
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
#SBATCH --nodelist=worker-1
set -euo pipefail
if [[ $# -ne 1 ]]; then echo 'Expected absolute output parent' >&2; exit 2; fi
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LP_NUM_THREADS=1
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/loopwam_triton_${SLURM_JOB_ID}"
mkdir -p "$TRITON_CACHE_DIR"
python scripts/run_full_loop_queue.py --output-root "$1/full_loop_job${SLURM_JOB_ID}" --pairs 33 22
