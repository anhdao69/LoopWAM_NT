#!/usr/bin/env bash
#SBATCH --job-name=test_training
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${MIX_SOURCE:?verified source checkout}"
: "${MIX_RELEASE:?verified preflight release}"
: "${MIX_OUTPUT_PARENT:?output parent}"
cd "$MIX_SOURCE"
source scripts/h100_env.sh
export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec python -u scripts/run_full_mix_job.py --release "$MIX_RELEASE" \
    --output "$MIX_OUTPUT_PARENT/mix_full_job${SLURM_JOB_ID}"
