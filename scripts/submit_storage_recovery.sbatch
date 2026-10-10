#!/usr/bin/env bash
#SBATCH --job-name=test_training
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=120:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${TRAINING_SOURCE:?immutable verified source}"
: "${RECOVERY_RUNNER:?storage recovery runner}"
: "${RECOVERY_ROOT:?fresh output root}"
: "${RECOVERY_PLAN:?queue name}"
cd "$TRAINING_SOURCE"
source scripts/h100_env.sh
export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
exec python -u "$RECOVERY_RUNNER" --plan "$RECOVERY_PLAN" --recovery-root "$RECOVERY_ROOT"
