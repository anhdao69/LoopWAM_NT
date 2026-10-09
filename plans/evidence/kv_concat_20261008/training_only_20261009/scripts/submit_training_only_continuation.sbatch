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
: "${TRAINING_SOURCE:?immutable source checkout}"
: "${TRAINING_RUNNER:?reviewed continuation script}"
: "${TRAINING_RUNNER_SHA256:?verified continuation checksum}"
: "${TRAINING_QUEUE:?concat or mix}"
: "${PREDECESSOR_JOB:?old job ID}"
: "${PREDECESSOR_ROOT:?old campaign root}"
: "${HANDOFF_DIR:?verified handoff records}"
: "${TRAINING_OUTPUT_PARENT:?fresh output parent}"
cd "$TRAINING_SOURCE"
source scripts/h100_env.sh
export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec python -u "$TRAINING_RUNNER" --phase run --runner-sha256 "$TRAINING_RUNNER_SHA256" --queue "$TRAINING_QUEUE" \
  --predecessor-job "$PREDECESSOR_JOB" --predecessor-root "$PREDECESSOR_ROOT" \
  --handoff-dir "$HANDOFF_DIR" \
  --output-root "$TRAINING_OUTPUT_PARENT/training_only_job${SLURM_JOB_ID}"
