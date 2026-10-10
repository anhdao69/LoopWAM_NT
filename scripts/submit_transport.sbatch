#!/usr/bin/env bash
#SBATCH --job-name=ResidualTransport
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2
#SBATCH --mem=256G
#SBATCH --time=120:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
set -euo pipefail
: "${RT_SOURCE:?immutable source path}"
export RT_CAN_REQUEUE=1
exec bash "$RT_SOURCE/scripts/run_transport_job.sh"
