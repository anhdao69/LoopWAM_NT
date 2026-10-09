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
parent=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt
out="$parent/kv_campaign_job${SLURM_JOB_ID}"
mkdir "$out"
printf 'Job %s on %s: waiting for verified campaign launch file in %s\n' "$SLURM_JOB_ID" "$SLURMD_NODENAME" "$out"
deadline=$((SECONDS + 43200))
while [[ ! -f "$out/launch.sh" ]]; do
 if ((SECONDS > deadline)); then echo 'Verified launch gate timed out' >&2; exit 1; fi
 sleep 5
done
exec bash "$out/launch.sh"
