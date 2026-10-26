#!/usr/bin/env bash
set -euo pipefail
source /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_grid_20261026/env.sh
export PARTITION_RUN=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_centered_20261026
export TMPDIR=/tmp/loopwam_centered_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
[ "${#devices[@]}" -eq 4 ] || { echo 'Four GPUs required'; exit 1; }
exec python -u scripts/run_centered_partition_grid.py --root "$PARTITION_RUN" --lane "${1:?lane}" --phase "${2:?phase}"
