#!/bin/bash
set -euo pipefail
R=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_centered_20261026
while squeue --steps -h -j 4925 -o %i | grep -qx 4925.197; do sleep 5; done
exec srun --jobid=4925 --overlap --ntasks=1 --cpus-per-task=48 --gres=gpu:4 --job-name=test_training bash -c "source /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_grid_20261026/env.sh; export TMPDIR=/tmp/loopwam_centered_4925; python -u scripts/run_centered_partition_grid.py --root $R/clean_benchmark --lane B --phase benchmark"
