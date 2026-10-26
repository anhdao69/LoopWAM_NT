#!/usr/bin/env bash
set -euo pipefail
source /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_grid_20261026/env.sh
phase=${1:?phase required}
export PARTITION_RUN=${2:-$PARTITION_RUN}
IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
if [ "${#devices[@]}" -ne 4 ]; then echo 'Expected exactly four assigned GPUs' >&2; exit 1; fi
export TMPDIR=/tmp/loopwam_partition_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
CUDA_VISIBLE_DEVICES="${devices[0]},${devices[1]}" python -u scripts/run_partition_grid.py --root "$PARTITION_RUN" --lane A --phase "$phase" > "$PARTITION_RUN/${phase}_A_driver.log" 2>&1 &
a_pid=$!
CUDA_VISIBLE_DEVICES="${devices[2]},${devices[3]}" python -u scripts/run_partition_grid.py --root "$PARTITION_RUN" --lane B --phase "$phase" > "$PARTITION_RUN/${phase}_B_driver.log" 2>&1 &
b_pid=$!
a_status=0; b_status=0
wait "$a_pid" || a_status=$?
wait "$b_pid" || b_status=$?
if [ "$a_status" -ne 0 ] || [ "$b_status" -ne 0 ]; then exit 1; fi
