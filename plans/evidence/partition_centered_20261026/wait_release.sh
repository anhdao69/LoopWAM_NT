#!/bin/bash
set -euo pipefail
source /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_grid_20261026/env.sh
R=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_centered_20261026
for attempt in $(seq 1 240); do
 if compgen -G "$R/*_failed.json" > /dev/null; then echo "Preflight failed; no release"; exit 1; fi
 if [ -f "$R/smoke_A_complete.json" ] && [ -f "$R/smoke_B_complete.json" ]; then exec python -u "$R/release_queue.py"; fi
 sleep 10
done
echo "Smoke evidence timeout; no release"
exit 1
