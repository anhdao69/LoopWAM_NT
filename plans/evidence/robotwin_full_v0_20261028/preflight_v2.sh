#!/bin/bash
set -euo pipefail
R=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin_full_v0_20261028
source "$R/env.sh"
pytest -q tests > "$R/gpu_final_v2.log" 2>&1
python -u scripts/run_loopwam_robotwin_full.py --root "$R/release_v2" --phase smoke
