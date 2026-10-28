#!/bin/bash
set -euo pipefail
R=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin_full_v0_20261028
source "$R/env.sh"
pytest -q tests > "$R/gpu_final_tests.log" 2>&1
python -u scripts/audit_loopwam_robotwin_full.py --shared /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin50_20261025/shared/robotwin50 --output "$R/exact_audit" > "$R/exact_audit/audit.log" 2>&1
python -u scripts/run_loopwam_robotwin_full.py --root "$R" --phase smoke
