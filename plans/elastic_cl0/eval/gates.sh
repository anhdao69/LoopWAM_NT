#!/usr/bin/env bash
# CUDA-graph gates of CL0-E0/E1/E2 over all nine flat schedules, run ahead of the lanes (one GPU, sequential).
# Usage: gates.sh <GPU>
set -uo pipefail
TOOLS=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_cl0/plans/elastic_cl0/eval
GRID=flat_k1_n2,flat_k2_n2,flat_k1_n5,flat_k4_n2,flat_k2_n5,flat_k1_n10,flat_k4_n5,flat_k2_n10,flat_k4_n10
for r in CL0-E0 CL0-E1 CL0-E2; do bash $TOOLS/gate_one.sh "$r" "$r" "$GRID" "$1"; echo "[$(date -Is)] gate $r rc=$?"; done
