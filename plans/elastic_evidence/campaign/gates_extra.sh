#!/usr/bin/env bash
# One-off (2026-10-29): gates for the conditions added at the user's request (SP11 [1,1]; E0/E1/E2 x the six remaining
# flat9 schedules), run ahead of the evaluation lanes on job 1080499 GPUs 0 and 1 (two chains in parallel).
set -uo pipefail
W=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work
GRID=flat_k2_n2,flat_k1_n5,flat_k4_n2,flat_k2_n5,flat_k4_n5,flat_k2_n10
( bash $W/gate_one.sh SP11 SP11 flat_k1_n2 0; echo "SP11 rc=$?"; bash $W/gate_one.sh E0__grid E0 $GRID 0; echo "E0__grid rc=$?" ) &
( bash $W/gate_one.sh E1__grid E1 $GRID 1; echo "E1__grid rc=$?"; bash $W/gate_one.sh E2__grid E2 $GRID 1; echo "E2__grid rc=$?" ) &
wait
echo "[$(date -Is)] gates done"
