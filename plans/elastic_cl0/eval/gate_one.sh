#!/usr/bin/env bash
# CUDA-graph gate for one run (copy of plans/elastic_evidence/campaign/gate_one.sh with this campaign's paths).
# Writes $EVAL_OUT/gates/<name>.json: actions of every listed schedule must be bit-exact between the CUDA-graph and
# eager backends on real simulator observations (scripts/preflight_rt_graph.py, 8 queries in each of two tasks).
# A lock directory makes lanes gate a name only once.
# Usage: gate_one.sh <gate name> <run> <comma-separated schedules> <GPU>
# Exit: 0 gate file present, 1 gate failed to run, 2 another process holds the lock.
set -uo pipefail
NAME=$1; RUN=$2; SCHEDULES=$3; GPU=$4
source /mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_cl0/plans/elastic_cl0/eval/eval_env.sh
cd "$ELASTIC_ROOT"
mkdir -p "$EVAL_OUT/gates" "$EVAL_OUT/logs"
out=$EVAL_OUT/gates/$NAME.json; lock=$EVAL_OUT/gates/$NAME.lock
[ -f "$out" ] && exit 0
mkdir "$lock" 2>/dev/null || exit 2
echo "[$(date -Is)] gate $NAME: $RUN $SCHEDULES on $(hostname) GPU $GPU" >> "$EVAL_OUT/logs/gate_${NAME}.log"
CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU nice -n 10 python scripts/preflight_rt_graph.py \
  --run "$RUN" --schedules "$SCHEDULES" --weights-root "$EVAL_RUNS" \
  --vae-path "$ELASTIC_VAE" --text-cache-dir "$EVAL_TEXT" --queries 8 --output "$out.tmp.$$" \
  >> "$EVAL_OUT/logs/gate_${NAME}.log" 2>&1 && mv "$out.tmp.$$" "$out"
rmdir "$lock"
[ -f "$out" ]
