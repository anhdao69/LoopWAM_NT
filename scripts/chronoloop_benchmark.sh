#!/usr/bin/env bash
# Throughput benchmark for the interactive allocation: 2 concurrent 2-GPU runs vs one 4-GPU run.
#   bash scripts/chronoloop_benchmark.sh OUT_ROOT [UPDATES]
set -uo pipefail
cd "$(dirname "$0")/.."; source scripts/chrono_env.sh
ROOT="$1"; N="${2:-40}"; mkdir -p "$ROOT"
export TRITON_CACHE_DIR=/tmp/chrono_bench_triton
run() {  # name gpus run micro
  local name=$1 gpus=$2 exp=$3 micro=$4
  CUDA_VISIBLE_DEVICES=$gpus torchrun --standalone --nproc_per_node=$(echo $gpus | tr ',' '\n' | wc -l) \
    scripts/train_chronoloop.py $(python scripts/chronoloop_experiments.py $exp) --output-dir "$ROOT/$name" \
    --stream-micro $micro --max-updates $N --no-save > "$ROOT/$name.log" 2>&1
}
date -Is > "$ROOT/start_2x2"
run cl0_2gpu 0,1 CL-0 16 & P1=$!
run cla_2gpu 2,3 CL-A 16 & P2=$!
wait $P1 $P2; date -Is > "$ROOT/end_2x2"
run cl0_4gpu 0,1,2,3 CL-0 8; date -Is > "$ROOT/end_cl0_4"
run cla_4gpu 0,1,2,3 CL-A 8; date -Is > "$ROOT/end_cla_4"
