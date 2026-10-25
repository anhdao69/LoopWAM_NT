#!/usr/bin/env bash
# Training-speed benchmark of a ChronoLoop robot run on a GPU pair (steady state, compiled, no checkpoints).
# Usage: bash scripts/chronoloop_robot_benchmark.sh <dataset> <tag> <gpus> [extra trainer flags...]
# Needs the latents of the first updates: train_chronoloop_robot.py --prepare-only 60, then
# chronoloop_robot_precompute.py --windows-file <shared>/needed_windows_u60.npy.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/${CHRONO_ENV:-chrono_env.sh}     # HiPerGator: CHRONO_ENV=chrono_env_hpg.sh
dataset=$1; tag=$2; gpus=$3; shift 3
root=${CHRONO_ROBOT_SHARED:-/groups/yshang/an221229/checkpoints/ChronoLoop/robot}
out=$root/bench/${dataset}_${tag}
mkdir -p "$out"
nproc=$(awk -F, '{print NF}' <<< "$gpus")
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,power.draw,memory.used --format=csv,noheader -lms 1000 \
  -i "$gpus" > "$out/gpu.csv" &
mon=$!
trap 'kill $mon 2>/dev/null || true' EXIT
CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=/tmp/chrono_robot_${SLURM_JOB_ID:-local}_triton \
  torchrun --standalone --nproc_per_node="$nproc" scripts/train_chronoloop_robot.py \
  --dataset "$dataset" --run-name "CL-A-bench-$tag" --output-dir "$out" \
  --memory-tokens 16 --mem-source learned --mem-write loop --action-loops 4 \
  --stream-micro 16 --max-updates 60 --no-save --stage-latents none "$@" 2>&1 | tee "$out/train.log"
