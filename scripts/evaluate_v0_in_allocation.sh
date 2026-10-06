#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "Usage: $0 TRAIN_DIR EVALUATION_DIR" >&2
    exit 2
fi
cd "$(dirname "$0")/.."
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export TRITON_CACHE_DIR="/tmp/anhdh35_v0_eval_${SLURM_JOB_ID:?Slurm allocation required}_triton"
exec torchrun --standalone --nproc_per_node=2 scripts/evaluate_loopwam_libero.py \
    --checkpoint "$1/latest.pt" \
    --vae-path checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
    --stats "$1/data/dataset_stats.json" \
    --output-dir "$2" --episodes-per-task 10
