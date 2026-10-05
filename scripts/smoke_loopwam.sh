#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/h100_env.sh
for version in v0 v1 v2; do
  torchrun --standalone --nproc_per_node=2 scripts/train_loopwam.py \
    --version "$version" --output-dir "runs/loopwam_nt/smoke_${version}_bs128" \
    --dataset-dir data/lerobot_v30/libero_10_no_noops_lerobot \
    --global-batch 128 --microbatch 2 --max-updates 3 --smoke --workers 4 \
    > "runs/loopwam_nt/smoke_${version}_bs128.log" 2>&1
done
