#!/usr/bin/env bash
# Full RoboTwin-10 frozen-VAE latent cache, built on node-local NVMe (copied to Lustre afterwards).
set -euo pipefail
source /groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_work/env.sh
cd "$RT10_CODE"
export OMP_NUM_THREADS=1
date -Is
torchrun --standalone --nproc_per_node=${NPROC:-4} scripts/chronoloop_robot_precompute.py --dataset robotwin10 \
  --shared-dir /tmp/loopwam_rt10/shared --batch 16 --workers 3 --decode-threads 2
date -Is
