#!/usr/bin/env bash
set -e
source /groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work/env.sh
export CUDA_VISIBLE_DEVICES=2,3 ELASTIC_RUN=E2 ELASTIC_OUTPUT=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work/smoke/E2_B ELASTIC_INIT=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work/init/elastic_common_init_v1.pt ELASTIC_WORLD=2 ELASTIC_SAVE_EVERY=4 ELASTIC_CKPT_MIN_BLOCKS=180 ELASTIC_WORKERS=2
ELASTIC_MAX_UPDATES=4 bash $ELASTIC_ROOT/scripts/run_elastic.sh
ELASTIC_MAX_UPDATES=8 bash $ELASTIC_ROOT/scripts/run_elastic.sh
