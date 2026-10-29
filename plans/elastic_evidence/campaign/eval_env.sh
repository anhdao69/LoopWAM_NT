#!/usr/bin/env bash
# Environment for the matched LIBERO evaluation of the elastic runs (residual_trans protocol runner
# scripts/evaluate_rt_matched.py on branch `elastic`; same FastWAM env/torch 2.7.1 as training; mujoco 3.3.2, robosuite 1.4.0).
source /groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work/env.sh
export PYTHONPATH="$PYTHONPATH:/groups/yshang/an221229/code/LIBERO"
export LIBERO_CONFIG_PATH=/groups/yshang/an221229/checkpoints/LoopWAM_NT/rt_eval_h100/libero_config
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export EVAL_TEXT=/groups/yshang/an221229/data/FastWAM/text_embeds_cache/libero
export EVAL_RUNS=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic
export EVAL_OUT=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_eval
# Decisive first pass (plan section 12.2), E0/E1/E2 x [1,1], [1]x10, [4]x10.
export EVAL_ORDER="E0 E2 E1"
export EVAL_SCHEDULES="flat_k1_n2 flat_k1_n10 flat_k4_n10"
# Private CUDA MPS daemon per eval lane (see eval_lane.sh); verified identical actions (parity_rt.sh under MPS, 20/20).
export EVAL_MPS=1
