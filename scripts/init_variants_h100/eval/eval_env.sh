#!/usr/bin/env bash
# Environment for the matched LIBERO evaluation of the v0 init-variant checkpoints (H100, LoopWAM_initvar code).
source /groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work/env.sh
export PYTHONPATH="$PYTHONPATH:/groups/yshang/an221229/code/LIBERO"
export LIBERO_CONFIG_PATH=/groups/yshang/an221229/checkpoints/LoopWAM_NT/rt_eval_h100/libero_config
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export EVAL_VAE=$LOOPWAM_ROOT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
export EVAL_TEXT=/groups/yshang/an221229/data/FastWAM/text_embeds_cache/libero
export EVAL_RUNS=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants
