#!/usr/bin/env bash
# Source before scripts/evaluate_rt_matched.py on the H100 cluster.
RT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RT_ROOT
export PATH="/lustre/fs1/home/an221229/code/LoopWAM_NT/.venv/bin:$PATH"
export PYTHONPATH="$RT_ROOT/src:$RT_ROOT:$RT_ROOT/scripts:/groups/yshang/an221229/code/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH=/groups/yshang/an221229/checkpoints/LoopWAM_NT/rt_eval_h100/libero_config
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/apps/cuda/cuda-12.6.0
export RT_WEIGHTS=/groups/yshang/an221229/checkpoints/LoopWAM_NT
export RT_VAE=/groups/yshang/an221229/checkpoints/RollingForcing/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
export RT_TEXT=/groups/yshang/an221229/data/FastWAM/text_embeds_cache/libero
export RT_OUT=/groups/yshang/an221229/checkpoints/LoopWAM_NT/rt_eval_h100
