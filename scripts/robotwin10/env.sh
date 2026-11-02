#!/usr/bin/env bash
# RoboTwin-10 campaign environment (H100 cluster). Code: LoopWAM_rt10 worktree first on PYTHONPATH.
export RT10_CODE=${RT10_CODE:-/lustre/fs1/home/an221229/code/LoopWAM_rt10}
export RT10_WORK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_work
export RT10_RUNS=/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10
export PYTHONPATH="$RT10_CODE/src:$RT10_CODE:$RT10_CODE/scripts"
export PATH="/groups/yshang/an221229/envs/FastWAM/bin:$PATH"
export DIFFSYNTH_MODEL_BASE_PATH="$RT10_CODE/checkpoints"
export HF_HOME=/groups/yshang/an221229/cache/LoopWAM_NT/huggingface
export HF_DATASETS_CACHE=/groups/yshang/an221229/cache/LoopWAM_NT/datasets
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export CUDA_HOME=/apps/cuda/cuda-12.6.0
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
