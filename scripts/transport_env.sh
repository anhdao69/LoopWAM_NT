#!/usr/bin/env bash
RT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RT_ROOT
export PATH="/mnt/data/vmo-ai-task/anhdh35/FastWAM/.venv/bin:$PATH"
export PYTHONPATH="$RT_ROOT/src:$RT_ROOT/scripts:$RT_ROOT:${PYTHONPATH:-}"
export HF_HOME="/mnt/data/vmo-ai-task/anhdh35/.cache/huggingface"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
