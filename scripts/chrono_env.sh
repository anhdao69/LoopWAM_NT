#!/usr/bin/env bash
# Source before any ChronoLoop command (interactive or Slurm) on the H100 cluster.
CHRONO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CHRONO_ROOT
export PYTHONPATH="$CHRONO_ROOT/src:$CHRONO_ROOT:$CHRONO_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$CHRONO_ROOT/.venv/bin:$PATH"
export DIFFSYNTH_MODEL_BASE_PATH="$CHRONO_ROOT/checkpoints"
export HF_HOME=/groups/yshang/an221229/cache/LoopWAM_NT/huggingface
export HF_DATASETS_CACHE=/groups/yshang/an221229/cache/LoopWAM_NT/datasets
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export CUDA_HOME=/apps/cuda/cuda-12.6.0
export PATH="$CUDA_HOME/bin:$PATH"
export CHRONO_SHARED=/groups/yshang/an221229/checkpoints/ChronoLoop/shared
export CHRONO_RUNS=/groups/yshang/an221229/checkpoints/ChronoLoop/runs
export CHRONO_PARENT=/groups/yshang/an221229/checkpoints/ChronoLoop/parent/libero-all-suites/v0-video4-action4/policy.pt
