#!/usr/bin/env bash
# Source before any ChronoLoop command on UF HiPerGator (hpg-b200). Same settings as chrono_env.sh, with every
# environment, weight, dataset, cache and run directory under /blue.
CHRONO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CHRONO_ROOT
export CHRONO_BLUE=/blue/yu954774.ucf/an221229.ucf
export PYTHONPATH="$CHRONO_ROOT/src:$CHRONO_ROOT:$CHRONO_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$CHRONO_BLUE/envs/chrono/bin:$PATH"
export DIFFSYNTH_MODEL_BASE_PATH="$CHRONO_BLUE/checkpoints"
export HF_HOME=$CHRONO_BLUE/.cache/huggingface
export HF_DATASETS_CACHE=$CHRONO_BLUE/.cache/huggingface/datasets
export UV_CACHE_DIR=$CHRONO_BLUE/.cache/uv
export TORCHINDUCTOR_CACHE_DIR=$CHRONO_BLUE/.cache/inductor
export TRITON_CACHE_DIR=$CHRONO_BLUE/.cache/triton
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export CUDA_HOME=/apps/compilers/cuda/12.8.1
[ -d "$CUDA_HOME" ] && export PATH="$CUDA_HOME/bin:$PATH"
export CHRONO_DATA_ROOT=$CHRONO_BLUE/datasets
export CHRONO_SHARED=$CHRONO_BLUE/checkpoints/ChronoLoop/shared
export CHRONO_RUNS=$CHRONO_BLUE/checkpoints/ChronoLoop/runs
export CHRONO_ROBOT_SHARED=$CHRONO_BLUE/checkpoints/ChronoLoop/robot
export CHRONO_PARENT=$CHRONO_BLUE/checkpoints/ChronoLoop/parent/libero-all-suites/v0-video4-action4/policy.pt
