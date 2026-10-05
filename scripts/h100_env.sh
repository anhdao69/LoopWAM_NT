#!/usr/bin/env bash
# Source from any directory before running FastWAM on the H100 cluster.
FASTWAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FASTWAM_ROOT
export VIRTUAL_ENV="$FASTWAM_ROOT/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export DIFFSYNTH_MODEL_BASE_PATH="$FASTWAM_ROOT/checkpoints"
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export HF_HOME="${HF_HOME:-/mnt/data/vmo-ai-task/anhdh35/.cache/huggingface}"
export PYTHONPATH="$FASTWAM_ROOT/third_party/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="$FASTWAM_ROOT/third_party/LIBERO/.config"
export LD_LIBRARY_PATH="$FASTWAM_ROOT/third_party/system_libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
