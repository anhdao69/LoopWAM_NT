#!/usr/bin/env bash
export RUN_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export GR1_EVAL_ROOT=$RUN_ROOT
if [[ -x "$RUN_ROOT/policy_env/bin/python" ]]; then
 export FASTWAM_PY=$RUN_ROOT/policy_env/bin/python
else
 export FASTWAM_PY=/storage/anhdh35/miniconda3/envs/loopwam-libero-eval/bin/python
fi
export ROBOCASA_GR1_TEXT_CACHE=$RUN_ROOT/models/text_cache PYTHONPATH=$RUN_ROOT/code/src:$RUN_ROOT/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true DIFFSYNTH_MODEL_BASE_PATH=$RUN_ROOT/code/checkpoints
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NUMBA_DISABLE_JIT=1
export TMPDIR=$RUN_ROOT/tmp
mkdir -p "$TMPDIR"
