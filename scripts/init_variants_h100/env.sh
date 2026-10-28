#!/usr/bin/env bash
# Environment for the LoopWAM v0 initialization-variant runs on the H100 cluster (v0-init-variants worktree, unmodified).
# The FastWAM env has an editable `fastwam` from another repo; PYTHONPATH must put this checkout first.
export LOOPWAM_ROOT=/lustre/fs1/home/an221229/code/LoopWAM_initvar
export PYTHONPATH="$LOOPWAM_ROOT/src:$LOOPWAM_ROOT:$LOOPWAM_ROOT/scripts"
export PATH="/groups/yshang/an221229/envs/FastWAM/bin:$PATH"
export DIFFSYNTH_MODEL_BASE_PATH="$LOOPWAM_ROOT/checkpoints"
export HF_HOME=/groups/yshang/an221229/cache/LoopWAM_NT/huggingface
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export CUDA_HOME=/apps/cuda/cuda-12.6.0
export PYTHONUNBUFFERED=1
python - <<'PY'
import fastwam, os
expected = os.path.join(os.environ['LOOPWAM_ROOT'], 'src', 'fastwam', '__init__.py')
assert fastwam.__file__ == expected, f'fastwam imported from {fastwam.__file__}, expected {expected}'
PY
