# RoboCasa-GR1 epochs 6-10 continuation (2026-11-06). Same environment as the five-epoch production
# (robocasa_gr1_v0_20261101/env.sh) except the code = separate clone LoopWAM_gr1cont_20261106 (branch gr1_cont_run @b748d92).
export GR1_CODE=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_gr1cont_20261106
export GR1_RUN=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101
export GR1_CONT=$GR1_RUN/continuation_20261106
export PATH=/mnt/data/vmo-ai-task/anhdh35/FastWAM/.venv/bin:$PATH
export PYTHONPATH=$GR1_CODE/src:$GR1_CODE:$GR1_CODE/scripts:$GR1_CONT/tools
export ROBOCASA_GR1_ROOT=/mnt/data/vmo-ai-task/anhdh35/datasets/robocasa_gr1
export ROBOCASA_GR1_TEXT_CACHE=/mnt/data/vmo-ai-task/anhdh35/datasets/LoopWAM_NT/text_embeds_cache/robocasa_gr1
export ROBOCASA_GR1_STATS=$GR1_RUN/shared/data/dataset_stats_29d.json
export HF_HOME=/mnt/data/vmo-ai-task/anhdh35/.cache/huggingface
export HF_XET_CACHE=$HF_HOME/xet
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/data/vmo-ai-task/anhdh35/FastWAM/checkpoints DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export TMPDIR=/tmp/gr1_v0_${SLURM_JOB_ID:-local}
mkdir -p "$TMPDIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$GR1_CODE"
