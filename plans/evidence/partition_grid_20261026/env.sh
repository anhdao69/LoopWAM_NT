export PARTITION_SOURCE=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_partition_grid
export PARTITION_RUN=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_grid_20261026
export PATH=/mnt/data/vmo-ai-task/anhdh35/FastWAM/.venv/bin:$PATH
export PYTHONPATH=$PARTITION_SOURCE/src:$PARTITION_SOURCE/scripts:$PARTITION_SOURCE
export HF_HOME=/mnt/data/vmo-ai-task/anhdh35/.cache/huggingface
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/data/vmo-ai-task/anhdh35/FastWAM/checkpoints
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export TMPDIR=/mnt/data/vmo-ai-task/anhdh35/tmp
export PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
cd "$PARTITION_SOURCE"
export PYTHONPATH=$PYTHONPATH:/mnt/data/vmo-ai-task/anhdh35/FastWAM/third_party/LIBERO
export LIBERO_CONFIG_PATH=/mnt/data/vmo-ai-task/anhdh35/FastWAM/third_party/LIBERO/.config
export LD_LIBRARY_PATH=/mnt/data/vmo-ai-task/anhdh35/FastWAM/third_party/system_libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
