#!/usr/bin/env bash
# Environment of the RoboTwin-50 elastic port (branch elastic_robotwin) on the H100 cluster (worker-2, job 5758).
# The FastWAM venv carries an editable `fastwam` from another checkout: PYTHONPATH puts this checkout first, so never run
# these scripts with `python -I`.
export ELASTIC_ROOT=${ELASTIC_ROOT:-/mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_rt}
export ERT_WORK=${ERT_WORK:-/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin}
export PYTHONPATH="$ELASTIC_ROOT/src:$ELASTIC_ROOT/scripts:$ELASTIC_ROOT"
export PATH="/mnt/data/vmo-ai-task/anhdh35/FastWAM/.venv/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/data/vmo-ai-task/anhdh35/FastWAM/checkpoints
# RoboTwin-50 data root of robot_spec('robotwin50') (episodes, tasks, statistics, T5 text cache).
export CHRONO_DATA_ROOT=/mnt/data/vmo-ai-task/anhdh35/datasets
SHARED=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin50_20261025/shared/robotwin50
# Teacher = student initializer: full-RoboTwin v0 4/4 epoch 10 (= HF robotwin/v0-video4-action4/epoch_10/policy.pt).
export ELASTIC_TEACHER=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin_full_v0_20261028/posttrain_20261030/epochs06_10/checkpoints/epoch_10.pt
export ELASTIC_VAE=/mnt/data/vmo-ai-task/anhdh35/FastWAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
export ELASTIC_MANIFEST=$SHARED/data/data_manifest.json
# The v0 training latent cache on NFS, opened read-only (first latent frame only; ~7,600 random windows/s at 32 threads).
export ELASTIC_LATENTS=$SHARED/latents
export ELASTIC_INIT=$ERT_WORK/init/rt50_common_init_v1.pt
export TMPDIR=/tmp/ert_tmp_${SLURM_JOB_ID:-local}
mkdir -p "$TMPDIR"
