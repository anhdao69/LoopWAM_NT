#!/usr/bin/env bash
# Environment of the RoboCasa-GR1 elastic port (branch elastic_gr1) on the H100 cluster (worker-2, job 5758, GPUs 4-7).
# The FastWAM venv carries an editable `fastwam` from another checkout: PYTHONPATH puts this checkout first, so never run
# these scripts with `python -I`.
export ELASTIC_ROOT=${ELASTIC_ROOT:-/mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_gr1}
export EGR_WORK=${EGR_WORK:-/mnt/data/vmo-ai-task/anhdh35/elastic_gr1}
export PYTHONPATH="$ELASTIC_ROOT/src:$ELASTIC_ROOT/scripts:$ELASTIC_ROOT"
export PATH="/mnt/data/vmo-ai-task/anhdh35/FastWAM/.venv/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/data/vmo-ai-task/anhdh35/FastWAM/checkpoints
# RoboCasa-GR1 data (gr1_spec defaults, stated explicitly: the packed-window and manifest identities include them).
export ROBOCASA_GR1_ROOT=/mnt/data/vmo-ai-task/anhdh35/datasets/robocasa_gr1
export ROBOCASA_GR1_TEXT_CACHE=/mnt/data/vmo-ai-task/anhdh35/datasets/LoopWAM_NT/text_embeds_cache/robocasa_gr1
GR1_RUN=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101
export ELASTIC_VAE=/mnt/data/vmo-ai-task/anhdh35/FastWAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
export ELASTIC_MANIFEST=$GR1_RUN/shared/data/data_manifest.json
export ELASTIC_STATS=$GR1_RUN/shared/data/dataset_stats_29d.json
# The v0 training latent cache's rsync-verified NFS backup, opened read-only (first latent frame only).
export ELASTIC_LATENTS=$GR1_RUN/shared/latents_backup
# Teacher = student initializer: chosen once by scripts/select_gr1_teacher.py (epoch 10 if it beats epoch 9, else 9).
export ELASTIC_SELECTION=$EGR_WORK/teacher_selection.json
if [ -f "$ELASTIC_SELECTION" ]; then
  ELASTIC_TEACHER=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher"])' "$ELASTIC_SELECTION")
  export ELASTIC_TEACHER
fi
export ELASTIC_INIT=$EGR_WORK/init/gr1_common_init_v1.pt
# common_v1 step-condition / T4 tensors are label-seeded and width-only (512): the RoboTwin init's digests must match.
export ELASTIC_REFERENCE_INIT_JSON=/mnt/data/vmo-ai-task/anhdh35/elastic_robotwin/init/rt50_common_init_v1.json
export TMPDIR=/tmp/egr_tmp_${SLURM_JOB_ID:-local}
mkdir -p "$TMPDIR"
