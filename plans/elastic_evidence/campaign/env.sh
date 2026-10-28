#!/usr/bin/env bash
# Environment for the Elastic LoopWAM campaign (branch `elastic`) on the H100 cluster.
# The FastWAM env has an editable `fastwam` from another repo; PYTHONPATH must put this checkout first.
export ELASTIC_ROOT=${ELASTIC_ROOT:-/lustre/fs1/home/an221229/code/LoopWAM_elastic}
export PYTHONPATH="$ELASTIC_ROOT/src:$ELASTIC_ROOT/scripts:$ELASTIC_ROOT"
export PATH="/groups/yshang/an221229/envs/FastWAM/bin:$PATH"
export HF_HOME=/groups/yshang/an221229/cache/LoopWAM_NT/huggingface
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export ELASTIC_TEACHER=/groups/yshang/an221229/checkpoints/ChronoLoop/parent/libero-all-suites/v0-video4-action4/policy.pt
export ELASTIC_VAE=/groups/yshang/an221229/checkpoints/RollingForcing/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
export ELASTIC_MANIFEST=/groups/yshang/an221229/checkpoints/LoopWAM_NT/full_libero_v0_v4a2_20261026/train/data/data_manifest.json
export ELASTIC_LUSTRE_LATENTS=/groups/yshang/an221229/checkpoints/LoopWAM_NT/full_libero_v0_v4a2_20261026/latents
export ELASTIC_LATENTS=${ELASTIC_LATENTS:-/dev/shm/elastic_latents_${SLURM_JOB_ID:-1080499}}
