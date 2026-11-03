#!/usr/bin/env bash
# Environment of the CL-0 elastic campaign (branch elastic_cl0) on the H100 cluster (login 10.254.152.73, worker-2).
# The FastWAM venv carries an editable `fastwam` from another checkout: PYTHONPATH must put this checkout first,
# so never run these scripts with `python -I`.
export ELASTIC_ROOT=${ELASTIC_ROOT:-/mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_cl0}
export CL0_WORK=${CL0_WORK:-/mnt/data/vmo-ai-task/anhdh35/elastic_cl0}
export PYTHONPATH="$ELASTIC_ROOT/src:$ELASTIC_ROOT/scripts:$ELASTIC_ROOT"
export PATH="/mnt/data/vmo-ai-task/anhdh35/FastWAM/.venv/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
ASSETS=/mnt/data/vmo-ai-task/anhdh35/FastWAM
PARENT_RUN=$ASSETS/runs/loopwam_nt/v0_full_libero_job4659_20261006
export ELASTIC_TEACHER=$CL0_WORK/assets/CL-0_loopwam/policy.pt          # convert_chronoloop_parent.py output
export ELASTIC_VAE=$ASSETS/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
export ELASTIC_MANIFEST=$PARENT_RUN/train/data/data_manifest.json
export ELASTIC_NFS_LATENTS=$PARENT_RUN/latents
# Node-local RAM copy (stage_latents.py). Not /dev/shm: it is a 64 GB tmpfs that the GR1 job's DataLoader workers
# (job 5624, same node) need for batch transfer; worker-2 /tmp is a large RAM-backed tmpfs.
export ELASTIC_LATENTS=${ELASTIC_LATENTS:-/tmp/elastic_cl0_latents}
export ELASTIC_INIT=$CL0_WORK/init/cl0_common_init_v1.pt
# Publishing to anhdao69/LoopWAM_NT (public) is opt-in: set ELASTIC_HF_TOKEN_FILE=$HOME/.cache/huggingface/token
# to upload checkpoints to libero-all-suites/Elastic/<run> while training (scripts/upload_elastic.py).
export ELASTIC_HF_TOKEN_FILE=${ELASTIC_HF_TOKEN_FILE:-}
