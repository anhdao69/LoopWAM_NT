#!/bin/bash
# usage: launch_train.sh TAG TRAINER_ARGS...   (Slurm step in job 5758: -c120 --cpu-bind=none --gres=gpu:8; started by orchestrate.sh)
# One 4-GPU launch of the GR1 epochs 6-10 continuation on GPUs 4-7, each rank pinned to 15 NUMA-1 CPUs (bind_gpu.sh, the
# binding used by the RoboTwin-10 4/2 run on the same GPUs). Code must be the frozen clone at b748d92 with clean scripts/src.
set -eu
source /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101/continuation_20261106/env.sh
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
TAG=$1; shift
busy=1
for i in $(seq 90); do
  busy=$(nvidia-smi -i 4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
  [ "$busy" = 0 ] && break
  sleep 10
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'; echo
[ "$busy" = 0 ] || { echo "GPUs 4-7 still in use after 15 min"; exit 3; }
[ "$(git rev-parse HEAD)" = b748d92cb8c8672869d9ae9a2d911e10613e3aba ] && [ -z "$(git status --porcelain -- scripts src)" ] \
  || { echo "code is not the frozen b748d92 with clean scripts/src"; exit 2; }
echo "[$(date -u +%FT%TZ)] $TAG: code $(git rev-parse --short HEAD) host $(hostname) job ${SLURM_JOB_ID} step ${SLURM_STEP_ID:-?}"
export CUDA_VISIBLE_DEVICES=4,5,6,7 BIND_LOG=$GR1_CONT/logs/binding.log BIND_TAG=$TAG
exec torchrun --standalone --nproc_per_node=4 --no-python "$GR1_CONT/tools/bind_gpu.sh" scripts/train_loopwam_gr1.py "$@"
