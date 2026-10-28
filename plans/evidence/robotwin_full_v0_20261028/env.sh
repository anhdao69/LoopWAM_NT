source /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin50_20261025/env.sh
export ROBOTWIN_CODE=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_robotwin_full
export ROBOTWIN_RUN=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin_full_v0_20261028
export PYTHONPATH=$ROBOTWIN_CODE/src:$ROBOTWIN_CODE/scripts:$ROBOTWIN_CODE:/mnt/data/vmo-ai-task/anhdh35/FastWAM/third_party/LIBERO
export LIBERO_CONFIG_PATH=/mnt/data/vmo-ai-task/anhdh35/FastWAM/third_party/LIBERO/.config
export TMPDIR=/tmp/loopwam_robot_${SLURM_JOB_ID:-local}
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROBOTWIN_CODE"
export LD_LIBRARY_PATH=/mnt/data/vmo-ai-task/anhdh35/FastWAM/third_party/system_libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
