source /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin_full_v0_20261028/env.sh
export ROBOTWIN_CODE=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_robotwin_queue
export ROBOTWIN_EVAL_RUNTIME=/mnt/data/vmo-ai-task/anhdh35/robotwin_eval_runtime_20261030
export PYTHONPATH=$ROBOTWIN_EVAL_RUNTIME/extras:$ROBOTWIN_EVAL_RUNTIME/numpy1:$ROBOTWIN_EVAL_RUNTIME/overlay:$ROBOTWIN_EVAL_RUNTIME/RoboTwin:$ROBOTWIN_CODE/src:$ROBOTWIN_CODE/scripts:$ROBOTWIN_CODE
export PATH=$ROBOTWIN_EVAL_RUNTIME/overlay/bin:$PATH
export CUDA_HOME=/mnt/data/vmo-ai-task/anhdh35/cuda-12.8.1
export TORCH_EXTENSIONS_DIR=$ROBOTWIN_EVAL_RUNTIME/torch_extensions
export XDG_CACHE_HOME=/mnt/data/vmo-ai-task/anhdh35/.cache
unset MUJOCO_GL PYOPENGL_PLATFORM
cd "$ROBOTWIN_CODE"

export LD_LIBRARY_PATH=$ROBOTWIN_EVAL_RUNTIME/nvidia570:$LD_LIBRARY_PATH
export VK_ICD_FILENAMES=$ROBOTWIN_EVAL_RUNTIME/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=$ROBOTWIN_EVAL_RUNTIME/nvidia_egl.json

export LD_LIBRARY_PATH=$ROBOTWIN_EVAL_RUNTIME/nvidia570
export SAPIEN_DISABLE_RAY_TRACING=1

export LOOPWAM_ROBOTWIN_SHADER=default

export VK_LAYER_PATH=/mnt/data/vmo-ai-task/anhdh35/robotwin_eval_runtime_20261030/mesa_layer/layers
export VK_INSTANCE_LAYERS=VK_LAYER_MESA_device_select
export MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1
export LOOPWAM_VULKAN_DEVICE_SELECT=1
