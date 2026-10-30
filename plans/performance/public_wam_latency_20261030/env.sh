# Environment for measure_public_latency.py (public FastWAM / FasterWAM / Light-WAM code, FastWAM conda env).
export PY=/groups/yshang/an221229/envs/FastWAM/bin/python
export CODE=/groups/yshang/an221229/code/wam_baselines_20261030
export CK=/groups/yshang/an221229/checkpoints/wam_baselines
export LAT=$CK/latency
export DIFFSYNTH_MODEL_BASE_PATH=$CK/wan_base
export DIFFSYNTH_SKIP_DOWNLOAD=true
export LIBERO_CONFIG_PATH=/groups/yshang/an221229/checkpoints/LoopWAM_NT/rt_eval_h100/libero_config
export PYTHONPATH=/groups/yshang/an221229/code/LIBERO
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8
export HF_HUB_OFFLINE=1
