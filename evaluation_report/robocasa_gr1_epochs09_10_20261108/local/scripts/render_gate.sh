#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$RUN_ROOT/third_party/starVLA"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=$PWD "$RUN_ROOT/envs/robocasa/bin/python" "$RUN_ROOT/scripts/render_bench.py" "$RUN_ROOT/evidence/render" PnPCanToDrawerClose > "$RUN_ROOT/evidence/render_gate.log" 2>&1
grep '^BENCH' "$RUN_ROOT/evidence/render_gate.log"
