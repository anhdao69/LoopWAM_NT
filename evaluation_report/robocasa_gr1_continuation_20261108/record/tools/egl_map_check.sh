#!/bin/bash
# For EGL device ids 4-7 (as run_eval.py passes them: CUDA_VISIBLE_DEVICES=k MUJOCO_EGL_DEVICE_ID=k), open a MuJoCo EGL
# context, render, and record which physical GPU (nvidia-smi index) shows the process. Expect k -> k.
EGL=/mnt/data/vmo-ai-task/anhdh35/elastic_cl0/eval/egl_libs
PY=/mnt/data/vmo-ai-task/anhdh35/gr1_eval/envs/robocasa/bin/python
for k in 4 5 6 7; do
  env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=$k MUJOCO_EGL_DEVICE_ID=$k LD_LIBRARY_PATH=$EGL:${LD_LIBRARY_PATH:-}       __EGL_VENDOR_LIBRARY_FILENAMES=$EGL/10_nvidia.json $PY -c '
import mujoco, time, os
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos=\"0 0 3\"/><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>")
d = mujoco.MjData(m); r = mujoco.Renderer(m, 64, 64); mujoco.mj_forward(m, d)
for _ in range(20): r.update_scene(d); r.render()
print("pid", os.getpid(), "rendered", flush=True); time.sleep(12)' &
  pid=$!; sleep 8
  echo "EGL id $k (pid $pid): $(nvidia-smi | sed -n '/Processes/,$p' | awk -v p=$pid '$5==p {print "gpu", $2, $6, $7, $8}')"
  wait $pid
done
