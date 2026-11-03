#!/usr/bin/env bash
# Environment of the matched LIBERO evaluation of CL0-E0/E1/E2 (H100 cluster, worker-2, job 5623 GPUs 0-3).
# Mirrors the reference H100 evaluation of the BASE10 elastic runs (plans/elastic_evidence/campaign/eval_env.sh):
# the same runner scripts/evaluate_rt_matched.py (protocol lightwam_20261009_matched, 6,000 episodes per condition),
# FastWAM venv (torch 2.7.1+cu128, mujoco 3.3.2, robosuite 1.4.0), EGL, one thread per worker, and the training
# prompt-cache bytes (checked file by file with --verify-text-hash).
source /mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_cl0/plans/elastic_cl0/env.sh
export LIBERO_ROOT=/mnt/data/vmo-ai-task/anhdh35/FastWAM/third_party/LIBERO
export PYTHONPATH="$PYTHONPATH:$LIBERO_ROOT"
export EVAL_OUT=$CL0_WORK/eval
# Private LIBERO config (setup_eval.sh writes it): without one, importing libero prompts on stdin and hangs.
export LIBERO_CONFIG_PATH=$EVAL_OUT/libero_config
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# NVIDIA 570.158.01 EGL/GL user-space libs (egl_libs.sh); the container has only the compute part of the driver.
export LD_LIBRARY_PATH=$EVAL_OUT/egl_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export __EGL_VENDOR_LIBRARY_FILENAMES=$EVAL_OUT/egl_libs/10_nvidia.json
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export EVAL_TEXT=/mnt/data/vmo-ai-task/anhdh35/FastWAM/data/text_embeds_cache/libero
export EVAL_RUNS=$CL0_WORK/runs
# epoch_10.pt sha256 of each run, pinned from the pre-evaluation audit (validation/eval_exports_audit.json).
export EVAL_EXPECTED_SHA="CL0-E0=1d47ff90a279570401f5036ea6f8c2ac32a5bd88394a32598f5fa0e8f70ea4b8,CL0-E1=d2f192e4ed62986fe07852cbb85e062f99c1209f5a0e40b35f9032d5c21edae6,CL0-E2=8e235d15cfa23410acd83165e2466e3b3781ee2a8d98a186a86a4fec79a2d8db"
# Conditions RUN:SCHEDULE:GATE in queue order: schedule-major in ascending action calls (24, 36, 60, 60, 90, 120,
# 150, 180, 300), CL0-E0, CL0-E1, CL0-E2 at each schedule (as the reference grid tier), so paired contrasts complete
# schedule by schedule. A worker reloads its checkpoint at every condition switch, which releases the previous
# schedule's CUDA graph: ~5.4 GB per worker instead of ~8.5 GB with all nine graphs resident (run-major order),
# which is what lets 12 workers share one 80 GB GPU. Queue order never changes per-group seeding. One gate per run
# covers its nine schedules.
EVAL_CONDITIONS=""
for s in flat_k1_n2 flat_k2_n2 flat_k1_n5 flat_k4_n2 flat_k2_n5 flat_k1_n10 flat_k4_n5 flat_k2_n10 flat_k4_n10; do
  for r in CL0-E0 CL0-E1 CL0-E2; do
    EVAL_CONDITIONS="$EVAL_CONDITIONS $r:$s:$r"
  done
done
export EVAL_CONDITIONS=${EVAL_CONDITIONS# }
# Private CUDA MPS daemon per lane (eval_lane.sh), as in the reference lanes; parity is re-checked under MPS here.
export EVAL_MPS=${EVAL_MPS:-1}
