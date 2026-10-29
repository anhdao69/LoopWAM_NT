#!/usr/bin/env bash
# Environment for the matched LIBERO evaluation of the elastic runs (residual_trans protocol runner
# scripts/evaluate_rt_matched.py on branch `elastic`; same FastWAM env/torch 2.7.1 as training; mujoco 3.3.2, robosuite 1.4.0).
source /groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work/env.sh
export PYTHONPATH="$PYTHONPATH:/groups/yshang/an221229/code/LIBERO"
export LIBERO_CONFIG_PATH=/groups/yshang/an221229/checkpoints/LoopWAM_NT/rt_eval_h100/libero_config
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export EVAL_TEXT=/groups/yshang/an221229/data/FastWAM/text_embeds_cache/libero
export EVAL_RUNS=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic
export EVAL_OUT=/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_eval
# Conditions in priority order, RUN:SCHEDULE:GATE (eval_lane.sh v3). The runner queues them condition-major in this
# order, so a later tier never delays an earlier one. GATE names the CUDA-graph gate file gates/<GATE>.json, which
# covers every schedule listed with that gate name.
# 1) Decisive pass (plan section 12.2): E0/E2/E1 x [1,1], [1]x10, [4]x10, then SP11 x [1,1].
# 2) User request 2026-10-29 08:20: the rest of the flat9 grid of E0/E1/E2 (plan sections 3.2 and 12.3), schedule by
#    schedule in ascending action calls (36, 60, 60, 90, 150, 180), each schedule for E0, E1 and E2 in turn so paired
#    contrasts become available schedule by schedule.
EVAL_CONDITIONS=""
for r in E0 E2 E1; do for s in flat_k1_n2 flat_k1_n10 flat_k4_n10; do EVAL_CONDITIONS="$EVAL_CONDITIONS $r:$s:$r"; done; done
EVAL_CONDITIONS="$EVAL_CONDITIONS SP11:flat_k1_n2:SP11"
for s in flat_k2_n2 flat_k1_n5 flat_k4_n2 flat_k2_n5 flat_k4_n5 flat_k2_n10; do
  for r in E0 E1 E2; do EVAL_CONDITIONS="$EVAL_CONDITIONS $r:$s:${r}__grid"; done
done
# 3) 2026-10-29 13:00, after F0-PB finished training: F0-PB at its decisive points [4,1] (S8) and [4]+[1]x9 (S2)
#    (plan section 12.2; F2-PB = RT-A2 already has both on H100 with the same runner).
EVAL_CONDITIONS="$EVAL_CONDITIONS F0-PB:S8:F0-PB F0-PB:S2:F0-PB"
export EVAL_CONDITIONS=${EVAL_CONDITIONS# }
# Private CUDA MPS daemon per eval lane (see eval_lane.sh); verified identical actions (parity_rt.sh under MPS, 20/20).
export EVAL_MPS=1
