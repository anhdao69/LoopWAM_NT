#!/usr/bin/env python3
"""Render backend check for the RoboCasa-GR1 eval (robocasa venv): one seeded env per task, reset, then 60 env
steps holding the current joint state (action.* = state.*). Reports reset time, mean step time, and saves the
served view (video.ego_view_bg_crop_pad_res256_freq20) at reset and after the 60 steps for a pixel comparison
across backends. MUJOCO_GL decides the backend (osmesa | egl)."""
import json, os, sys, time
import gymnasium as gym
import numpy as np
import robocasa  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

out = sys.argv[1]; tasks = sys.argv[2].split(',')
VIEW = 'video.ego_view_bg_crop_pad_res256_freq20'
GROUPS = ('left_arm', 'right_arm', 'left_hand', 'right_hand', 'waist')
res = {}
for t in tasks:
    env = gym.make(f'gr1_unified/{t}_GR1ArmsAndWaistFourierHands_Env', enable_render=True, seed=0)
    t0 = time.perf_counter(); obs, _ = env.reset(); reset_s = time.perf_counter() - t0
    frame0 = np.asarray(obs[VIEW]).copy()
    act = {f'action.{g}': np.asarray(obs[f'state.{g}'], dtype=np.float64).reshape(-1) for g in GROUPS}
    times = []
    for _ in range(60):
        t0 = time.perf_counter(); obs, r, term, trunc, info = env.step(act); times.append(time.perf_counter() - t0)
    frame60 = np.asarray(obs[VIEW]).copy()
    np.savez_compressed(f'{out}_{t}.npz', frame0=frame0, frame60=frame60)
    res[t] = dict(reset_s=round(reset_s, 2), step_ms_mean=round(1e3 * float(np.mean(times[5:])), 1),
                  step_ms_p90=round(1e3 * float(np.percentile(times[5:], 90)), 1), frame_shape=list(frame0.shape))
    env.close()
print('BENCH ' + json.dumps(dict(backend=os.environ.get('MUJOCO_GL'), results=res)), flush=True)
