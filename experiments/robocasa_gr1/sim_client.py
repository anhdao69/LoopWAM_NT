#!/usr/bin/env python3
"""RoboCasa-GR1 tabletop rollouts against a starVLA-protocol policy server (runs in the robocasa venv).

Reuses starVLA's examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py as is:
SimulationInferenceEnv (episode loop, success = any-step success, video.ego_view_bg_crop_pad_res256_freq20 ->
video.ego_view rename), VideoRecordingWrapper and MultiStepWrapper (n_action_steps env steps per query,
max_episode_steps env steps per episode). The only change is environment seeding: starVLA builds every env
unseeded, so episodes are not reproducible; here env i of task-run `seed` is built with
robosuite seed = seed * 1000 + i (scene/object sampling uses that env-owned RNG across resets).

Policies:
  loopwam : sends the GR00T observation dict at native 256x256 with raw state.*; the server returns
            de-normalized action.* groups (n_action_steps, D).
  starvla : starVLA's own PolicyWarper (224x224 resize, optional sin/cos state; --no-send-state for
            Qwen3-VL-OFT-Robocasa), i.e. the released reference client, for check (a).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np

from examples.simBenchmarks.Robocasa_tabletop.eval_files import simulation_env as sim
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

ACTION_KEYS = ('action.left_arm', 'action.right_arm', 'action.left_hand', 'action.right_hand', 'action.waist')
ACTION_DIMS = dict(zip(ACTION_KEYS, (7, 7, 6, 6, 3)))
STATE_KEYS = tuple(k.replace('action.', 'state.') for k in ACTION_KEYS)
TEXT_KEY = 'annotation.human.coarse_action'


class LoopWAMClient:
    """BasePolicy-style object for SimulationInferenceEnv: step(observations) -> {'actions': {action.*: (B, n, D)}}."""

    def __init__(self, host, port, n_action_steps):
        self.client = WebsocketClientPolicy(host, port)
        self.metadata = self.client.get_server_metadata()
        self.n_action_steps = int(n_action_steps)
        self.calls, self.latency, self.instructions = 0, [], set()

    def step(self, observations):
        payload = {'video.ego_view': np.ascontiguousarray(observations['video.ego_view'], dtype=np.uint8)}
        for k in STATE_KEYS:
            payload[k] = np.asarray(observations[k], dtype=np.float64)
        text = observations[TEXT_KEY]
        text = [str(t) for t in (text.tolist() if isinstance(text, np.ndarray) else (list(text) if isinstance(text, (list, tuple)) else [text]))]
        payload[TEXT_KEY] = text
        self.instructions.update(text)
        t = time.perf_counter()
        resp = self.client.predict_action({'type': 'infer', 'request_id': f'q{self.calls}',
                                           'payload': {'observation': payload, 'n_action_steps': self.n_action_steps}})
        self.latency.append(time.perf_counter() - t); self.calls += 1
        if not resp.get('ok', False):
            raise RuntimeError(f'policy server error: {resp.get("error")}')
        data = resp['data']
        batch = payload['video.ego_view'].shape[0]
        actions = {}
        for k in ACTION_KEYS:
            a = np.asarray(data[k], dtype=np.float32)
            if a.shape != (batch, self.n_action_steps, ACTION_DIMS[k]) or not np.isfinite(a).all():
                raise ValueError(f'{k}: bad action {a.shape}')
            actions[k] = a
        return {'actions': actions}


def _create_seeded_env(config, idx, seed):
    """starVLA's _create_single_env with an explicit robosuite seed."""
    env = gym.make(config.env_name, enable_render=True, seed=seed * 1000 + idx)
    if config.video.video_dir is not None:
        recorder = sim.VideoRecorder.create_h264(fps=config.video.fps, codec=config.video.codec,
                                                 input_pix_fmt=config.video.input_pix_fmt, crf=config.video.crf,
                                                 thread_type=config.video.thread_type, thread_count=config.video.thread_count)
        env = sim.VideoRecordingWrapper(env, recorder, video_dir=Path(config.video.video_dir),
                                        steps_per_render=config.video.steps_per_render)
    return sim.MultiStepWrapper(env, video_delta_indices=config.multistep.video_delta_indices,
                                state_delta_indices=config.multistep.state_delta_indices,
                                n_action_steps=config.multistep.n_action_steps,
                                max_episode_steps=config.multistep.max_episode_steps)


class SeededSimulationInferenceEnv(sim.SimulationInferenceEnv):
    def __init__(self, model, seed):
        super().__init__(model)
        self.seed = int(seed)

    def setup_environment(self, config):
        fns = [partial(_create_seeded_env, config=config, idx=i, seed=self.seed) for i in range(config.n_envs)]
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(fns)
        return gym.vector.AsyncVectorEnv(fns, shared_memory=False, context='spawn')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--policy', choices=('loopwam', 'starvla'), default='loopwam')
    p.add_argument('--env-name', required=True, help='e.g. gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env')
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, required=True)
    p.add_argument('--n-episodes', type=int, default=10); p.add_argument('--n-envs', type=int, default=1)
    p.add_argument('--n-action-steps', type=int, default=16); p.add_argument('--max-episode-steps', type=int, default=720)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--video-dir', default=None); p.add_argument('--out', required=True, help='result json')
    p.add_argument('--no-send-state', action='store_true', help='starvla policy: ckpts trained without state')
    p.add_argument('--resize', type=int, default=224, help='starvla policy image size')
    a = p.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f'{out} exists')
    np.random.seed(a.seed)
    if a.policy == 'loopwam':
        model = LoopWAMClient(a.host, a.port, a.n_action_steps)
        meta = model.metadata
    else:
        from examples.simBenchmarks.Robocasa_tabletop.eval_files.model2robocasa_interface import PolicyWarper
        model = PolicyWarper(policy_ckpt_path=None, host=a.host, port=a.port, image_size=[a.resize, a.resize],
                             n_action_steps=a.n_action_steps, send_state=not a.no_send_state)
        meta = model.client.get_server_metadata()
    config = sim.SimulationConfig(env_name=a.env_name, n_episodes=a.n_episodes, n_envs=a.n_envs,
                                  video=sim.VideoConfig(video_dir=a.video_dir),
                                  multistep=sim.MultiStepConfig(n_action_steps=a.n_action_steps,
                                                                max_episode_steps=a.max_episode_steps))
    start = time.time()
    env_name, successes = SeededSimulationInferenceEnv(model, a.seed).run_simulation(config)
    successes = [bool(s) for s in successes]
    first = successes[:a.n_episodes]
    result = dict(env_name=env_name, policy=a.policy, seed=a.seed, n_episodes=a.n_episodes, n_envs=a.n_envs,
                  n_action_steps=a.n_action_steps, max_episode_steps=a.max_episode_steps,
                  successes=successes, success_rate=float(np.mean(first)), successes_first_n=sum(first),
                  wall_seconds=round(time.time() - start, 1), server_metadata=meta,
                  client=dict(queries=getattr(model, 'calls', None),
                              mean_roundtrip_s=float(np.mean(model.latency)) if getattr(model, 'latency', None) else None,
                              instructions=sorted(getattr(model, 'instructions', []))),
                  env=dict(MUJOCO_GL=os.getenv('MUJOCO_GL'), MUJOCO_EGL_DEVICE_ID=os.getenv('MUJOCO_EGL_DEVICE_ID'),
                           NUMBA_DISABLE_JIT=os.getenv('NUMBA_DISABLE_JIT')))
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('env_name', 'policy', 'n_action_steps', 'success_rate', 'successes_first_n',
                                             'n_episodes', 'wall_seconds')}), flush=True)


if __name__ == '__main__':
    main()
