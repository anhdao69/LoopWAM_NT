#!/usr/bin/env python3
"""LoopWAM v0 RoboCasa-GR1 policy server speaking the starVLA deployment/model_server websocket protocol.

Runs in the FastWAM venv (plus websockets/msgpack from $GR1_EVAL_ROOT/pydeps and the pinned starVLA checkout
on PYTHONPATH). Uses starVLA's WebsocketPolicyServer unchanged: msgpack-numpy frames, metadata sent on
connect, requests {"type": "infer"|"ping", "request_id", "payload"}, responses
{"status", "ok", "type": "inference_result", "request_id", "data"}.

payload = {"observation": {...GR00T keys...}, "n_action_steps": int (optional)}
  video.ego_view                  uint8 (B, T, 256, 256, 3)   latest frame used; the sim's
                                  video.ego_view_bg_crop_pad_res256_freq20 (renamed by starVLA's
                                  simulation_env) or, with --video-key, any other 256x256 view
  state.{left_arm,right_arm,left_hand,right_hand,waist}   (B, T, 7|7|6|6|3) raw joint values
  annotation.human.coarse_action  str or list[str] of length B
data = {"action.left_arm": (B, n, 7), "action.right_arm": (B, n, 7), "action.left_hand": (B, n, 6),
        "action.right_hand": (B, n, 6), "action.waist": (B, n, 3)}  de-normalized absolute joint targets,
       plus "action_chunk" (B, 32, 29) and per-call diagnostics.
Inference: 32-action chunk, 10 denoising steps (NFE 10), CFG 1, fixed noise seed per query (as RoboTwin v0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT)); sys.path.insert(0, str(CODE_ROOT / 'src'))
os.environ.setdefault('DIFFSYNTH_MODEL_BASE_PATH', str(CODE_ROOT / 'checkpoints'))

from fastwam.datasets.lerobot.transforms import gr1 as gr1_select  # noqa: E402
from fastwam.datasets.loopwam_long import DEFAULT_PROMPT  # noqa: E402
from fastwam.datasets.robocasa_gr1 import (ROBOT_DIM, TEXT_CACHE_DIR, VIDEO_HW, build_normalizer,  # noqa: E402
                                           preprocess_frames_gr1, stats_sha256)

VIDEO_KEY = 'video.ego_view'
TEXT_KEY = 'annotation.human.coarse_action'
STATE_KEYS = tuple(f'state.{g}' for g in gr1_select.GROUP_ORDER)
ACTION_KEYS = tuple(f'action.{g}' for g in gr1_select.GROUP_ORDER)
TEXT_SUFFIX = 't5_len128.wan22ti2v5b.pt'


def _texts(value, batch):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        value = [value]
    value = [v.decode() if isinstance(v, bytes) else v for v in (list(value) if isinstance(value, (list, tuple)) else [value])]
    value = [v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v for v in value]
    if len(value) == 1 and batch > 1:
        value = value * batch
    if len(value) != batch or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f'{TEXT_KEY}: expected {batch} non-empty strings, got {value!r}')
    return value


class LoopWAMGR1Policy:
    """Stateless per query: (latest ego frame, current 29-D state, instruction) -> 32 x 29 joint targets."""

    def __init__(self, checkpoint, stats_path, vae_path, text_cache_dir=TEXT_CACHE_DIR, seed=42, n_action_steps=16,
                 num_inference_steps=10, device='cuda', expected_step=None, t5_server=None, video_key=VIDEO_KEY):
        from experiments.robotwin.chronoloop_policy.deploy_policy import TextEncoder
        from fastwam.models.wan22.loop_mot import LoopMoT
        from fastwam.models.wan22.loopwam_gr1 import create_loopwam_gr1
        from fastwam.models.wan22.loopwam_init import sha256_file
        torch.set_num_threads(int(os.getenv('SERVER_THREADS', 4)))
        if num_inference_steps != 10:
            raise ValueError('GR1 v0 protocol: NFE 10')
        header = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        contract = header['training_state']['contract']
        if sha256_file(vae_path) != contract['vae_sha256']:
            raise ValueError('VAE differs from the one used for training latents')
        step = header['step']; del header
        if expected_step is not None and step != int(expected_step):
            raise ValueError(f'Checkpoint step {step} != expected {expected_step}')
        stats = json.loads(Path(stats_path).read_text())
        if stats_sha256(stats) != contract['normalization_sha256']:
            raise ValueError('Evaluation normalization differs from training')
        self.model, payload = create_loopwam_gr1(vae_path, checkpoint_path=checkpoint, device=device, checkpoint_blocks=False)
        if type(self.model.mot) is not LoopMoT:
            raise ValueError('Expected the original LoopMoT')
        self.model.eval(); del payload
        self.normalizer = build_normalizer(stats)
        self.text = TextEncoder(text_cache_dir, TEXT_SUFFIX, device, server=t5_server)
        self.device, self.seed, self.nfe = device, int(seed), int(num_inference_steps)
        self.n_action_steps, self.video_key = int(n_action_steps), video_key
        self.queries, self.latency, self.prompts = 0, [], {}
        self.metadata = dict(
            policy='LoopWAM v0 4/4 (3/6/3) RoboCasa-GR1', checkpoint=str(Path(checkpoint).resolve()), step=step,
            checkpoint_contract_sha256=hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
            normalization_sha256=contract['normalization_sha256'], normalization_mode=stats['normalization']['mode'],
            action_chunk_size=32, n_action_steps=self.n_action_steps, num_inference_steps=self.nfe, cfg_scale=1.0,
            noise_seed=self.seed, action_keys=list(ACTION_KEYS), state_keys=list(STATE_KEYS), video_key=video_key,
            training_obs_image_size=list(VIDEO_HW), action_dims=dict(gr1_select.GROUP_DIMS), absolute_actions=True)

    def _chunk(self, frame, state29, instruction):
        if frame.shape != (*VIDEO_HW, 3) or frame.dtype != np.uint8:
            raise ValueError(f'Expected uint8 256x256x3 ego frame, got {frame.shape} {frame.dtype}')
        image = preprocess_frames_gr1(torch.from_numpy(np.array(frame, dtype=np.uint8, copy=True)).permute(2, 0, 1)[None])[0]
        proprio = self.normalizer['state'].forward(torch.tensor(state29, dtype=torch.float32)[None])[0]
        prompt = DEFAULT_PROMPT.format(task=instruction)
        hits, encoded = self.text.cache_hits, self.text.encoded
        context, mask = self.text(prompt)
        if prompt not in self.prompts:
            self.prompts[prompt] = 'cache' if self.text.cache_hits > hits else ('encoded' if self.text.encoded > encoded else 'memo')
            logging.info('prompt %r -> %s', instruction, self.prompts[prompt])
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = self.model.infer_action(input_image=image, proprio=proprio, context=context, context_mask=mask,
                                          action_horizon=32, num_inference_steps=self.nfe, seed=self.seed, rand_device='cpu')
        torch.cuda.synchronize(); self.latency.append(time.perf_counter() - start); self.queries += 1
        action = out['action'].detach().float().cpu()
        if action.shape != (32, ROBOT_DIM) or not torch.isfinite(action).all():
            raise ValueError(f'Invalid action output {tuple(action.shape)}')
        return self.normalizer['action'].backward(action[None])[0].numpy().astype(np.float32)

    def predict_action(self, observation=None, n_action_steps=None, **kwargs):
        obs = observation if observation is not None else kwargs
        n = self.n_action_steps if n_action_steps is None else int(n_action_steps)
        if not 1 <= n <= 32:
            raise ValueError('n_action_steps must be in [1, 32]')
        video = np.asarray(obs[self.video_key])
        if video.ndim == 4:
            video = video[:, None]
        batch = video.shape[0]
        texts = _texts(obs[TEXT_KEY], batch)
        states = {k: np.asarray(obs[k], np.float32).reshape(batch, -1, gr1_select.GROUP_DIMS[k[6:]])[:, -1] for k in STATE_KEYS}
        chunks = np.stack([self._chunk(video[b, -1], np.concatenate([states[k][b] for k in STATE_KEYS]), texts[b])
                           for b in range(batch)])
        out = {f'action.{g}': np.ascontiguousarray(chunks[:, :n, lo:hi]) for g, (lo, hi) in gr1_select.OUTPUT_SLICES.items()}
        out.update(action_chunk=chunks, n_action_steps=n, query_index=self.queries,
                   latency_s=float(np.mean(self.latency[-batch:])))
        return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--stats', required=True, help='dataset_stats_29d.json used in training')
    p.add_argument('--vae-path', default=str(CODE_ROOT / 'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'))
    p.add_argument('--text-cache-dir', default=TEXT_CACHE_DIR)
    p.add_argument('--port', type=int, default=6500)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--n-action-steps', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--expected-step', type=int, default=None)
    p.add_argument('--t5-server', default=None)
    p.add_argument('--video-key', default=VIDEO_KEY)
    p.add_argument('--idle-timeout', type=int, default=-1)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, force=True, format='%(asctime)s %(levelname)s %(message)s')
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    policy = LoopWAMGR1Policy(a.checkpoint, a.stats, a.vae_path, a.text_cache_dir, seed=a.seed, n_action_steps=a.n_action_steps,
                              expected_step=a.expected_step, t5_server=a.t5_server, video_key=a.video_key)
    meta = dict(policy.metadata, host=socket.gethostname(), port=a.port)
    logging.info('serving %s', json.dumps(meta))
    WebsocketPolicyServer(policy=policy, host=a.host, port=a.port, idle_timeout=a.idle_timeout, metadata=meta).serve_forever()


if __name__ == '__main__':
    main()
