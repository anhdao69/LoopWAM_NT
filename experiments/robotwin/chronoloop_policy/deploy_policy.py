"""ChronoLoop policy for the RoboTwin 2.0 evaluation harness (RoboTwin `script/eval_policy.py`).

Inputs follow the training protocol of src/fastwam/datasets/chronoloop_robot.py exactly:
- image: the head, left-wrist and right-wrist RGB frames through `preprocess_frames` (each to 240x320, then the
  256x320 head over the two 128x160 wrists, 384x320 in [-1, 1]);
- proprio: the current 14-D joint state, z-scored with the training statistics (training feeds proprio[:, 0]);
- text: DEFAULT_PROMPT around the episode instruction, encoded by the training T5 (Wan2.2-TI2V-5B umt5-xxl,
  128 tokens); prompts already in the training text cache are read from it.
Each replanning query predicts 32 normalized actions (10 Euler steps, no CFG); they are de-normalized with the
training statistics and the first `replan_steps` are executed as absolute joint targets (qpos).
CL-A's memory state is reset at episode start and advances once per query; CL-0 carries no state. CL-FRAME also
reads the frame of query k-3 (the episode's first query while k < 3), as in training and the LIBERO evaluator.
Noise is fixed per episode (the ChronoLoop LIBERO protocol): seed = `seed` * 1_000_003 + episode index.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

CODE_ROOT = Path(__file__).resolve().parents[3]          # this file resolves into the chrono worktree
for p in (CODE_ROOT / 'src', CODE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault('DIFFSYNTH_MODEL_BASE_PATH', str(CODE_ROOT / 'checkpoints'))   # the harness runs from RoboTwin
AUTHKEY = b'chronoloop-t5'

from fastwam.datasets.chronoloop_robot import ROBOT_DIM, preprocess_frames, robot_spec  # noqa: E402
from fastwam.datasets.loopwam_long import DEFAULT_PROMPT  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer  # noqa: E402

CAMERAS = ('head_camera', 'left_camera', 'right_camera')   # = cam_high, cam_left_wrist, cam_right_wrist


class TextEncoder:
    """Training-identical T5 contexts: cache hit by sha256(prompt), else the same encoder on the GPU, either in this
    process or in a shared t5_server.py (`server` = its socket path)."""

    def __init__(self, cache_dir, suffix, device, server=None):
        self.cache_dir, self.suffix, self.device = Path(cache_dir), suffix, device
        self.memo, self.encoder, self.tokenizer = {}, None, None
        self.cache_hits = self.encoded = 0
        self.client = None
        if server:
            from multiprocessing.connection import Client
            self.client = Client(server, family='AF_UNIX', authkey=AUTHKEY)

    def _load_encoder(self):
        from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
        from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
        _, text_config, _, tokenizer_config = _resolve_configs(model_id='Wan-AI/Wan2.2-TI2V-5B',
            tokenizer_model_id='Wan-AI/Wan2.1-T2V-1.3B', redirect_common_files=True)
        text_config.download_if_necessary(); tokenizer_config.download_if_necessary()   # resolves local files
        self.encoder = _load_registered_model(text_config.path, 'wan_video_text_encoder', torch_dtype=torch.bfloat16,
                                              device=self.device).eval()
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=128, clean='whitespace')

    def __call__(self, prompt):
        if prompt in self.memo:
            return self.memo[prompt]
        path = self.cache_dir / f'{hashlib.sha256(prompt.encode("utf-8")).hexdigest()}.{self.suffix}'
        if path.exists():
            payload = torch.load(path, map_location='cpu', weights_only=True)
            mask = payload['mask'].bool()
            if payload.get('packed_context', False):
                context = torch.zeros(128, 4096, dtype=payload['context'].dtype)
                context[mask] = payload['context']
            else:
                context = payload['context'].clone()
            self.cache_hits += 1
        elif self.client is not None:
            self.client.send(prompt)
            context, mask = self.client.recv()                  # bf16 bits as int16, bool
            context, mask = torch.from_numpy(context).view(torch.bfloat16), torch.from_numpy(mask)
            self.encoded += 1
        else:
            if self.encoder is None:
                self._load_encoder()
            with torch.no_grad():
                ids, mask = self.tokenizer([prompt], return_mask=True, add_special_tokens=True)
                mask = mask.to(self.device, torch.bool)
                context = self.encoder(ids.to(self.device), mask)[0].to('cpu', torch.bfloat16)
                mask = mask[0].cpu()
            self.encoded += 1
        context[~mask] = 0
        self.memo[prompt] = (context, mask)
        return context, mask


class ChronoLoopRoboTwinPolicy:
    def __init__(self, checkpoint, dataset, data_dir, vae_path, replan_steps, num_inference_steps, seed, device,
                 t5_server=None, episode_offset=0):
        from fastwam.models.wan22.chronoloop import ChronoConfig
        from fastwam.models.wan22.chronoloop_robot import create_chronoloop_robot
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        self.cfg = ChronoConfig(**payload['chrono'])
        self.step_trained = payload.get('step')
        del payload
        self.device = device
        self.model, _ = create_chronoloop_robot(self.cfg, vae_path, action_dim=ROBOT_DIM, proprio_dim=ROBOT_DIM,
                                                checkpoint_path=str(checkpoint), device=device)
        self.model.eval()
        data_dir = Path(data_dir)
        stats = json.loads((data_dir / 'dataset_stats.json').read_text())
        self.normalizer = {kind: SingleFieldLinearNormalizer(            # as RobotWindows builds it in training
            {k.removeprefix('global_'): torch.tensor(v, dtype=torch.float32)
             for k, v in stats[kind]['default'].items() if k.startswith('global_')}, mode='z-score')
            for kind in ('action', 'state')}
        spec = robot_spec(dataset)
        self.text = TextEncoder(spec.text_cache_dir, spec.text_suffix, device, server=t5_server)
        self.replan_steps, self.num_inference_steps, self.seed = int(replan_steps), int(num_inference_steps), int(seed)
        self.pending = deque()
        # episode index k (noise seed) counts from episode_offset when a job is split at its k-th accepted seed
        self.state, self.episode, self.queries, self.latency = None, int(episode_offset) - 1, 0, []
        self.images = []                                  # this episode's query frames (CL-FRAME history)

    def should_request_observation(self):
        return not self.pending

    def reset(self):
        self.pending.clear()
        self.state, self.queries = None, 0
        self.images = []
        self.episode += 1

    def _query(self, observation, instruction):
        obs = observation['observation']
        views = [torch.from_numpy(np.ascontiguousarray(obs[c]['rgb'])).permute(2, 0, 1).unsqueeze(0) for c in CAMERAS]
        image = preprocess_frames(views)[0]                                   # bf16 [3, 384, 320]
        self.images.append(image)
        history = self.images[max(0, self.queries - 3)] if self.cfg.history_frame else None
        joint = torch.as_tensor(np.asarray(observation['joint_action']['vector'], dtype=np.float32))
        proprio = self.normalizer['state'].forward(joint[None])[0]
        context, mask = self.text(DEFAULT_PROMPT.format(task=instruction))
        if self.queries == 0:
            print(json.dumps(dict(chronoloop_episode=self.episode, instruction=instruction)), flush=True)
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = self.model.chrono_infer_action(image, proprio, context, mask, state=self.state, history_image=history,
                                                 num_inference_steps=self.num_inference_steps,
                                                 seed=self.seed * 1_000_003 + self.episode)
        torch.cuda.synchronize()
        self.latency.append(time.perf_counter() - t0)
        self.state = out['state']
        self.queries += 1
        return self.normalizer['action'].backward(out['action'][None])[0].numpy()   # [32, 14] absolute joints

    def step(self, task_env, observation):
        if not self.pending:
            if observation is None:
                raise ValueError('An observation is required at a replanning step')
            chunk = self._query(observation, task_env.get_instruction())
            self.pending.extend(np.asarray(a, dtype=np.float32) for a in chunk[:self.replan_steps])
        task_env.take_action(self.pending.popleft(), action_type='qpos')


def encode_obs(observation):
    return observation


def get_model(usr_args):
    return ChronoLoopRoboTwinPolicy(
        checkpoint=usr_args['ckpt_setting'], dataset=usr_args.get('dataset', 'robotwin5hard'), data_dir=usr_args['data_dir'],
        vae_path=usr_args.get('vae_path') or str(CODE_ROOT / 'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'),
        replan_steps=usr_args.get('replan_steps', 24), num_inference_steps=usr_args.get('num_inference_steps', 10),
        seed=usr_args.get('seed', 0), device='cuda', t5_server=usr_args.get('t5_server'),
        episode_offset=usr_args.get('episode_offset', 0))


def eval(TASK_ENV, model, observation):
    model.step(TASK_ENV, encode_obs(observation))


def reset_model(model):
    model.reset()
