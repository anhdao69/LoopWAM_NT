#!/usr/bin/env python3
"""Elastic (GR1-E0) RoboCasa-GR1 policy server: the deployed LoopWAM v0 GR1 server with the elastic student rollout.

Same starVLA websocket protocol, observation keys, preprocessing, proprio normalization, T5 text cache, fixed noise
seed per query, de-normalization and output slicing as loopwam_policy_server.py: LoopWAMGR1ElasticPolicy subclasses
LoopWAMGR1Policy and reuses its _chunk/predict_action unchanged. Only the model differs: `self.model` is an adapter
whose infer_action does what FastWAM.infer_action does with the inputs (current image -> frozen-VAE latents through
_encode_input_image_latents_tensor, CPU-seeded float32 noise [1, 32, 29]) and then runs the elastic export's
TransportPolicy.rollout with the requested schedule (loops per denoising step, NFE = len(schedule)) instead of the
parent's 10-step 4/4 loop. The export is a loopwam-s-gr1-rt-v1 file (transport_inference_gr1.load_gr1_transport).

--teacher-parity (tests only): the adapter runs the frozen-teacher 10-step velocity loop of TransportPolicy on a GR1
teacher checkpoint (--checkpoint is then the loopwam-s-gr1-v1 teacher), which preflight_gr1.py compares bit for bit
with the deployed LoopWAMGR1Policy on the same observations; it checks the adapter's image/noise/text/proprio wiring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import sys
from pathlib import Path

import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT)); sys.path.insert(0, str(CODE_ROOT / 'src'))
os.environ.setdefault('DIFFSYNTH_MODEL_BASE_PATH', str(CODE_ROOT / 'checkpoints'))

from experiments.robocasa_gr1.loopwam_policy_server import (ACTION_KEYS, STATE_KEYS, TEXT_SUFFIX, VIDEO_KEY,  # noqa: E402
                                                            LoopWAMGR1Policy)
from fastwam.datasets.lerobot.transforms import gr1 as gr1_select  # noqa: E402
from fastwam.datasets.robocasa_gr1 import ROBOT_DIM, TEXT_CACHE_DIR, VIDEO_HW, build_normalizer, stats_sha256  # noqa: E402

FLAT9 = tuple(f'flat_k{k}_n{n}' for k in (1, 2, 4) for n in (2, 5, 10))


class ElasticAdapter:
    """FastWAM.infer_action's input handling + TransportPolicy.rollout (student) or the teacher loop (parity tests)."""

    def __init__(self, policy, loops, teacher=False):
        self.policy, self.parent, self.loops, self.teacher = policy, policy.parent, tuple(loops), teacher

    def eval(self):
        self.policy.eval(); return self

    @torch.no_grad()
    def infer_action(self, input_image, proprio, context, context_mask, action_horizon, num_inference_steps, seed, rand_device):
        p = self.parent
        if action_horizon != 32 or rand_device != 'cpu' or num_inference_steps != (10 if self.teacher else len(self.loops)):
            raise ValueError('Elastic GR1 inference: 32-action chunk, CPU noise, NFE = schedule length')
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if tuple(input_image.shape) != (1, 3, *VIDEO_HW) or proprio.shape != (ROBOT_DIM,):
            raise ValueError(f'Unexpected image/proprio {tuple(input_image.shape)} {tuple(proprio.shape)}')
        # As FastWAM.infer_action: noise first (CPU generator), then the current image through the frozen VAE.
        generator = torch.Generator(device=rand_device).manual_seed(seed)
        noise = torch.randn((1, action_horizon, p.action_expert.action_dim), generator=generator, device=rand_device,
                            dtype=torch.float32).to(device=p.device, dtype=p.torch_dtype)
        latents = p._encode_input_image_latents_tensor(input_image=input_image.to(device=p.device, dtype=p.torch_dtype), tiled=False)
        # The training/verification sample layout: [B, 1, 29] proprio, BF16 text context as cached, [B, 128] mask.
        sample = dict(first_frame_latents=latents, proprio=proprio[None, None].to(p.device),
                      context=context[None].to(p.device), context_mask=context_mask[None].to(p.device))
        if not self.teacher:
            return dict(action=self.policy.rollout(sample, self.loops, noise)[0])
        context_, mask, keys, values, attention = self.policy.prefill(sample)
        condition = self.policy.conditioning(p.action_expert, context_, mask)
        times, deltas = p.infer_action_scheduler.build_inference_schedule(10, p.device, noise.dtype, shift_override=1.)
        z = noise
        for j in range(10):
            v, _ = self.policy.velocity(z, (times[j] / 1000).expand(1), condition, (keys, values, attention), teacher=True,
                                        native_timestep=times[j].expand(1))
            z = z + v * deltas[j]
        return dict(action=z[0])


class LoopWAMGR1ElasticPolicy(LoopWAMGR1Policy):
    def __init__(self, checkpoint, stats_path, vae_path, schedule, text_cache_dir=TEXT_CACHE_DIR, seed=42, n_action_steps=12,
                 device='cuda', expected_step=None, t5_server=None, video_key=VIDEO_KEY, teacher_parity=False):
        from experiments.robotwin.chronoloop_policy.deploy_policy import TextEncoder
        from fastwam.models.wan22.loopwam_init import sha256_file
        from fastwam.models.wan22.residual_transport import SCHEDULES
        from fastwam.models.wan22.transport_inference_gr1 import EXPORT_FORMAT, load_gr1_transport
        torch.set_num_threads(int(os.getenv('SERVER_THREADS', 4)))
        if schedule not in FLAT9:
            raise ValueError(f'schedule must be one of {FLAT9}')
        header = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        contract = header['training_state']['contract']
        # teacher checkpoint: the GR1 v0 contract's vae_sha256; elastic export: the elastic contract's asset digest
        trained_vae = contract['vae_sha256'] if teacher_parity else contract['asset_sha256']['vae']
        if sha256_file(vae_path) != trained_vae:
            raise ValueError('VAE differs from the one used for training')
        step = header['step']
        if expected_step is not None and step != int(expected_step):
            raise ValueError(f'Checkpoint step {step} != expected {expected_step}')
        stats = json.loads(Path(stats_path).read_text())
        if stats_sha256(stats) != contract['normalization_sha256']:
            raise ValueError('Evaluation normalization differs from training')
        if teacher_parity:
            from fastwam.models.wan22.loopwam_gr1 import create_loopwam_gr1
            from fastwam.models.wan22.transport_policy import TransportPolicy
            parent, _ = create_loopwam_gr1(vae_path, checkpoint_path=checkpoint, device=device, checkpoint_blocks=False)
            policy = TransportPolicy(parent.requires_grad_(False).eval(), kind='T0', scope='pc', step_cond=True,
                                     checkpoint_blocks=False).to(device).eval().requires_grad_(False)
            config, export_sha, name = None, None, 'teacher parity (tests only)'
        else:
            if header.get('format_version') != EXPORT_FORMAT:
                raise ValueError('Expected a loopwam-s-gr1-rt-v1 elastic export')
            policy, payload = load_gr1_transport(checkpoint, vae_path, device=device); del payload
            config, export_sha = header['transport_config'], sha256_file(checkpoint)
            name = 'LoopWAM elastic GR1-E0 (T0, P-c, step-conditioned student of the GR1 v0 4/4 teacher)'
        del header
        self.loops = tuple(SCHEDULES[schedule])
        self.model = ElasticAdapter(policy, self.loops, teacher=teacher_parity).eval()
        self.normalizer = build_normalizer(stats)
        self.text = TextEncoder(text_cache_dir, TEXT_SUFFIX, device, server=t5_server)
        self.device, self.seed = device, int(seed)
        self.nfe = 10 if teacher_parity else len(self.loops)
        self.n_action_steps, self.video_key = int(n_action_steps), video_key
        self.queries, self.latency, self.prompts = 0, [], {}
        self.metadata = dict(
            policy=name, checkpoint=str(Path(checkpoint).resolve()), step=step, export_sha256=export_sha,
            checkpoint_contract_sha256=hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
            normalization_sha256=contract['normalization_sha256'], normalization_mode=stats['normalization']['mode'],
            schedule=schedule, loops=list(self.loops), recipe=None if config is None else config.get('campaign'),
            transport_kind=None if config is None else config['kind'],
            teacher_sha256=None if config is None else config['parent']['sha256'],
            teacher_epoch=None if config is None else config['parent']['epoch'],
            action_chunk_size=32, n_action_steps=self.n_action_steps, num_inference_steps=self.nfe, cfg_scale=1.0,
            noise_seed=self.seed, action_keys=list(ACTION_KEYS), state_keys=list(STATE_KEYS), video_key=video_key,
            training_obs_image_size=list(VIDEO_HW), action_dims=dict(gr1_select.GROUP_DIMS), absolute_actions=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True, help='loopwam-s-gr1-rt-v1 elastic export')
    p.add_argument('--schedule', required=True, choices=FLAT9)
    p.add_argument('--stats', required=True, help='dataset_stats_29d.json used in training')
    p.add_argument('--vae-path', default=str(CODE_ROOT / 'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'))
    p.add_argument('--text-cache-dir', default=TEXT_CACHE_DIR)
    p.add_argument('--port', type=int, default=6500)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--n-action-steps', type=int, default=12)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--expected-step', type=int, default=None)
    p.add_argument('--t5-server', default=None)
    p.add_argument('--video-key', default=VIDEO_KEY)
    p.add_argument('--idle-timeout', type=int, default=-1)
    p.add_argument('--max-gpu-memory-gb', type=float, default=None)
    p.add_argument('--no-t5-fallback', action='store_true')
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, force=True, format='%(asctime)s %(levelname)s %(message)s')
    if a.max_gpu_memory_gb:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.max_gpu_memory_gb * 2**30 / total), 0)
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    policy = LoopWAMGR1ElasticPolicy(a.checkpoint, a.stats, a.vae_path, a.schedule, a.text_cache_dir, seed=a.seed,
                                     n_action_steps=a.n_action_steps, expected_step=a.expected_step, t5_server=a.t5_server,
                                     video_key=a.video_key)
    if a.no_t5_fallback:
        def no_encoder():
            raise RuntimeError('prompt not in the training text cache and --no-t5-fallback is set')
        policy.text._load_encoder = no_encoder
    meta = dict(policy.metadata, host=socket.gethostname(), port=a.port)
    logging.info('serving %s', json.dumps(meta))
    WebsocketPolicyServer(policy=policy, host=a.host, port=a.port, idle_timeout=a.idle_timeout, metadata=meta).serve_forever()


if __name__ == '__main__':
    main()
