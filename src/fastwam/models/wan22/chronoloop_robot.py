"""ChronoLoop for dual-arm robots (14-D joint actions and proprio).

The backbone (video expert, action-expert transformer, memory) is exactly ChronoLoop's. From the
LIBERO v0 4/4 parent, every backbone tensor is kept; only the robot interfaces whose shapes depend
on the embodiment are freshly initialized, as in the RoboTwin v0 setup (adapt_robot_dimensions):
the action encoder (14 -> 512), the action head (512 -> 14) and the proprio encoder (14 -> 4096).
"""
from __future__ import annotations

import copy
from dataclasses import asdict

import torch
import torch.nn as nn

from .chronoloop import FORMAT, ChronoConfig, ChronoLoopMoT, ChronoLoopWAM
from .loopwam import _validate_checkpoint_depth

def robot_io_modules(model):
    """The embodiment-specific (freshly initialized) layers."""
    return model.action_expert.action_encoder, model.action_expert.head, model.proprio_encoder


def _fresh_robot_interfaces(model, action_dim, proprio_dim, seed):
    action = model.action_expert
    ref = action.action_encoder.weight
    kw = dict(device=ref.device, dtype=ref.dtype)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)   # reproducible init, independent of everything before it
        action.action_encoder = nn.Linear(action_dim, action.hidden_dim, **kw)
        action.head = nn.Linear(action.hidden_dim, action_dim, **kw)
        model.proprio_encoder = nn.Linear(proprio_dim, model.proprio_encoder.out_features, **kw)
    action.action_dim = action_dim
    model.proprio_dim = proprio_dim


def create_chronoloop_robot(cfg: ChronoConfig, vae_path, *, action_dim, proprio_dim, parent_path=None,
                            checkpoint_path=None, device='cpu', checkpoint_blocks=False, model_dtype=torch.float32,
                            parent_sha256=None, interface_seed=0):
    from .loopwam_init import load_wan21_vae, target_configs
    from .wan_video_dit import WanVideoDiT
    from .action_dit import ActionDiT
    source = checkpoint_path or parent_path
    payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    video_cfg, action_cfg = target_configs(12)
    metadata = copy.deepcopy(payload['architecture'])
    if checkpoint_path is not None:
        if payload.get('format_version') != FORMAT:
            raise ValueError('Expected a ChronoLoop checkpoint')
        if payload['chrono'] != asdict(cfg):
            raise ValueError(f'Checkpoint guard: memory flags differ: saved {payload["chrono"]} vs requested {asdict(cfg)}')
        if (metadata.get('robot_action_dim'), metadata.get('robot_proprio_dim')) != (action_dim, proprio_dim):
            raise ValueError('Checkpoint guard: robot dimensions differ')
        built_action, built_proprio = action_dim, proprio_dim
    else:
        if payload.get('format_version') != 'loopwam-s-v1' or payload['version'] != 'v0':
            raise ValueError('ChronoLoop initializes from a LoopWAM v0 checkpoint')
        _validate_checkpoint_depth(payload)
        loops = payload['inference_loops']
        if (payload.get('video_loops', loops), payload.get('action_loops', loops)) != (4, 4):
            raise ValueError('ChronoLoop parent must be the v0 4/4 checkpoint')
        built_action, built_proprio = action_cfg['action_dim'], 8
    expected_action_cfg = {**action_cfg, 'action_dim': built_action}
    if metadata['target_video_config'] != video_cfg or metadata['target_action_config'] != expected_action_cfg:
        raise ValueError('Checkpoint architecture is not the declared LoopWAM-S')
    video, action = WanVideoDiT(**video_cfg), ActionDiT(**expected_action_cfg)
    vae = load_wan21_vae(vae_path, device=device, dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32)
    video, action = video.to(device=device, dtype=model_dtype), action.to(device=device, dtype=model_dtype)
    mot = ChronoLoopMoT({'video': video, 'action': action}, loops=cfg.video_loops, version='v0',
                        checkpoint_blocks=checkpoint_blocks,
                        action_loops=None if cfg.action_loops == cfg.video_loops else cfg.action_loops)
    model = ChronoLoopWAM(video_expert=video, action_expert=action, mot=mot, vae=vae, text_dim=4096,
        proprio_dim=built_proprio, device=device, torch_dtype=model_dtype, version='v0',
        exit_weight_scale=payload.get('exit_weight_scale', 1.0), architecture_metadata=metadata,
        video_train_shift=5.0, video_infer_shift=5.0, action_train_shift=1.0, action_infer_shift=1.0)
    state = {k: v for k, v in payload['mot'].items() if not k.startswith('memory.')}
    missing, unexpected = model.mot.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f'Backbone load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}')
    model.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
    if checkpoint_path is None:
        _fresh_robot_interfaces(model, action_dim, proprio_dim, interface_seed)
        metadata['target_action_config'] = {**action_cfg, 'action_dim': action_dim}
        metadata.update(robot_action_dim=action_dim, robot_proprio_dim=proprio_dim,
                        robot_interface_initialization=f'fresh nn.Linear default init, torch seed {interface_seed}',
                        robot_backbone_source='LIBERO full-suite v0 4/4 parent (all backbone tensors kept)')
        model.architecture_metadata = metadata
    memory = model.configure_chrono(cfg, parent_sha256=payload.get('parent_sha256', parent_sha256))
    if checkpoint_path is not None and memory is not None:
        mem_state = {k[len('memory.'):]: v for k, v in payload['mot'].items() if k.startswith('memory.')}
        memory.load_state_dict(mem_state, strict=True)
    model.loaded_payload_meta = {k: payload.get(k) for k in ('step', 'chrono_flags_sha256', 'parent_sha256')}
    return model, payload
