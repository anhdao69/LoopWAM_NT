"""Original LoopWAM v0 (3/6/3, video/action loops 4/4) with fresh 29-D RoboCasa-GR1 interfaces.

Identical model contract to the RoboTwin v0 run (loopwam_robot.create_loopwam_robot): fresh
canonical Wan2.1 donors (same donor mapping), original LoopMoT, original forward/loss/inference,
CPU-seeded default-linear initialization of the new action encoder / action head / proprio
encoder, FP32 weights. Only the robot dimension (29 instead of 14) and the cached latent shape
([16, 3, 32, 32]: one 256x256 ego camera) differ. No ChronoLoop module, memory or history.
"""
import copy

import torch
from torch import nn

from .loop_mot import LoopMoT
from .loopwam_init import (build_target_experts, load_init_artifact, load_wan21_vae,
                           architecture_metadata_for_version, target_configs, DONOR_INDICES, sha256_file)
from .loopwam_robot import RobotLoopWAM, RobotTrainForward, verify_optimizer_restore  # noqa: F401 (re-export)
from .loopwam_robot import PARAMETERS as ROBOTWIN_PARAMETERS

FORMAT = 'loopwam-s-gr1-v1'
ROBOT_DIM = 29
LATENT_SHAPE = (16, 3, 32, 32)
ACTION_HIDDEN, TEXT_DIM = 512, 4096
# RoboTwin 14-D policy + the 15 extra input/output dims of the three robot interfaces.
PARAMETERS = ROBOTWIN_PARAMETERS + 15 * ACTION_HIDDEN + (15 * ACTION_HIDDEN + 15) + 15 * TEXT_DIM
assert PARAMETERS == 584644701


class GR1LoopWAM(RobotLoopWAM):
    def _encode_training_video(self, sample, input_video, tiled=False):
        latents = sample.get('latents')
        if tiled or latents is None or latents.ndim != 5 or tuple(latents.shape[1:]) != LATENT_SHAPE:
            raise ValueError('RoboCasa-GR1 training requires verified cached [B,16,3,32,32] latents')
        return latents

    def save_robot(self, path, optimizer=None, step=None, training_state=None):
        from pathlib import Path
        path = Path(path)
        payload = dict(format_version=FORMAT, version='v0', video_loops=4, action_loops=4,
                       inference_loops=4, trained_max_loops=4, exit_weight_scale=1., memory_tokens=0, history_frame=0,
                       robot_dim=ROBOT_DIM, latent_shape=list(LATENT_SHAPE),
                       architecture=self.architecture_metadata, mot=self.mot.state_dict(),
                       proprio_encoder=self.proprio_encoder.state_dict(), step=step, training_state=training_state)
        if optimizer is not None:
            payload['optimizer'] = optimizer.state_dict()
        tmp = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, tmp)
        tmp.replace(path)


def fresh_gr1_interfaces(model, seed):
    ref = model.action_expert.action_encoder.weight
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        enc = nn.Linear(ROBOT_DIM, model.action_expert.hidden_dim)
        head = nn.Linear(model.action_expert.hidden_dim, ROBOT_DIM)
        prop = nn.Linear(ROBOT_DIM, model.proprio_encoder.out_features)
    model.action_expert.action_encoder = enc.to(device=ref.device, dtype=ref.dtype)
    model.action_expert.head = head.to(device=ref.device, dtype=ref.dtype)
    model.proprio_encoder = prop.to(device=ref.device, dtype=ref.dtype)
    model.action_expert.action_dim = ROBOT_DIM
    model.proprio_dim = ROBOT_DIM


def validate_gr1_payload(p):
    expected = dict(format_version=FORMAT, version='v0', video_loops=4, action_loops=4, memory_tokens=0, history_frame=0,
                    inference_loops=4, trained_max_loops=4, exit_weight_scale=1., robot_dim=ROBOT_DIM,
                    latent_shape=list(LATENT_SHAPE))
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Not an original v0 4/4 RoboCasa-GR1 checkpoint')
    v, a = target_configs(12)
    a['action_dim'] = ROBOT_DIM
    m = p['architecture']
    if (m['target_video_config'] != v or m['target_action_config'] != a or m.get('robot_proprio_dim') != ROBOT_DIM
            or m.get('donor_indices') != list(DONOR_INDICES)):
        raise ValueError('GR1 checkpoint architecture/donor mismatch')
    if m.get('initialization_mode') != 'fresh_wan_donors_no_embodied_pretraining':
        raise ValueError('GR1 initialization provenance mismatch')


def create_loopwam_gr1(vae_path, *, init_artifact=None, checkpoint_path=None, device='cpu', checkpoint_blocks=True,
                       seed=42, artifact_sha256=None, payload=None):
    # payload: an already loaded GR1 checkpoint dict (the GR1 elastic export loader); checkpoint_path is unchanged.
    if payload is not None and checkpoint_path:
        raise ValueError('Pass checkpoint_path or payload, not both')
    if checkpoint_path or payload is not None:
        p = payload if payload is not None else torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
        validate_gr1_payload(p)
        from .wan_video_dit import WanVideoDiT
        from .action_dit import ActionDiT
        m = copy.deepcopy(p['architecture'])
        video = WanVideoDiT(**m['target_video_config'])
        action = ActionDiT(**m['target_action_config'])
    else:
        if init_artifact is None:
            raise ValueError('Canonical Wan donor artifact required; trained parents are not accepted')
        artifact = load_init_artifact(init_artifact)
        video, action, _ = build_target_experts(artifact, device=device, dtype=torch.float32)
        m = architecture_metadata_for_version(artifact['metadata'], 'v0')
        m.update(initialization_mode='fresh_wan_donors_no_embodied_pretraining',
                 init_artifact_sha256=artifact_sha256 or sha256_file(init_artifact), robot_interface_seed=seed,
                 robot_proprio_dim=ROBOT_DIM, embodiment='robocasa_gr1_arms_waist_fourier_hands_29d')
        p = None
    video = video.to(device=device, dtype=torch.float32)
    action = action.to(device=device, dtype=torch.float32)
    vae = load_wan21_vae(vae_path, device=device, dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32)
    mot = LoopMoT({'video': video, 'action': action}, loops=4, version='v0', checkpoint_blocks=checkpoint_blocks)
    model = GR1LoopWAM(video_expert=video, action_expert=action, mot=mot, vae=vae, text_dim=TEXT_DIM,
                       proprio_dim=ROBOT_DIM if p else 8, device=device, torch_dtype=torch.float32, version='v0',
                       exit_weight_scale=1., architecture_metadata=m,
                       video_train_shift=5., video_infer_shift=5., action_train_shift=1., action_infer_shift=1.)
    if p:
        model.mot.load_state_dict(p['mot'], strict=True)
        model.proprio_encoder.load_state_dict(p['proprio_encoder'], strict=True)
    else:
        fresh_gr1_interfaces(model, seed)
        m['target_action_config'] = {**m['target_action_config'], 'action_dim': ROBOT_DIM}
    return model, p
