"""LoopWAM-S: native Wan priors, shared depth, and joint exit supervision."""
from __future__ import annotations

from pathlib import Path
import math
import torch
from torch import nn

from .fastwam import FastWAM
from .loop_mot import LoopMoT, resolve_loop_count, validate_action_kv_mode


def exit_weights(version: str, loops: int, scale: float = 1.0) -> dict[int, float]:
    loops = resolve_loop_count(version, loops)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Exit weight scale must be finite and positive')
    if version == 'v0' or loops == 1:
        return {loops: scale}
    return {k: scale * (0.5 if k == loops else 0.5 / (loops - 1)) for k in range(1, loops + 1)}


def masked_action_loss(pred, target, is_pad=None):
    error = (pred.float() - target.float()).square().mean(-1)
    if is_pad is None:
        return error.mean(-1)
    if is_pad.shape != error.shape:
        raise ValueError('Action padding shape does not match predictions')
    valid = (~is_pad.bool()).to(error)
    return (error * valid).sum(-1) / valid.sum(-1).clamp_min(1)


class LoopWAM(FastWAM):
    def __init__(self, *args, version='v0', exit_weight_scale=1.0, architecture_metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.mot, LoopMoT) or self.mot.version != version:
            raise ValueError('LoopWAM requires a matching LoopMoT version')
        if self.video_expert.action_conditioned:
            raise ValueError('LoopWAM v0-v2 forbids clean action conditioning in the world stream')
        self.version = version
        self.exit_weight_scale = float(exit_weight_scale)
        exit_weights(version, self.mot.loops, self.exit_weight_scale)
        self.architecture_metadata = architecture_metadata or {}
        self.vae.requires_grad_(False).eval()
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        return self

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, **kwargs):
        if tiled:
            raise ValueError('Initial LoopWAM uses untiled per-clip VAE encoding')
        dtype = next(self.vae.parameters()).dtype
        latents = self.vae.model.encode(video_tensor.to(device=self.device, dtype=dtype), self.vae.scale)
        # Frozen per-clip encoding is complete. Native encode resets at entry,
        # but otherwise retains temporal feature tensors during policy backward.
        self.vae.model.clear_cache()
        return latents.to(self.torch_dtype)

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image, tiled=False, **kwargs):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError('Expected current images [B,3,H,W]')
        return self._encode_video_latents(input_image.unsqueeze(2), tiled=tiled)

    def _encode_training_video(self, sample, input_video, tiled=False):
        cache = getattr(self, "training_latent_cache", None)
        if cache is None:
            return self._encode_video_latents(input_video, tiled=tiled)
        if tiled:
            raise ValueError("Training latent cache requires untiled per-clip encoding")
        return cache.encode_batch(sample["training_index"], input_video, self._encode_video_latents)

    def prepare_training_batch(self, sample, tiled=False, *, noise_video=None, noise_action=None,
                               timestep_video=None, timestep_action=None):
        inputs = self.build_inputs(sample, tiled=tiled)
        clean_video, clean_action = inputs['input_latents'], inputs['action']
        batch = clean_video.shape[0]
        if inputs['first_frame_latents'] is None:
            raise ValueError('LoopWAM requires a clean current-frame anchor')
        noise_video = torch.randn_like(clean_video) if noise_video is None else noise_video
        noise_action = torch.randn_like(clean_action) if noise_action is None else noise_action
        if timestep_video is None:
            timestep_video = self.train_video_scheduler.sample_training_t(batch, clean_video.device, torch.float32)
        if timestep_action is None:
            timestep_action = self.train_action_scheduler.sample_training_t(batch, clean_action.device, torch.float32)
        noisy_video = self.train_video_scheduler.add_noise(clean_video, noise_video, timestep_video)
        noisy_video = torch.cat([clean_video[:, :, :1], noisy_video[:, :, 1:]], dim=2)
        noisy_action = self.train_action_scheduler.add_noise(clean_action, noise_action, timestep_action)
        pt, ph, pw = self.video_expert.patch_size
        frames, height, width = clean_video.shape[-3:]
        tokens_per_frame = (height // ph) * (width // pw)
        return dict(
            latents_video=noisy_video, latents_action=noisy_action,
            timestep_video=timestep_video, timestep_action=timestep_action,
            target_video=noise_video-clean_video, target_action=noise_action-clean_action,
            context=inputs['context'], context_mask=inputs['context_mask'],
            action_is_pad=inputs['action_is_pad'], image_is_pad=inputs['image_is_pad'],
            attention_mask=self._build_mot_attention_mask((frames // pt)*tokens_per_frame,
                noisy_action.shape[1], tokens_per_frame, clean_video.device),
        )

    def forward_exits(self, noisy, loops=None, exits=None):
        video = self.video_expert.prepare(x=noisy['latents_video'], timestep=noisy['timestep_video'],
            context=noisy['context'], context_mask=noisy['context_mask'], action=None,
            fuse_vae_embedding_in_latents=True)
        action = self.action_expert.prepare(action_tokens=noisy['latents_action'],
            timestep=noisy['timestep_action'], context=noisy['context'], context_mask=noisy['context_mask'])
        states = self.mot.forward_joint_exits(video_tokens=video[0], action_tokens=action[0],
            video_freqs=video[5], action_freqs=action[5], video_t_mod=video[2], action_t_mod=action[2],
            video_context=video[3], video_context_mask=video[4], action_context=action[3],
            action_context_mask=action[4], attention_mask=noisy['attention_mask'], loops=loops, exits=exits)
        return {k: (self.video_expert.post(v, video[1], video[6], video[7], video[8]),
                    self.action_expert.post(a)) for k, (v, a) in states.items()}

    def reduce_flow_losses(self, outputs, noisy, weights=None):
        weights = weights or exit_weights(self.version, self.mot.loops, self.exit_weight_scale)
        if outputs.keys() != weights.keys():
            raise ValueError('Output exits and loss weights must match exactly')
        total = noisy['latents_action'].new_zeros((), dtype=torch.float32)
        logs = {}
        video_weight = self.train_video_scheduler.training_weight(noisy['timestep_video'])
        action_weight = self.train_action_scheduler.training_weight(noisy['timestep_action'])
        for k, (pred_video, pred_action) in outputs.items():
            raw_v = self._compute_video_loss_per_sample(pred_video[:, :, 1:], noisy['target_video'][:, :, 1:],
                noisy['image_is_pad'], include_initial_video_step=False)
            raw_a = masked_action_loss(pred_action, noisy['target_action'], noisy['action_is_pad'])
            loss_v, loss_a = (raw_v * video_weight).mean(), (raw_a * action_weight).mean()
            total = total + weights[k] * (self.loss_lambda_video*loss_v + self.loss_lambda_action*loss_a)
            for name, value in [('video_raw',raw_v.mean()), ('action_raw',raw_a.mean()),
                                ('video',loss_v), ('action',loss_a)]:
                logs[f'exit{k}/{name}'] = value.detach()
        logs['loss_video'] = sum(weights[k]*logs[f'exit{k}/video'] for k in weights)
        logs['loss_action'] = sum(weights[k]*logs[f'exit{k}/action'] for k in weights)
        return total, logs

    def training_loss(self, sample, tiled=False):
        noisy = self.prepare_training_batch(sample, tiled=tiled)
        loss, logs = self.reduce_flow_losses(self.forward_exits(noisy), noisy)
        logs.update(getattr(self.mot, 'last_diagnostics', {}))
        return loss, logs

    @torch.no_grad()
    def infer_action(self, prompt=None, input_image=None, action_horizon=32, proprio=None,
                     num_inference_steps=10, **kwargs):
        if proprio is None:
            raise ValueError('LoopWAM policy inference requires current proprioception')
        if action_horizon != 32:
            raise ValueError('LoopWAM-S predicts a 32-action chunk')
        if kwargs.get('text_cfg_scale', 1.0) != 1.0:
            raise ValueError('LoopWAM v0-v2 inference uses CFG 1.0')
        return super().infer_action(prompt=prompt, input_image=input_image, action_horizon=action_horizon,
            proprio=proprio, num_inference_steps=num_inference_steps, **kwargs)

    def policy_parameters(self):
        # Module.parameters deduplicates aliases registered by the inherited wrapper.
        return [p for p in self.parameters() if p.requires_grad]

    def save_checkpoint(self, path, optimizer=None, step=None, training_state=None):
        path = Path(path)
        payload = dict(format_version='loopwam-s-v1', architecture=self.architecture_metadata,
            version=self.version, video_loops=self.mot.loops, action_loops=self.mot.action_loops,
            action_kv_mode=self.mot.action_kv_mode, loop_alignment="late", trained_max_loops=self.mot.trained_max_loops, inference_loops=self.mot.loops,
            exit_weight_scale=self.exit_weight_scale, training_state=training_state, mot=self.mot.state_dict(),
            proprio_encoder=self.proprio_encoder.state_dict(), step=step)
        if optimizer is not None:
            payload['optimizer'] = optimizer.state_dict()
        tmp = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, tmp)
        tmp.replace(path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('format_version') != 'loopwam-s-v1':
            raise ValueError('Expected a LoopWAM student checkpoint')
        if payload['version'] != self.version:
            raise ValueError('Checkpoint version differs; use the stored version when constructing the model')
        _validate_checkpoint_depth(payload)
        if payload.get('action_kv_mode', 'aligned') != self.mot.action_kv_mode:
            raise ValueError('Checkpoint action KV mode differs from constructed model')
        if payload.get('action_loops', payload['inference_loops']) != self.mot.action_loops or payload.get('video_loops', payload['inference_loops']) != self.mot.loops:
            raise ValueError('Checkpoint video/action depth differs from constructed model')
        self.mot.load_state_dict(payload['mot'], strict=True)
        self.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
        self.architecture_metadata = payload['architecture']
        self.exit_weight_scale = payload['exit_weight_scale']
        if optimizer is not None:
            optimizer.load_state_dict(payload['optimizer'])
        return payload


def _validate_checkpoint_depth(payload):
    version = payload['version']
    mode = validate_action_kv_mode(payload.get('action_kv_mode', 'aligned'), version)
    logits = payload.get('mot', {}).get('action_kv_logits')
    if mode == 'mix':
        # Infer the actual head count from the saved expert RMSNorm, including tiny tests.
        config = payload.get('architecture', {}).get('target_video_config', {})
        heads = config.get('num_heads')
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 3
                or logits.shape[0] != 6 or logits.shape[2] != payload.get('inference_loops')
                or (heads is not None and logits.shape[1] != heads)):
            raise ValueError('Invalid mix logit shape in checkpoint')
    elif logits is not None:
        raise ValueError('Non-mix checkpoint contains mix logits')
    if 'action_loops' in payload or 'video_loops' in payload:
        v, a = payload.get('video_loops'), payload.get('action_loops')
        if (type(v) is not int or type(a) is not int or not (1 <= a <= 4 and 1 <= v <= 4)
                or v != payload.get('inference_loops') or payload.get('loop_alignment') != 'late'
                or (a != v and version != 'v0')):
            raise ValueError(f'{version}: invalid checkpoint video/action loop contract')
    if version in {'dense_s12', 'dense_s30'}:
        for field in ('trained_max_loops', 'inference_loops'):
            value = payload.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value != 1:
                raise ValueError(f'{version} checkpoint requires {field}=1')

    if version == 'dense_s30':
        from .loopwam_init import target_configs
        metadata = payload.get('architecture', {})
        video_cfg, action_cfg = target_configs(30)
        if (metadata.get('architecture_version') != 'Dense-S30-native-v1'
                or metadata.get('donor_indices') != list(range(30))
                or metadata.get('target_video_config') != video_cfg
                or metadata.get('target_action_config') != action_cfg):
            raise ValueError('dense_s30 checkpoint requires native 30-layer architecture and donor identity')


def create_loopwam(init_artifact=None, vae_path=None, version='v0', loops=None, checkpoint_blocks=False,
                   model_dtype=torch.float32, device='cpu', exit_weight_scale=1.0,
                   checkpoint_path=None, action_loops=None, action_kv_mode=None):
    from .loopwam_init import (build_target_experts, load_wan21_vae, load_init_artifact,
                               target_configs, architecture_metadata_for_version)
    if vae_path is None:
        raise ValueError('Native Wan2.1 VAE path is required')
    if checkpoint_path is not None:
        # Reconstruct the declared compact student directly, without native donors.
        payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
        if payload.get('format_version') != 'loopwam-s-v1':
            raise ValueError('Expected a LoopWAM student checkpoint')
        version = payload['version']
        _validate_checkpoint_depth(payload)
        saved_mode = payload.get('action_kv_mode', 'aligned')
        if action_kv_mode is not None and action_kv_mode != saved_mode:
            raise ValueError('Checkpoint action KV mode differs from requested mode')
        action_kv_mode = saved_mode
        if saved_mode != 'aligned' and loops is not None and loops != payload['inference_loops']:
            raise ValueError('Cannot override all-loop KV trained video depth')
        loops = resolve_loop_count(version, payload['inference_loops'] if loops is None else loops)
        saved_action_loops = payload.get('action_loops', payload['inference_loops'])
        if 'action_loops' in payload and saved_action_loops != payload['inference_loops']:
            if loops != payload['inference_loops'] or (action_loops is not None and action_loops != saved_action_loops):
                raise ValueError('Cannot override trained asymmetric video/action depth')
            action_loops = saved_action_loops
        elif action_loops is not None and action_loops != loops:
            raise ValueError('Asymmetric depth must be trained from initialization')
        metadata = payload['architecture']
        video_cfg, action_cfg = target_configs(30 if version == 'dense_s30' else 12)
        if metadata['target_video_config'] != video_cfg or metadata['target_action_config'] != action_cfg:
            raise ValueError('Checkpoint architecture is not the declared LoopWAM-S')
        from .wan_video_dit import WanVideoDiT
        from .action_dit import ActionDiT
        video, action = WanVideoDiT(**video_cfg), ActionDiT(**action_cfg)
        proprio = None
        exit_weight_scale = payload['exit_weight_scale']
    else:
        action_kv_mode = validate_action_kv_mode('aligned' if action_kv_mode is None else action_kv_mode, version)
        loops = resolve_loop_count(version, loops)
        if init_artifact is None:
            raise ValueError('Provide a canonical initialization artifact or a student checkpoint')
        artifact = load_init_artifact(init_artifact)
        video, action, proprio = build_target_experts(artifact, device=device, dtype=model_dtype,
                                                       dense30=version == 'dense_s30')
        metadata = architecture_metadata_for_version(artifact['metadata'], version)
    vae = load_wan21_vae(vae_path, device=device, dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32)
    video, action = video.to(device=device,dtype=model_dtype), action.to(device=device,dtype=model_dtype)
    mot = LoopMoT({'video':video, 'action':action}, loops=loops, version=version, checkpoint_blocks=checkpoint_blocks, action_loops=action_loops, action_kv_mode=action_kv_mode)
    model = LoopWAM(video_expert=video, action_expert=action, mot=mot, vae=vae, text_dim=4096,
        proprio_dim=8, device=device, torch_dtype=model_dtype, version=version,
        exit_weight_scale=exit_weight_scale, architecture_metadata=metadata,
        video_train_shift=5.0, video_infer_shift=5.0, action_train_shift=1.0, action_infer_shift=1.0)
    if checkpoint_path:
        model.mot.load_state_dict(payload['mot'],strict=True)
        model.proprio_encoder.load_state_dict(payload['proprio_encoder'],strict=True)
    else:
        model.proprio_encoder.load_state_dict(proprio.state_dict(), strict=True)
    return model
