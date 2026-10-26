"""Canonical, CPU FP32 Wan2.1 donor conversion for all LoopWAM variants.

Source validation uses the native architecture; student construction never goes
through a native model preset. Only action input/output and proprio are random.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from contextlib import ExitStack
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .action_dit import ActionDiT
from .wan_video_dit import WanVideoDiT, precompute_freqs_cis, precompute_freqs_cis_3d

SOURCE_MODEL = 'Wan-AI/Wan2.1-T2V-1.3B'
FASTWAM_REVISION = '7faa71108368fbb3b6885649f112af607427a2d4'
FORMAT_VERSION = 'loopwam-canonical-donors-v1'
DONOR_INDICES = (0, 1, 2, 9, 10, 11, 12, 13, 14, 27, 28, 29)
NATIVE_CONFIG = dict(model_type='t2v', dim=1536, ffn_dim=8960, freq_dim=256,
                     in_dim=16, out_dim=16, num_heads=12, num_layers=30, eps=1e-6)
VAE_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
VAE_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
           3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


def target_configs(num_layers: int = 12) -> tuple[dict, dict]:
    if num_layers not in (12, 30):
        raise ValueError('Supported physical depths are 12 and 30.')
    common = dict(text_dim=4096, freq_dim=256, eps=1e-6, num_heads=12,
                  attn_head_dim=128, num_layers=num_layers)
    video = dict(**common, hidden_dim=1536, ffn_dim=6144, in_dim=16, out_dim=16,
                 patch_size=(1, 2, 2), has_image_input=False,
                 seperated_timestep=True, video_attention_mask_mode='first_frame_causal')
    action = dict(**common, hidden_dim=512, ffn_dim=2048, action_dim=7)
    return video, action


def architecture_metadata_for_version(metadata: Mapping, version: str) -> dict:
    """Describe a student without rewriting the canonical donor artifact manifest."""
    from .loop_mot import resolve_loop_count
    resolve_loop_count(version)
    result = deepcopy(dict(metadata))
    if version == 'dense_s30':
        video_cfg, action_cfg = target_configs(30)
        result.update(architecture_version='Dense-S30-native-v1',
                      target_video_config=video_cfg, target_action_config=action_cfg,
                      donor_indices=list(range(30)))
    return result


def expected_policy_parameter_count(version: str) -> int:
    from .loop_mot import resolve_loop_count
    resolve_loop_count(version)
    return 1416114247 if version == 'dense_s30' else 584536135


def validate_source_config(config: Mapping[str, Any]) -> None:
    for key, expected in NATIVE_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f'Native Wan2.1 config {key}: expected {expected!r}, got {config.get(key)!r}')
    for key, expected in [('text_dim', 4096), ('patch_size', (1, 2, 2))]:
        value = config.get(key, expected)
        if key == 'patch_size':
            value = tuple(value)
        if value != expected:
            raise ValueError(f'Native Wan2.1 config {key}: expected {expected!r}, got {value!r}')


def _state_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


@lru_cache(maxsize=1)
def _expected_shapes() -> tuple[dict, dict, dict]:
    video_cfg, action_cfg = target_configs(30)
    with torch.device('meta'):
        native = WanVideoDiT(**{**video_cfg, 'ffn_dim': 8960})
        video, action = WanVideoDiT(**video_cfg), ActionDiT(**action_cfg)
    return _state_shapes(native), _state_shapes(video), _state_shapes(action)


def validate_state_coverage(state: Mapping, expected: Mapping[str, tuple], *, label='state') -> None:
    missing, unexpected = set(expected) - set(state), set(state) - set(expected)
    if missing or unexpected:
        raise ValueError(f'{label} coverage: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    for key, shape in expected.items():
        actual = state.shape(key) if hasattr(state, 'shape') else tuple(state[key].shape)
        if tuple(actual) != tuple(shape):
            raise ValueError(f'{label} {key}: expected shape {shape}, got {actual}')


def canonical_ffn_indices(native_width=8960, target_width=6144) -> torch.Tensor:
    if not 1 < target_width <= native_width:
        raise ValueError('FFN selection requires 1 < target_width <= native_width.')
    indices = torch.linspace(0, native_width - 1, target_width, dtype=torch.float64).round().long()
    if indices.unique().numel() != target_width or not torch.all(indices[1:] > indices[:-1]):
        raise ValueError('FFN selection must be sorted and unique.')
    return indices


def prune_paired_ffn(w1, b1, w2, b2, indices=None):
    indices = canonical_ffn_indices() if indices is None else indices
    if w1.shape[0] != b1.shape[0] or w1.shape[0] != w2.shape[1] or w2.shape[0] != b2.shape[0]:
        raise ValueError('Incompatible paired FFN shapes.')
    return w1.index_select(0, indices), b1.index_select(0, indices), w2.index_select(1, indices), b2


def resize_action_tensor(source: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """FastWAM sequential linear interpolation, then last-axis matrix scaling."""
    source = source.to(device='cpu', dtype=torch.float32)
    if tuple(source.shape) == tuple(target_shape):
        return source
    result = source
    while result.ndim < len(target_shape):
        result = result.unsqueeze(0)
    while result.ndim > len(target_shape):
        if result.shape[0] != 1:
            raise ValueError(f'Cannot resize rank {source.shape} to {target_shape}')
        result = result.squeeze(0)
    for dim, size in enumerate(target_shape):
        if result.shape[dim] == size:
            continue
        moved = result.movedim(dim, -1).contiguous()
        flat = moved.reshape(-1, 1, moved.shape[-1])
        resized = F.interpolate(flat, size=size, mode='linear', align_corners=True)
        result = resized.reshape(*moved.shape[:-1], size).movedim(-1, dim).contiguous()
    if source.ndim >= 2 and source.shape[-1] != target_shape[-1]:
        result = result * (source.shape[-1] / target_shape[-1]) ** 0.5
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class NativeSafetensors(Mapping):
    """Lazy native checkpoint reader with strict one-to-one source key mapping."""
    def __init__(self, paths):
        from safetensors import safe_open
        self._stack = ExitStack()
        self._entries = {}
        self.source_tensor_map = {}
        try:
            for path in paths:
                path = Path(path)
                handle = self._stack.enter_context(safe_open(str(path), framework='pt', device='cpu'))
                for raw_key in handle.keys():
                    key = raw_key[6:] if raw_key.startswith('model.') else raw_key
                    if key in self._entries:
                        raise ValueError(f'Duplicate source tensor {key}')
                    self._entries[key] = (handle, raw_key)
                    self.source_tensor_map[key] = {'file': path.name, 'key': raw_key}
        except BaseException:
            self.close()
            raise

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, key):
        handle, raw_key = self._entries[key]
        return handle.get_tensor(raw_key)

    def shape(self, key):
        handle, raw_key = self._entries[key]
        return tuple(handle.get_slice(raw_key).get_shape())

    def close(self):
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def build_wan_initialized_donors(source: Mapping, source_config: Mapping, *,
                               source_revision: str, source_hashes: Mapping[str, str], seed: int = 42) -> dict:
    """Transform every native tensor on CPU; retain all 30 compact donor layers."""
    validate_source_config(source_config)
    if not source_revision or not source_hashes:
        raise ValueError('An immutable source revision and checkpoint hashes are required.')
    native_shapes, video_shapes, action_shapes = _expected_shapes()
    validate_state_coverage(source, native_shapes, label='native Wan2.1')
    video_state, action_state, tensor_map = {}, {}, {}
    indices = canonical_ffn_indices()
    backbone_keys = ActionDiT.backbone_key_set(action_shapes)
    for key in sorted(native_shapes):
        native = source[key].detach().to(device='cpu', dtype=torch.float32).contiguous()
        # Always derive action from native, before pruning any video FFN neurons.
        if key in backbone_keys:
            action_state[key] = resize_action_tensor(native, action_shapes[key])
            tensor_map[f'action.{key}'] = {'source': key, 'transform': 'copy' if native.shape == action_state[key].shape else 'sequential_linear_align_corners_then_last_dim_sqrt_scale'}
        if key.startswith('blocks.') and key.endswith(('ffn.0.weight', 'ffn.0.bias')):
            video_state[key] = native.index_select(0, indices)
            operation = 'paired_ffn_select_axis_0'
        elif key.startswith('blocks.') and key.endswith('ffn.2.weight'):
            video_state[key] = native.index_select(1, indices)
            operation = 'paired_ffn_select_axis_1'
        else:
            video_state[key] = native
            operation = 'copy'
        tensor_map[f'video.{key}'] = {'source': key, 'transform': operation}
    # These are exactly the default nn.Linear initializers instantiated by ActionDiT.
    # Isolated RNG makes new heads independent of constructor order/model depth.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        action_encoder, action_head, proprio = nn.Linear(7, 512), nn.Linear(512, 7), nn.Linear(8, 4096)
    for prefix, module in [('action_encoder', action_encoder), ('head', action_head)]:
        for key, value in module.state_dict().items():
            action_state[f'{prefix}.{key}'] = value
            tensor_map[f'action.{prefix}.{key}'] = {'source': None, 'transform': 'new_nn_Linear_default', 'seed': seed}
    proprio_state = proprio.state_dict()
    for key in proprio_state:
        tensor_map[f'proprio.{key}'] = {'source': None, 'transform': 'new_nn_Linear_default', 'seed': seed}
    validate_state_coverage(video_state, video_shapes, label='video donors')
    validate_state_coverage(action_state, action_shapes, label='action donors')
    source_tensor_map = getattr(source, 'source_tensor_map', {key: {'key': key} for key in source})
    video_cfg, action_cfg = target_configs(12)
    donor_video_cfg, donor_action_cfg = target_configs(30)
    return dict(format_version=FORMAT_VERSION, video_state_dict=video_state,
                action_state_dict=action_state, proprio_state_dict=dict(proprio_state), metadata={
                    'architecture_version': 'LoopWAM-S-3-6-3', 'source_model': SOURCE_MODEL,
                    'source_revision': source_revision, 'source_hashes': dict(source_hashes),
                    'fastwam_source_revision': FASTWAM_REVISION, 'source_config': dict(source_config),
                    'target_video_config': video_cfg, 'target_action_config': action_cfg,
                    'donor_video_config': donor_video_cfg, 'donor_action_config': donor_action_cfg,
                    'donor_indices': list(DONOR_INDICES), 'ffn_indices': indices.tolist(),
                    'new_parameter_seed': seed, 'conversion_dtype': 'float32', 'conversion_device': 'cpu',
                    'source_tensor_map': source_tensor_map, 'tensor_map': tensor_map,
                    'action_resize': {'source': 'native_before_video_pruning', 'interpolation': 'sequential_1d_linear_align_corners_true', 'scaling': 'sqrt(source_last/target_last)_if_rank_ge_2_and_changed'},
                    'tokenization': {'video_latent_channels': 16, 'vae_stride': [4, 8, 8], 'patch_size': [1, 2, 2], 'text_length': 128, 'text_width': 4096, 'context_mask_required': True, 'action_horizon': 32},
                    'normalization': {'vae_mean': VAE_MEAN, 'vae_std': VAE_STD, 'vae_apply_count': 1, 'action_proprio': 'train_split_statistics_required'},
                })


def select_donor_layers(state: Mapping, indices=DONOR_INDICES) -> dict:
    lookup = {source: target for target, source in enumerate(indices)}
    if len(lookup) != len(indices):
        raise ValueError('Donor indices must be unique.')
    result = {}
    for key, value in state.items():
        if not key.startswith('blocks.'):
            result[key] = value
            continue
        _, index, suffix = key.split('.', 2)
        if int(index) in lookup:
            result[f'blocks.{lookup[int(index)]}.{suffix}'] = value
    return result


def load_init_artifact(path: str | Path) -> dict:
    artifact = torch.load(str(path), map_location='cpu', weights_only=True, mmap=True)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping) -> None:
    if artifact.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported LoopWAM initialization artifact version.')
    meta = artifact['metadata']
    revision = meta.get('source_revision', '')
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision.lower()):
        raise ValueError('Artifact requires an immutable source revision.')
    hashes = meta.get('source_hashes', {})
    if not hashes or any(len(value) != 64 or any(c not in '0123456789abcdef' for c in value.lower()) for value in hashes.values()):
        raise ValueError('Artifact requires source SHA256 hashes.')
    validate_source_config(meta['source_config'])
    if meta['source_model'] != SOURCE_MODEL or meta['donor_indices'] != list(DONOR_INDICES):
        raise ValueError('Source model or donor selection does not match LoopWAM-S.')
    if meta['ffn_indices'] != canonical_ffn_indices().tolist():
        raise ValueError('Artifact FFN selection differs from the declared paired selection.')
    expected_video, expected_action = target_configs(12)
    if meta['target_video_config'] != expected_video or meta['target_action_config'] != expected_action:
        raise ValueError('Artifact target architecture differs from LoopWAM-S.')
    native_shapes, video_shapes, action_shapes = _expected_shapes()
    for family, shapes in [('video', video_shapes), ('action', action_shapes), ('proprio', {'weight': (4096, 8), 'bias': (4096,)})]:
        validate_state_coverage(artifact[f'{family}_state_dict'], shapes, label=family)
    expected_map = {f'{family}.{key}' for family in ('video', 'action', 'proprio') for key in artifact[f'{family}_state_dict']}
    if set(meta['tensor_map']) != expected_map:
        raise ValueError('Artifact tensor provenance does not cover all target tensors.')
    if set(meta['source_tensor_map']) != set(native_shapes):
        raise ValueError('Artifact source provenance does not cover the native checkpoint.')
    for family in ('video', 'action', 'proprio'):
        for key in artifact[f'{family}_state_dict']:
            record = meta['tensor_map'][f'{family}.{key}']
            is_new = family == 'proprio' or (family == 'action' and key.startswith(('head.', 'action_encoder.')))
            expected_source = None if is_new else key
            if is_new:
                operation = 'new_nn_Linear_default'
            elif family == 'action':
                operation = 'copy' if native_shapes[key] == action_shapes[key] else 'sequential_linear_align_corners_then_last_dim_sqrt_scale'
            elif key.startswith('blocks.') and key.endswith(('ffn.0.weight', 'ffn.0.bias')):
                operation = 'paired_ffn_select_axis_0'
            elif key.startswith('blocks.') and key.endswith('ffn.2.weight'):
                operation = 'paired_ffn_select_axis_1'
            else:
                operation = 'copy'
            if record.get('source') != expected_source or record.get('transform') != operation:
                raise ValueError(f'Invalid provenance transformation for {family}.{key}')
            if is_new and record.get('seed') != meta['new_parameter_seed']:
                raise ValueError(f'Inconsistent new-parameter seed for {family}.{key}')


# Initialization variants of the 12-layer student (3 prelude + 6 core + 3 coda). A variant artifact stores the
# finished student tensors, built from the canonical 30-layer donor artifact (pruned video FFN, resized action),
# and records exactly how every student block was produced. 'default' reproduces select_donor_layers bit for bit.
STUDENT_FORMAT_VERSION = 'loopwam-student-init-v1'
INIT_VARIANTS = ('default', 'core16_21', 'core_mean', 'core_random', 'ffn_magnitude', 'stepwise')
CORE_SLOTS = tuple(range(3, 9))


def variant_block_plan(variant: str) -> list:
    """Per student slot: ['wan', donor_layer], ['mean', [donor_layers]] or ['random', core_index]."""
    if variant not in INIT_VARIANTS:
        raise ValueError(f'Unknown initialization variant {variant!r}')
    core = {
        'default': [['wan', 9 + i] for i in range(6)],
        'ffn_magnitude': [['wan', 9 + i] for i in range(6)],
        'core16_21': [['wan', 15 + i] for i in range(6)],
        'stepwise': [['wan', i] for i in (4, 8, 12, 16, 20, 24)],
        'core_mean': [['mean', [3 + i, 9 + i, 15 + i, 21 + i]] for i in range(6)],
        'core_random': [['random', i] for i in range(6)],
    }[variant]
    return [['wan', i] for i in (0, 1, 2)] + core + [['wan', i] for i in (27, 28, 29)]


def magnitude_ffn_indices(w1: torch.Tensor, w2: torch.Tensor, target_width: int = 6144) -> torch.Tensor:
    """Top neurons by ||W1[k,:]||_2 * ||W2[:,k]||_2 (FP64; ties keep the lower index), sorted ascending."""
    if w1.ndim != 2 or w2.ndim != 2 or w1.shape[0] != w2.shape[1] or not 1 < target_width <= w1.shape[0]:
        raise ValueError('Magnitude FFN selection requires paired [N,D] and [D,N] weights and 1 < width <= N.')
    score = w1.double().norm(dim=1) * w2.double().norm(dim=0)
    order = torch.sort(score, descending=True, stable=True).indices
    return order[:target_width].sort().values


def random_core_blocks(seed: int = 42) -> dict:
    """Fresh default-initialized DiTBlocks for the six core slots: one CPU RNG stream seeded once,
    video core blocks 0..5 constructed first, then action core blocks 0..5."""
    from .wan_video_dit import DiTBlock
    video_cfg, action_cfg = target_configs(12)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        blocks = {family: [DiTBlock(cfg['hidden_dim'], cfg['attn_head_dim'], cfg['num_heads'], cfg['ffn_dim'], cfg['eps'])
                           for _ in CORE_SLOTS]
                  for family, cfg in (('video', video_cfg), ('action', action_cfg))}
    return {family: [{k: v.detach().clone() for k, v in block.state_dict().items()} for block in group]
            for family, group in blocks.items()}


def compose_student_state(donor: Mapping, plan: list, random_blocks: list | None = None) -> tuple[dict, dict]:
    """Assemble one expert's 12-layer state from a 30-layer donor state following ``plan``.
    Means are taken in FP64 over the already pruned/resized donor tensors and rounded once to FP32."""
    suffixes = sorted({key.split('.', 2)[2] for key in donor if key.startswith('blocks.')})
    state, record = {}, {}
    for key, value in donor.items():
        if not key.startswith('blocks.'):
            state[key] = value.detach().clone()
            record[key] = dict(source=key, transform='copy')
    for slot, (kind, arg) in enumerate(plan):
        for suffix in suffixes:
            key = f'blocks.{slot}.{suffix}'
            if kind == 'wan':
                state[key] = donor[f'blocks.{arg}.{suffix}'].detach().clone()
                record[key] = dict(source=f'blocks.{arg}.{suffix}', transform='copy')
            elif kind == 'mean':
                stacked = torch.stack([donor[f'blocks.{i}.{suffix}'].double() for i in arg])
                state[key] = stacked.mean(0).to(donor[f'blocks.{arg[0]}.{suffix}'].dtype)
                record[key] = dict(source=[f'blocks.{i}.{suffix}' for i in arg], transform='elementwise_mean_fp64')
            elif kind == 'random':
                if random_blocks is None or set(random_blocks[arg]) != set(suffixes):
                    raise ValueError('Random core blocks do not cover the donor block tensors.')
                state[key] = random_blocks[arg][suffix].detach().clone()
                record[key] = dict(source=None, transform='new_DiTBlock_default', core_index=arg)
            else:
                raise ValueError(f'Unknown block source {kind!r}')
    return state, record


def build_init_variant(artifact: Mapping, variant: str, *, canonical_sha256: str, native: Mapping | None = None,
                       seed: int = 42) -> dict:
    """Student initialization artifact for ``variant`` from the canonical donor artifact (and, for
    ffn_magnitude, the native checkpoint it was converted from)."""
    validate_artifact(artifact)
    plan = variant_block_plan(variant)
    random_blocks = random_core_blocks(seed) if variant == 'core_random' else None
    states, tensor_map = {}, {}
    for family in ('video', 'action'):
        states[family], record = compose_student_state(artifact[f'{family}_state_dict'], plan,
                                                       None if random_blocks is None else random_blocks[family])
        tensor_map.update({f'{family}.{k}': v for k, v in record.items()})
    ffn_indices = {}
    if variant == 'ffn_magnitude':
        if native is None:
            raise ValueError('ffn_magnitude needs the native Wan2.1 checkpoint.')
        canonical = canonical_ffn_indices()
        for slot, (_, layer) in enumerate(plan):
            w1, b1 = native[f'blocks.{layer}.ffn.0.weight'].float(), native[f'blocks.{layer}.ffn.0.bias'].float()
            w2 = native[f'blocks.{layer}.ffn.2.weight'].float()
            # The native source must be the one the canonical artifact was pruned from.
            for suffix, value in (('ffn.0.weight', w1.index_select(0, canonical)), ('ffn.0.bias', b1.index_select(0, canonical)),
                                  ('ffn.2.weight', w2.index_select(1, canonical))):
                if not torch.equal(value, artifact['video_state_dict'][f'blocks.{layer}.{suffix}']):
                    raise ValueError(f'Native blocks.{layer}.{suffix} is not the canonical artifact source')
            indices = magnitude_ffn_indices(w1, w2)
            ffn_indices[slot] = indices.tolist()
            for suffix, value, axis in (('ffn.0.weight', w1, 0), ('ffn.0.bias', b1, 0), ('ffn.2.weight', w2, 1)):
                states['video'][f'blocks.{slot}.{suffix}'] = value.index_select(axis, indices).contiguous()
                tensor_map[f'video.blocks.{slot}.{suffix}'] = dict(
                    source=f'native:blocks.{layer}.{suffix}', transform=f'magnitude_ffn_select_axis_{axis}')
    proprio = {k: v.detach().clone() for k, v in artifact['proprio_state_dict'].items()}
    tensor_map.update({f'proprio.{k}': dict(source=f'proprio.{k}', transform='copy') for k in proprio})
    meta = deepcopy(dict(artifact['metadata']))
    wan_only = all(kind == 'wan' for kind, _ in plan)
    meta.update(donor_indices=[arg for _, arg in plan] if wan_only else None,
                ffn_indices=None if variant == 'ffn_magnitude' else meta['ffn_indices'],
                init_variant=variant, student_block_plan=plan, canonical_artifact_sha256=canonical_sha256,
                canonical_donor_indices=list(DONOR_INDICES), variant_seed=seed,
                ffn_rule='magnitude_top6144_per_block' if variant == 'ffn_magnitude' else 'canonical_linspace_6144',
                student_ffn_indices={str(k): v for k, v in ffn_indices.items()} or None,
                student_tensor_map=tensor_map)
    result = dict(format_version=STUDENT_FORMAT_VERSION, video_state_dict=states['video'],
                  action_state_dict=states['action'], proprio_state_dict=proprio, metadata=meta)
    validate_student_artifact(result)
    return result


@lru_cache(maxsize=1)
def _student_shapes() -> tuple[dict, dict]:
    video_cfg, action_cfg = target_configs(12)
    with torch.device('meta'):
        return _state_shapes(WanVideoDiT(**video_cfg)), _state_shapes(ActionDiT(**action_cfg))


def validate_student_artifact(artifact: Mapping) -> None:
    if artifact.get('format_version') != STUDENT_FORMAT_VERSION:
        raise ValueError('Unsupported LoopWAM student initialization artifact version.')
    meta = artifact['metadata']
    variant = meta.get('init_variant')
    if meta.get('student_block_plan') != variant_block_plan(variant):
        raise ValueError(f'Student block plan does not match variant {variant!r}')
    if meta['source_model'] != SOURCE_MODEL or len(meta.get('canonical_artifact_sha256', '')) != 64:
        raise ValueError('Student artifact must name its source model and canonical artifact hash.')
    validate_source_config(meta['source_config'])
    expected_video, expected_action = target_configs(12)
    if meta['target_video_config'] != expected_video or meta['target_action_config'] != expected_action:
        raise ValueError('Student artifact target architecture differs from LoopWAM-S.')
    video_shapes, action_shapes = _student_shapes()
    for family, shapes in [('video', video_shapes), ('action', action_shapes), ('proprio', {'weight': (4096, 8), 'bias': (4096,)})]:
        validate_state_coverage(artifact[f'{family}_state_dict'], shapes, label=f'student {family}')
    expected_map = {f'{family}.{key}' for family in ('video', 'action', 'proprio') for key in artifact[f'{family}_state_dict']}
    if set(meta['student_tensor_map']) != expected_map:
        raise ValueError('Student tensor provenance does not cover all target tensors.')
    for family in ('video', 'action', 'proprio'):
        for key, value in artifact[f'{family}_state_dict'].items():
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f'Student tensor {family}.{key} must be finite FP32')
    indices = meta.get('student_ffn_indices')
    if (variant == 'ffn_magnitude') != (indices is not None):
        raise ValueError('Only ffn_magnitude stores per-block FFN indices.')
    for slot, chosen in (indices or {}).items():
        if len(chosen) != 6144 or chosen != sorted(set(chosen)) or chosen[0] < 0 or chosen[-1] > 8959:
            raise ValueError(f'Invalid magnitude FFN indices for student block {slot}')


def load_any_init_artifact(path: str | Path) -> dict:
    artifact = torch.load(str(path), map_location='cpu', weights_only=True, mmap=True)
    if artifact.get('format_version') == STUDENT_FORMAT_VERSION:
        validate_student_artifact(artifact)
    else:
        validate_artifact(artifact)
    return artifact


def _materialize(module, state, *, device, dtype):
    # assign avoids a second full CPU model allocation before loading.
    converted = {key: value.to(device=device, dtype=dtype, copy=True) for key, value in state.items()}
    module.load_state_dict(converted, strict=True, assign=True)
    return module


def build_target_experts(artifact_or_path, *, device='cpu', dtype=torch.float32,
                         dense30=False, use_gradient_checkpointing=False):
    artifact = load_any_init_artifact(artifact_or_path) if isinstance(artifact_or_path, (str, Path)) else artifact_or_path
    student = artifact.get('format_version') == STUDENT_FORMAT_VERSION
    if student:
        validate_student_artifact(artifact)
        if dense30:
            raise ValueError('Student initialization variants are 12-layer; Dense-S30 needs the canonical artifact.')
    else:
        validate_artifact(artifact)
    video_cfg, action_cfg = target_configs(30 if dense30 else 12)
    video_cfg['use_gradient_checkpointing'] = use_gradient_checkpointing
    action_cfg['use_gradient_checkpointing'] = use_gradient_checkpointing
    with torch.device('meta'):
        video, action, proprio = WanVideoDiT(**video_cfg), ActionDiT(**action_cfg), nn.Linear(8, 4096)
    for family, module in [('video', video), ('action', action)]:
        state = artifact[f'{family}_state_dict']
        if not dense30 and not student:
            state = select_donor_layers(state)
        _materialize(module, state, device=device, dtype=dtype)
    _materialize(proprio, artifact['proprio_state_dict'], device=device, dtype=dtype)
    # RoPE tables are deliberately complex and are not checkpoint parameters.
    video.freqs = tuple(value.to(device=device) for value in precompute_freqs_cis_3d(128))
    action.freqs = precompute_freqs_cis(128).to(device=device)
    return video, action, proprio


def load_wan21_vae(path: str | Path, *, device='cpu', dtype=torch.float32):
    """Strict native 16-channel VAE with its original normalization buffers."""
    from .wan_video_vae import WanVideoVAE
    path = Path(path)
    if path.suffix == '.safetensors':
        from safetensors.torch import load_file
        raw = load_file(str(path), device='cpu')
    else:
        raw = torch.load(str(path), map_location='cpu', weights_only=True)
    if 'model_state' in raw:
        raw = raw['model_state']
    state = {key if key.startswith('model.') else f'model.{key}': value for key, value in raw.items()}
    vae = WanVideoVAE(z_dim=16)
    vae.load_state_dict(state, strict=True)
    return vae.eval().requires_grad_(False).to(device=device, dtype=dtype)
