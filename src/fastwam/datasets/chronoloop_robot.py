"""ChronoLoop on dual-arm LeRobot v2.1 data: a RoboTwin 2.0 task subset and RoboDojo.

Both datasets follow FastWAM's RoboTwin input protocol (configs/data/robotwin.yaml), as in
the RoboTwin v0 adapter (loopwam_robotwin.py on the LoopWAM_NT branch):

* cameras cam_high / cam_left_wrist / cam_right_wrist, each resized to 240x320; the high
  camera is resized to 256x320 above the two wrist cameras at 128x160 (final 384x320),
  scaled to [-1, 1];
* nine frames at offsets 0, 4, ..., 32 and 32 actions at offsets 0..31 from every anchor
  frame; offsets past the episode end repeat the last frame and are masked;
* 14-D absolute joint state/action, global z-score; the prompt is the anchor frame's task.

Windows are addressed by a global index (episode-major, every frame an anchor), which is
also the latent-cache index. Non-video fields come from in-memory parquet columns; video
comes only from the frozen-VAE latent cache (scripts/chronoloop_robot_precompute.py).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from torchvision.transforms.functional import resize

from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
from fastwam.datasets.loopwam_long import DEFAULT_PROMPT

CAMERAS = tuple('observation.images.' + k for k in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'))
VIDEO_HW = (384, 320)
LATENT_SHAPE = (16, 3, 48, 40)
OBS_TOKENS = (LATENT_SHAPE[2] // 2) * (LATENT_SHAPE[3] // 2)      # 480 tokens per latent frame
ACTION_HORIZON = 32
VIDEO_OFFSETS = tuple(range(0, 33, 4))
STATE_OFFSETS = tuple(range(33))
ROBOT_DIM = 14

# RoboTwin 2.0's 50 tasks. The FastWAM release (yuanty/robotwin2.0-fastwam) stores them in
# alphabetical blocks of 550 episodes: 50 clean demos, then 500 randomized demos (checked on
# first frames: episodes 0-49 have the plain white table, 50-549 clutter and textures).
ROBOTWIN50_TASKS = (
    'adjust_bottle', 'beat_block_hammer', 'blocks_ranking_rgb', 'blocks_ranking_size', 'click_alarmclock',
    'click_bell', 'dump_bin_bigbin', 'grab_roller', 'handover_block', 'handover_mic', 'hanging_mug', 'lift_pot',
    'move_can_pot', 'move_pillbottle_pad', 'move_playingcard_away', 'move_stapler_pad', 'open_laptop',
    'open_microwave', 'pick_diverse_bottles', 'pick_dual_bottles', 'place_a2b_left', 'place_a2b_right',
    'place_bread_basket', 'place_bread_skillet', 'place_burger_fries', 'place_can_basket',
    'place_cans_plasticbox', 'place_container_plate', 'place_dual_shoes', 'place_empty_cup', 'place_fan',
    'place_mouse_pad', 'place_object_basket', 'place_object_scale', 'place_object_stand', 'place_phone_stand',
    'place_shoe', 'press_stapler', 'put_bottles_dustbin', 'put_object_cabinet', 'rotate_qrcode', 'scan_object',
    'shake_bottle', 'shake_bottle_horizontally', 'stack_blocks_three', 'stack_blocks_two', 'stack_bowls_three',
    'stack_bowls_two', 'stamp_seal', 'turn_switch')
ROBOTWIN_BLOCK, ROBOTWIN_CLEAN = 550, 50


def robotwin_episode_ids(tasks, demos='clean+randomized'):
    """Episode indices of the given RoboTwin tasks in the FastWAM release."""
    kinds = set(demos.split('+'))
    if not kinds <= {'clean', 'randomized'} or not kinds:
        raise ValueError(f'demos must be clean, randomized or clean+randomized, got {demos}')
    out = {}
    for task in tasks:
        base = ROBOTWIN50_TASKS.index(task) * ROBOTWIN_BLOCK
        ids = []
        if 'clean' in kinds:
            ids += range(base, base + ROBOTWIN_CLEAN)
        if 'randomized' in kinds:
            ids += range(base + ROBOTWIN_CLEAN, base + ROBOTWIN_BLOCK)
        out[task] = ids
    return out


def preprocess_frames(views):
    """Per-frame FastWAM RoboTwin layout. ``views``: three uint8 [T, 3, H, W] camera tensors.

    Identical, frame by frame, to loopwam_robotwin.robotwin_video (resize acts on the last two
    dims independently per frame). Returns bf16 [T, 3, 384, 320] in [-1, 1]; bf16 is what the
    frozen VAE consumes, so storing it is lossless for the latents.
    """
    if len(views) != 3:
        raise ValueError('Expected high, left wrist and right wrist cameras')
    resized = []
    for view in views:
        if view.ndim != 4 or view.shape[1] != 3 or view.dtype != torch.uint8:
            raise ValueError(f'Expected uint8 [T, 3, H, W], got {tuple(view.shape)} {view.dtype}')
        resized.append(resize(view.float() / 255, [240, 320], antialias=True))
    high = resize(resized[0], [256, 320], antialias=True)
    wrists = torch.cat([resize(v, [128, 160], antialias=True) for v in resized[1:]], -1)
    return (torch.cat([high, wrists], -2) * 2 - 1).to(torch.bfloat16)


def preprocess_frames_cuda(views, device):
    """preprocess_frames run on a CUDA device: the same ops, but GPU float summation order, so a small
    fraction of bf16 outputs differ from the CPU chain by one rounding step. Opt-in per dataset
    (RobotSpec.preprocess); recorded in the latent-cache provenance."""
    return preprocess_frames([v.to(device, non_blocking=True) for v in views])


@dataclass
class RobotSpec:
    name: str
    root: str
    groups: dict                      # group (task) name -> list of episode indices
    text_cache_dir: str
    replan_stride: int
    stats_path: str | None = None     # published global z-score stats; None = compute from the episodes
    text_suffix: str = 't5_len128.wan22ti2v5b.pt'
    notes: dict = field(default_factory=dict)
    preprocess: str = 'cpu'           # 'cpu' (bit-exact with the v0 adapter) or 'cuda' (latent build only)

    def episode_list(self):
        return [(g, int(e)) for g, ids in self.groups.items() for e in ids]

    def identity(self):
        ident = dict(name=self.name, root=str(Path(self.root).resolve()), groups=self.groups,
                     replan_stride=self.replan_stride, stats_path=self.stats_path, text_suffix=self.text_suffix,
                     text_cache_dir=str(Path(self.text_cache_dir).resolve()))
        if self.preprocess != 'cpu':      # absent for the default so existing identities are unchanged
            ident['preprocess'] = self.preprocess
        return ident

    def sha256(self):
        return hashlib.sha256(json.dumps(self.identity(), sort_keys=True).encode()).hexdigest()


def _read_episode(root, ep, chunks_size):
    import pyarrow.parquet as pq
    t = pq.read_table(Path(root) / f'data/chunk-{ep // chunks_size:03d}/episode_{ep:06d}.parquet',
                      columns=['action', 'observation.state', 'task_index', 'episode_index', 'frame_index'])
    epi = t.column('episode_index').to_numpy()
    frames = t.column('frame_index').to_numpy()
    if not (epi == ep).all() or not (frames == np.arange(len(frames))).all():
        raise ValueError(f'Episode {ep}: rows are not one contiguous, ordered episode')
    act = np.stack(t.column('action').to_numpy(zero_copy_only=False)).astype(np.float32)
    st = np.stack(t.column('observation.state').to_numpy(zero_copy_only=False)).astype(np.float32)
    return act, st, t.column('task_index').to_numpy().astype(np.int64)


class RobotWindows:
    """Index-addressable non-video window fields (every frame of every episode is an anchor)."""

    def __init__(self, spec: RobotSpec, cache_dir=None, threads=16, text_threads=16):
        self.spec = spec
        root = Path(spec.root)
        info = json.loads((root / 'meta/info.json').read_text())
        if info['codebase_version'] != 'v2.1':
            raise ValueError('Expected a LeRobot v2.1 dataset')
        self.fps = int(info['fps'])
        eps = spec.episode_list()
        lengths = {}
        for line in (root / 'meta/episodes.jsonl').read_text().splitlines():
            row = json.loads(line)
            lengths[int(row['episode_index'])] = int(row['length'])
        packed = None if cache_dir is None else Path(cache_dir) / f'windows_{spec.sha256()[:16]}.npz'
        if packed is not None and packed.exists():
            z = np.load(packed)
            actions, states, task_idx = z['actions'], z['states'], z['task_index']
        else:
            with ThreadPoolExecutor(threads) as pool:
                parts = list(pool.map(lambda ge: _read_episode(root, ge[1], int(info['chunks_size'])), eps))
            for (g, ep), (a, _, _) in zip(eps, parts):
                if len(a) != lengths[ep]:
                    raise ValueError(f'Episode {ep}: parquet has {len(a)} rows, metadata {lengths[ep]}')
            actions = np.concatenate([p[0] for p in parts]); states = np.concatenate([p[1] for p in parts])
            task_idx = np.concatenate([p[2] for p in parts])
            if packed is not None:
                packed.parent.mkdir(parents=True, exist_ok=True)
                tmp = packed.with_suffix('.tmp.npz')
                np.savez(tmp, actions=actions, states=states, task_index=task_idx)
                tmp.replace(packed)
        if actions.shape[1:] != (ROBOT_DIM,) or states.shape[1:] != (ROBOT_DIM,):
            raise ValueError(f'Expected {ROBOT_DIM}-D action/state, got {actions.shape} {states.shape}')
        self.actions, self.states = torch.from_numpy(actions), torch.from_numpy(states)
        self.task_index = task_idx
        self.groups = list(spec.groups)
        self.episodes = []                       # (group, episode index, global start, length)
        offset = 0
        for g, ep in eps:
            self.episodes.append((g, ep, offset, lengths[ep]))
            offset += lengths[ep]
        if offset != len(actions):
            raise ValueError('Window count mismatch')
        self.size = offset
        self.ep_start = np.empty(offset, np.int64); self.ep_end = np.empty(offset, np.int64)
        self.episode_of = np.empty(offset, np.int64)
        for d, (_, _, start, length) in enumerate(self.episodes):
            self.ep_start[start:start + length] = start; self.ep_end[start:start + length] = start + length
            self.episode_of[start:start + length] = d
        self.group_of_episode = np.asarray([self.groups.index(e[0]) for e in self.episodes], np.int64)
        needed = np.unique(task_idx)
        tasks = {}
        with (root / 'meta/tasks.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                tasks[int(row['task_index'])] = row['task']
        missing = [int(t) for t in needed if int(t) not in tasks]
        if missing:
            raise ValueError(f'task_index without task text: {missing[:5]}')
        self.task_text = {int(t): tasks[int(t)] for t in needed}
        self.stats = self._stats()
        self.normalizer = {kind: SingleFieldLinearNormalizer(
            {k.removeprefix('global_'): torch.tensor(v, dtype=torch.float32)
             for k, v in self.stats[kind]['default'].items() if k.startswith('global_')}, mode='z-score')
            for kind in ('action', 'state')}
        self._text_cache = OrderedDict()
        self._text_lock = Lock()
        self._text_pool = ThreadPoolExecutor(text_threads)

    def __len__(self):
        return self.size

    # ---------------------------------------------------------------- normalization
    def _stats(self):
        if self.spec.stats_path is not None:
            return json.loads(Path(self.spec.stats_path).read_text())
        out = {}
        for kind, x in (('action', self.actions), ('state', self.states)):
            x64 = x.double()
            mean, std = x64.mean(0), x64.std(0, unbiased=False)
            out[kind] = {'default': {'global_mean': mean.tolist(), 'global_std': std.tolist(),
                                     'global_min': x64.min(0).values.tolist(), 'global_max': x64.max(0).values.tolist()}}
        out['source'] = 'per-frame global mean/std (population) over all training episodes'
        return out

    # ---------------------------------------------------------------- text
    def prompt(self, index):
        return DEFAULT_PROMPT.format(task=self.task_text[int(self.task_index[index])])

    def _load_text(self, prompt):
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        path = Path(self.spec.text_cache_dir) / f'{digest}.{self.spec.text_suffix}'
        payload = torch.load(path, map_location='cpu', weights_only=True)
        stored, mask = payload['context'], payload['mask'].bool()
        if mask.shape != (128,) or not mask.any():
            raise ValueError(f'Invalid text cache: {path}')
        if payload.get('packed_context', False):
            if stored.shape != (int(mask.sum()), 4096):
                raise ValueError(f'Invalid packed text cache: {path}')
            context = torch.zeros(128, 4096, dtype=stored.dtype)
            context[mask] = stored
        else:
            context = stored.clone()
        if context.shape != (128, 4096) or not torch.isfinite(context).all():
            raise ValueError(f'Invalid text cache: {path}')
        context[~mask] = 0
        return context, mask

    def warm_text(self, prompts, keep=8192):
        """Load the given prompts' T5 contexts in parallel threads (bounded LRU)."""
        with self._text_lock:
            todo = [p for p in dict.fromkeys(prompts) if p not in self._text_cache]
        for p, value in zip(todo, self._text_pool.map(self._load_text, todo)):
            with self._text_lock:
                self._text_cache[p] = value
        with self._text_lock:
            for p in prompts:
                if p in self._text_cache:
                    self._text_cache.move_to_end(p)
            while len(self._text_cache) > keep:
                self._text_cache.popitem(last=False)

    def text(self, prompt):
        with self._text_lock:
            hit = self._text_cache.get(prompt)
        if hit is None:
            self.warm_text([prompt])
            with self._text_lock:
                hit = self._text_cache[prompt]
        return hit

    # ---------------------------------------------------------------- fields
    def fields(self, index: int) -> dict:
        """Non-video fields of the FastWAM RoboTwin protocol for anchor ``index``."""
        start, end = int(self.ep_start[index]), int(self.ep_end[index])
        a_idx = index + np.arange(ACTION_HORIZON)
        action = self.normalizer['action'].forward(self.actions[np.clip(a_idx, start, end - 1)])
        s_idx = index + np.asarray(STATE_OFFSETS)
        proprio = self.normalizer['state'].forward(self.states[np.clip(s_idx, start, end - 1)])[:-1]
        context, mask = self.text(self.prompt(index))
        return {'action': action, 'proprio': proprio, 'action_is_pad': torch.from_numpy(a_idx >= end),
                'image_is_pad': torch.from_numpy(index + np.asarray(VIDEO_OFFSETS) >= end),
                'proprio_is_pad': torch.from_numpy(s_idx >= end)[:-1],
                'context': context, 'context_mask': mask, 'training_index': index}

    def manifest(self):
        return dict(format='chronoloop-robot-data-v1', spec=self.spec.identity(), spec_sha256=self.spec.sha256(),
                    fps=self.fps, episodes=len(self.episodes), windows=self.size, video_size=list(VIDEO_HW),
                    latent_shape=list(LATENT_SHAPE), camera_order=list(CAMERAS), video_offsets=list(VIDEO_OFFSETS),
                    action_offsets=list(range(ACTION_HORIZON)), action_dim=ROBOT_DIM, proprio_dim=ROBOT_DIM,
                    normalization=self.stats, normalization_sha256=hashlib.sha256(
                        json.dumps(self.stats, sort_keys=True).encode()).hexdigest(),
                    group_windows={g: int(sum(e[3] for e in self.episodes if e[0] == g)) for g in self.groups},
                    padding_policy='all anchors; repeat final absolute action/state/frame; mask padded losses')


def latent_provenance(windows: RobotWindows, vae_sha: str) -> dict:
    """Cache identity: data selection, preprocessing code and VAE weights (not the training code)."""
    vae_file = Path(__file__).parents[1] / 'models' / 'wan22' / 'wan_video_vae.py'
    extra = {}
    if windows.spec.preprocess != 'cpu':
        extra = dict(preprocess_device=windows.spec.preprocess, preprocess_cuda_sha256=hashlib.sha256(
            inspect.getsource(preprocess_frames_cuda).encode()).hexdigest())
    return dict(**extra,format='chronoloop-robot-latents-v1', spec_sha256=windows.spec.sha256(), windows=windows.size,
                episodes=[list(e) for e in windows.episodes], vae_sha256=vae_sha, latent_shape=list(LATENT_SHAPE),
                video_offsets=list(VIDEO_OFFSETS), video_size=list(VIDEO_HW), camera_order=list(CAMERAS),
                preprocess_sha256=hashlib.sha256(inspect.getsource(preprocess_frames).encode()).hexdigest(),
                vae_code_sha256=hashlib.sha256(vae_file.read_bytes()).hexdigest(),
                latent_precision='native_bfloat16_exact_bits', encoder_state_reset='each_clip')


# ---------------------------------------------------------------- dataset presets

# Override with CHRONO_DATA_ROOT on other clusters (HiPerGator: scripts/chrono_env_hpg.sh).
DATA_ROOT = Path(os.environ.get('CHRONO_DATA_ROOT', '/groups/yshang/an221229/data'))


def robot_spec(name: str) -> RobotSpec:
    if name.startswith('robotwin'):
        from fastwam.datasets.chronoloop_robot_tasks import ROBOTWIN_PREPROCESS, ROBOTWIN_SUBSETS
        tasks, demos = ROBOTWIN_SUBSETS[name]
        return RobotSpec(name=name, root=str(DATA_ROOT / 'robotwin2.0-fastwam/robotwin2.0'),
                         groups=robotwin_episode_ids(tasks, demos),
                         text_cache_dir=str(DATA_ROOT / 'LoopWAM_NT/text_embeds_cache/robotwin'),
                         replan_stride=24, stats_path=str(DATA_ROOT / 'robotwin2.0-fastwam/dataset_stats.json'),
                         notes=dict(replan_source='configs/sim_robotwin.yaml EVALUATION.replan_steps = 24',
                                    demos=demos), preprocess=ROBOTWIN_PREPROCESS.get(name, 'cpu'))
    if name == 'robodojo':
        root = DATA_ROOT / 'robodojo/data/RoboDojo_lerobot_v21_video'
        groups = {}
        for line in (root / 'meta/episodes.jsonl').read_text().splitlines():
            row = json.loads(line)
            if len(row['tasks']) != 1:
                raise ValueError('RoboDojo episodes are expected to have one task each')
            groups.setdefault(row['tasks'][0], []).append(int(row['episode_index']))
        from fastwam.datasets.chronoloop_robot_tasks import ROBODOJO_REPLAN_STRIDE
        return RobotSpec(name=name, root=str(root), groups=groups,
                         text_cache_dir=str(DATA_ROOT / 'LoopWAM_NT/text_embeds_cache/robodojo'),
                         replan_stride=ROBODOJO_REPLAN_STRIDE, stats_path=None)
    raise ValueError(f'Unknown robot dataset {name}')
