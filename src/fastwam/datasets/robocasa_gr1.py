"""RoboCasa-GR1 tabletop (24 tasks, `gr1_unified.*_1000`) windows for original LoopWAM v0.

Same window protocol as the RoboTwin v0 run (chronoloop_robot.RobotWindows, whose text and
`fields` code is reused unchanged):

* every frame of every episode is an anchor; 9 video frames at offsets 0, 4, ..., 32 and 32
  actions at offsets 0..31 (`num_frames 33`, `action_video_freq_ratio 4`); offsets past the
  episode end repeat the last frame/action/state and are masked;
* absolute joint actions (no delta mask); proprio = state at the anchor;
* the prompt is DEFAULT_PROMPT.format(task=tasks.jsonl[annotation.human.coarse_action]) of the
  anchor frame.

GR1 specifics: one camera `observation.images.ego_view`, 256x256 (no resize; scaled to [-1, 1]),
latents [16, 3, 32, 32] (256 tokens per latent frame); LeRobot v2.0 (meta/stats.json, no
frame_index column -> contiguity is checked from timestamps); 44-D vectors reduced to the 29
controllable dims by transforms/gr1.py; global min/max normalization computed over all
training frames (val_set_proportion 0), with SingleFieldLinearNormalizer's range guard for
near-constant dims.

Windows are addressed by a global index (folder-major in GR1_FOLDERS order, then episode
index), which is also the latent-cache index.
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

from fastwam.datasets.chronoloop_robot import (ACTION_HORIZON, STATE_OFFSETS, VIDEO_OFFSETS, RobotWindows)
from fastwam.datasets.lerobot.transforms import gr1 as gr1_select
from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
from fastwam.datasets.loopwam_long import DEFAULT_PROMPT

CAMERA = 'observation.images.ego_view'
VIDEO_HW = (256, 256)
LATENT_SHAPE = (16, 3, 32, 32)
OBS_TOKENS = (LATENT_SHAPE[2] // 2) * (LATENT_SHAPE[3] // 2)      # 256 tokens per latent frame
ROBOT_DIM = gr1_select.OUTPUT_DIM                                 # 29
FPS = 20
NORM_MODE = 'min/max'
TEXT_COLUMN = 'annotation.human.coarse_action'
assert (ACTION_HORIZON, VIDEO_OFFSETS, STATE_OFFSETS) == (32, tuple(range(0, 33, 4)), tuple(range(33)))

GR1_TASKS = (
    'PnPBottleToCabinetClose', 'PnPCanToDrawerClose', 'PnPCupToDrawerClose', 'PnPMilkToMicrowaveClose',
    'PnPPotatoToMicrowaveClose', 'PnPWineToCabinetClose',
    'PosttrainPnPNovelFromCuttingboardToBasketSplitA', 'PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA',
    'PosttrainPnPNovelFromCuttingboardToPanSplitA', 'PosttrainPnPNovelFromCuttingboardToPotSplitA',
    'PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA', 'PosttrainPnPNovelFromPlacematToBasketSplitA',
    'PosttrainPnPNovelFromPlacematToBowlSplitA', 'PosttrainPnPNovelFromPlacematToPlateSplitA',
    'PosttrainPnPNovelFromPlacematToTieredshelfSplitA', 'PosttrainPnPNovelFromPlateToBowlSplitA',
    'PosttrainPnPNovelFromPlateToCardboardboxSplitA', 'PosttrainPnPNovelFromPlateToPanSplitA',
    'PosttrainPnPNovelFromPlateToPlateSplitA', 'PosttrainPnPNovelFromTrayToCardboardboxSplitA',
    'PosttrainPnPNovelFromTrayToPlateSplitA', 'PosttrainPnPNovelFromTrayToPotSplitA',
    'PosttrainPnPNovelFromTrayToTieredbasketSplitA', 'PosttrainPnPNovelFromTrayToTieredshelfSplitA')
GR1_FOLDERS = tuple(f'gr1_unified.{t}_GR1ArmsAndWaistFourierHands_1000' for t in GR1_TASKS)
GR1_ENV_NAMES = tuple(f'gr1_unified/{t}_GR1ArmsAndWaistFourierHands_Env' for t in GR1_TASKS)
assert len(GR1_FOLDERS) == 24 and len(set(GR1_FOLDERS)) == 24

DATA_ROOT = Path(os.environ.get('ROBOCASA_GR1_ROOT', '/mnt/data/vmo-ai-task/anhdh35/datasets/robocasa_gr1'))
TEXT_CACHE_DIR = Path(os.environ.get(
    'ROBOCASA_GR1_TEXT_CACHE', '/mnt/data/vmo-ai-task/anhdh35/datasets/LoopWAM_NT/text_embeds_cache/robocasa_gr1'))


def preprocess_frames_gr1(frames: torch.Tensor) -> torch.Tensor:
    """uint8 [T, 3, 256, 256] -> bf16 [T, 3, 256, 256] in [-1, 1] (no resize: 256x256 is native).

    Same arithmetic as the RoboTwin chain (x.float() / 255 * 2 - 1, then bf16, which is what the
    frozen VAE consumes, so storing bf16 is lossless for the latents).
    """
    if frames.ndim != 4 or frames.shape[1] != 3 or tuple(frames.shape[-2:]) != VIDEO_HW or frames.dtype != torch.uint8:
        raise ValueError(f'Expected uint8 [T, 3, 256, 256], got {tuple(frames.shape)} {frames.dtype}')
    return (frames.float() / 255 * 2 - 1).to(torch.bfloat16)


@dataclass
class GR1Spec:
    name: str
    root: str
    folders: tuple
    text_cache_dir: str
    stats_path: str | None = None       # 29-D min/max stats JSON (scripts/gr1_prepare_data.py)
    text_suffix: str = 't5_len128.wan22ti2v5b.pt'
    notes: dict = field(default_factory=dict)

    def identity(self):
        return dict(name=self.name, root=str(Path(self.root).resolve()), folders=list(self.folders),
                    text_suffix=self.text_suffix, text_cache_dir=str(Path(self.text_cache_dir).resolve()),
                    select_indices=list(gr1_select.SELECT_INDICES), group_order=list(gr1_select.GROUP_ORDER),
                    text_column=TEXT_COLUMN, normalization_mode=NORM_MODE)

    def sha256(self):
        return hashlib.sha256(json.dumps(self.identity(), sort_keys=True).encode()).hexdigest()


def gr1_spec(stats_path=None, root=None, text_cache_dir=None) -> GR1Spec:
    return GR1Spec(name='robocasa_gr1', root=str(root or DATA_ROOT), folders=GR1_FOLDERS,
                   text_cache_dir=str(text_cache_dir or TEXT_CACHE_DIR), stats_path=None if stats_path is None else str(stats_path),
                   notes=dict(source='nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim gr1_unified *_1000',
                              mixture='starVLA fourier_gr1_unified_1000'))


def _read_folder_meta(folder: Path):
    info = json.loads((folder / 'meta/info.json').read_text())
    if not str(info['codebase_version']).startswith('v2.') or float(info['fps']) != FPS:
        raise ValueError(f'{folder.name}: unexpected version/fps {info["codebase_version"]} {info["fps"]}')
    if not (folder / 'meta/stats.json').is_file():
        raise FileNotFoundError(f'{folder.name}: missing meta/stats.json (needed by the <v2.1 loader)')
    feats = info['features']
    if tuple(feats[CAMERA]['shape']) != (*VIDEO_HW, 3) or feats['action']['shape'] != [44] \
            or feats['observation.state']['shape'] != [44]:
        raise ValueError(f'{folder.name}: unexpected feature shapes')
    for fld in ('action', 'state'):
        if gr1_select.indices_from_modality(folder / 'meta/modality.json', fld) != gr1_select.SELECT_INDICES:
            raise ValueError(f'{folder.name}: modality.json {fld} layout differs from the GR1 select indices')
    tasks = {}
    for line in (folder / 'meta/tasks.jsonl').read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tasks[int(row['task_index'])] = row['task']
    lengths = {}
    for line in (folder / 'meta/episodes.jsonl').read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            lengths[int(row['episode_index'])] = int(row['length'])
    if sorted(lengths) != list(range(int(info['total_episodes']))):
        raise ValueError(f'{folder.name}: episode indices are not 0..N-1')
    return info, tasks, lengths


def _read_episode_gr1(folder: Path, info: dict, ep: int, length: int):
    """(action44, state44 float64, text index) for one episode, validating row order/contiguity."""
    import pyarrow.parquet as pq
    path = folder / info['data_path'].format(episode_chunk=ep // int(info['chunks_size']), episode_index=ep)
    t = pq.read_table(path, columns=['action', 'observation.state', TEXT_COLUMN, 'task_index', 'episode_index',
                                     'timestamp'])
    if t.num_rows != length:
        raise ValueError(f'{path}: {t.num_rows} rows, metadata length {length}')
    if not (t.column('episode_index').to_numpy() == ep).all():
        raise ValueError(f'{path}: foreign episode_index rows')
    # LeRobot's `task` (task_index) and the GR00T coarse-action annotation must name the same string,
    # so the hydra RobotVideoDataset path and these windows get identical prompts.
    if not (t.column('task_index').to_numpy() == t.column(TEXT_COLUMN).to_numpy()).all():
        raise ValueError(f'{path}: task_index differs from {TEXT_COLUMN}')
    ts = t.column('timestamp').to_numpy()
    if not np.allclose(ts, np.arange(length) / FPS, atol=2e-3, rtol=0):
        raise ValueError(f'{path}: timestamps are not contiguous {FPS} Hz frames')
    act = np.stack(t.column('action').to_numpy(zero_copy_only=False))
    st = np.stack(t.column('observation.state').to_numpy(zero_copy_only=False))
    if act.shape != (length, 44) or st.shape != (length, 44):
        raise ValueError(f'{path}: unexpected vector shapes {act.shape} {st.shape}')
    return act, st, t.column(TEXT_COLUMN).to_numpy().astype(np.int64)


def compute_stats(actions: np.ndarray, states: np.ndarray, extra: dict | None = None) -> dict:
    """Global per-dim stats (float64, population std) over all frames, LinearNormalizer layout."""
    out = {}
    for kind, x in (('action', actions), ('state', states)):
        x64 = np.asarray(x, dtype=np.float64)
        q = np.quantile(x64, [0.01, 0.99], axis=0)
        out[kind] = {'default': dict(global_min=x64.min(0).tolist(), global_max=x64.max(0).tolist(),
                                     global_mean=x64.mean(0).tolist(), global_std=x64.std(0).tolist(),
                                     global_q01=q[0].tolist(), global_q99=q[1].tolist())}
    rng = {k: (np.asarray(out[k]['default']['global_max']) - np.asarray(out[k]['default']['global_min'])) for k in out}
    tol = SingleFieldLinearNormalizer.range_tol
    out['normalization'] = dict(
        mode=NORM_MODE, output_range=[-1.0, 1.0], clamp=[-5.0, 5.0], range_guard=tol,
        range_guard_rule='dims with max-min < range_guard map to x - min (no division)',
        guarded_dims={k: [int(i) for i in np.flatnonzero(rng[k] < tol)] for k in rng},
        smallest_range={k: dict(dim=int(rng[k].argmin()), value=float(rng[k].min())) for k in rng},
        dim_names=[f'{g}[{i}]' for g in gr1_select.GROUP_ORDER for i in range(gr1_select.GROUP_DIMS[g])])
    out['num_transition'] = int(len(actions))
    out['source'] = 'per-frame global min/max/mean/std/q01/q99 over all 24 gr1_unified *_1000 training episodes (29-D)'
    if extra:
        out.update(extra)
    return out


def build_normalizer(stats: dict) -> dict:
    return {kind: SingleFieldLinearNormalizer(
        {k.removeprefix('global_'): torch.tensor(v, dtype=torch.float32)
         for k, v in stats[kind]['default'].items() if k.startswith('global_')}, mode=NORM_MODE)
        for kind in ('action', 'state')}


def stats_sha256(stats: dict) -> str:
    return hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()


class GR1Windows(RobotWindows):
    """Index-addressable non-video GR1 window fields (every frame of every episode is an anchor).

    Reuses RobotWindows.prompt/_load_text/warm_text/text/fields (the RoboTwin v0 code path).
    """

    def __init__(self, spec: GR1Spec, cache_dir=None, threads=16, text_threads=16, require_stats=True):
        self.spec = spec
        root = Path(spec.root)
        metas = {f: _read_folder_meta(root / f) for f in spec.folders}
        self.fps = FPS
        packed = None if cache_dir is None else Path(cache_dir) / f'gr1_windows_{spec.sha256()[:16]}.npz'
        eps = [(f, ep, metas[f][2][ep]) for f in spec.folders for ep in sorted(metas[f][2])]
        # Global prompt table: distinct task strings in first-use order (folder-major).
        if packed is not None and packed.exists():
            z = np.load(packed, allow_pickle=False)
            actions, states, prompt_id = z['actions'], z['states'], z['prompt_id']
            texts = [str(s) for s in z['texts']]
            self.dropped_max_abs = dict(action=float(z['dropped_action_max']), state=float(z['dropped_state_max']))
        else:
            with ThreadPoolExecutor(threads) as pool:
                parts = list(pool.map(lambda e: _read_episode_gr1(root / e[0], metas[e[0]][0], e[1], e[2]), eps))
            texts, text_ids = [], {}
            prompt_id = np.empty(sum(e[2] for e in eps), np.int64)
            offset = 0
            for (f, ep, length), (_, _, text_idx) in zip(eps, parts):
                tasks = metas[f][1]
                for local in np.unique(text_idx):
                    if int(local) not in tasks:
                        raise ValueError(f'{f} episode {ep}: {TEXT_COLUMN}={local} not in tasks.jsonl')
                    s = tasks[int(local)]
                    if not s.strip():
                        raise ValueError(f'{f} episode {ep}: empty {TEXT_COLUMN} text (index {local})')
                    text_ids.setdefault(s, len(texts))
                    if text_ids[s] == len(texts):
                        texts.append(s)
                lut = {int(l): text_ids[tasks[int(l)]] for l in np.unique(text_idx)}
                prompt_id[offset:offset + length] = [lut[int(l)] for l in text_idx]
                offset += length
            act44 = np.concatenate([p[0] for p in parts]); st44 = np.concatenate([p[1] for p in parts])
            del parts
            self.dropped_max_abs = dict(action=float(np.abs(act44[:, gr1_select.DROPPED_INDICES]).max()),
                                        state=float(np.abs(st44[:, gr1_select.DROPPED_INDICES]).max()))
            actions = gr1_select.select(act44).astype(np.float32)
            states = gr1_select.select(st44).astype(np.float32)
            # Stats in float64 from the source values (before the float32 cast).
            self._stats64 = compute_stats(gr1_select.select(act44), gr1_select.select(st44),
                                          extra=dict(dropped_dims_max_abs=self.dropped_max_abs))
            del act44, st44
            if packed is not None:
                packed.parent.mkdir(parents=True, exist_ok=True)
                tmp = packed.with_suffix('.tmp.npz')
                np.savez(tmp, actions=actions, states=states, prompt_id=prompt_id, texts=np.asarray(texts),
                         dropped_action_max=self.dropped_max_abs['action'], dropped_state_max=self.dropped_max_abs['state'])
                tmp.replace(packed)
        if actions.shape[1:] != (ROBOT_DIM,) or states.shape[1:] != (ROBOT_DIM,):
            raise ValueError(f'Expected {ROBOT_DIM}-D action/state, got {actions.shape} {states.shape}')
        self.actions, self.states = torch.from_numpy(actions), torch.from_numpy(states)
        self.task_index = prompt_id                                # prompt id per frame (RobotWindows.prompt)
        self.task_text = dict(enumerate(texts))
        self.texts = texts
        self.groups = list(spec.folders)
        self.episodes = []                                         # (folder, episode index, global start, length)
        offset = 0
        for f, ep, length in eps:
            self.episodes.append((f, ep, offset, length))
            offset += length
        if offset != len(actions):
            raise ValueError('Window count mismatch')
        self.size = offset
        self.ep_start = np.empty(offset, np.int64); self.ep_end = np.empty(offset, np.int64)
        self.episode_of = np.empty(offset, np.int64)
        for d, (_, _, start, length) in enumerate(self.episodes):
            self.ep_start[start:start + length] = start; self.ep_end[start:start + length] = start + length
            self.episode_of[start:start + length] = d
        self.group_of_episode = np.asarray([self.groups.index(e[0]) for e in self.episodes], np.int64)
        self.info = {f: metas[f][0] for f in spec.folders}
        if spec.stats_path is not None:
            self.stats = json.loads(Path(spec.stats_path).read_text())
            self.normalizer = build_normalizer(self.stats)
        elif require_stats:
            raise ValueError('GR1Spec.stats_path is required (scripts/gr1_prepare_data.py writes it)')
        else:
            self.stats, self.normalizer = None, None
        self._text_cache = OrderedDict()
        self._text_lock = Lock()
        self._text_pool = ThreadPoolExecutor(text_threads)

    def prompts(self):
        """All distinct training prompts (DEFAULT_PROMPT applied), in prompt-id order."""
        return [DEFAULT_PROMPT.format(task=t) for t in self.texts]

    def video_path(self, d):
        f, ep, _, _ = self.episodes[d]
        info = self.info[f]
        return Path(self.spec.root) / f / info['video_path'].format(
            episode_chunk=ep // int(info['chunks_size']), video_key=CAMERA, episode_index=ep)

    def manifest(self):
        return dict(format='loopwam-gr1-data-v1', spec=self.spec.identity(), spec_sha256=self.spec.sha256(),
                    fps=self.fps, episodes=len(self.episodes), windows=self.size, video_size=list(VIDEO_HW),
                    latent_shape=list(LATENT_SHAPE), camera_order=[CAMERA], video_offsets=list(VIDEO_OFFSETS),
                    action_offsets=list(range(ACTION_HORIZON)), action_dim=ROBOT_DIM, proprio_dim=ROBOT_DIM,
                    select_indices=list(gr1_select.SELECT_INDICES), group_order=list(gr1_select.GROUP_ORDER),
                    output_slices={k: list(v) for k, v in gr1_select.OUTPUT_SLICES.items()},
                    action_type='absolute joint positions (no delta mask)',
                    normalization=self.stats, normalization_sha256=None if self.stats is None else stats_sha256(self.stats),
                    prompts=self.texts,
                    group_windows={g: int(sum(e[3] for e in self.episodes if e[0] == g)) for g in self.groups},
                    padding_policy='all anchors; repeat final absolute action/state/frame; mask padded losses')


def latent_provenance_gr1(windows: GR1Windows, vae_sha: str) -> dict:
    """Cache identity: data selection, preprocessing code and VAE weights (not the training code)."""
    vae_file = Path(__file__).parents[1] / 'models' / 'wan22' / 'wan_video_vae.py'
    return dict(format='loopwam-gr1-latents-v1', spec_sha256=windows.spec.sha256(), windows=windows.size,
                episodes=[list(e) for e in windows.episodes], vae_sha256=vae_sha, latent_shape=list(LATENT_SHAPE),
                video_offsets=list(VIDEO_OFFSETS), video_size=list(VIDEO_HW), camera_order=[CAMERA],
                decode='pyav rgb24, every frame', preprocess_sha256=hashlib.sha256(
                    inspect.getsource(preprocess_frames_gr1).encode()).hexdigest(),
                vae_code_sha256=hashlib.sha256(vae_file.read_bytes()).hexdigest(),
                latent_precision='native_bfloat16_exact_bits', encoder_state_reset='each_clip')
