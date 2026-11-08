"""RoboCasa-GR1 (GR1ArmsAndWaistFourierHands) 44-D -> 29-D joint selection.

The LeRobot v2.0 `gr1_unified.*_1000` datasets store 44-D `observation.state` / `action`
vectors; meta/modality.json names the slices:

    left_arm 0:7, left_hand 7:13, left_leg 13:19, neck 19:22,
    right_arm 22:29, right_hand 29:35, right_leg 35:41, waist 41:44

The policy uses the 29 controllable dims, concatenated in the starVLA / GR00T order
`left_arm, right_arm, left_hand, right_hand, waist` (the order its RoboCasa client splits a
29-D chunk back into `action.left_arm` [0:7], `action.right_arm` [7:14], `action.left_hand`
[14:20], `action.right_hand` [20:26], `action.waist` [26:29]). Legs and neck are dropped (they
are identically zero in the data; the transform's backward restores them as zeros).

`GR1ArmsWaistSelect` plugs into FastWAMProcessor's `action_state_transforms` hook.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

RAW_DIM = 44
GROUP_ORDER = ('left_arm', 'right_arm', 'left_hand', 'right_hand', 'waist')
GROUP_DIMS = dict(left_arm=7, right_arm=7, left_hand=6, right_hand=6, waist=3)
# Source slices in the 44-D vector (meta/modality.json), for the kept groups ...
SOURCE_SLICES = dict(left_arm=(0, 7), right_arm=(22, 29), left_hand=(7, 13), right_hand=(29, 35), waist=(41, 44))
# ... and the dropped groups.
DROPPED_SLICES = dict(left_leg=(13, 19), neck=(19, 22), right_leg=(35, 41))
SELECT_INDICES = tuple(i for g in GROUP_ORDER for i in range(*SOURCE_SLICES[g]))
DROPPED_INDICES = tuple(i for lo, hi in DROPPED_SLICES.values() for i in range(lo, hi))
OUTPUT_DIM = len(SELECT_INDICES)
# Slices of each group in the 29-D vector.
OUTPUT_SLICES = {}
_offset = 0
for _g in GROUP_ORDER:
    OUTPUT_SLICES[_g] = (_offset, _offset + GROUP_DIMS[_g])
    _offset += GROUP_DIMS[_g]
del _offset, _g

assert OUTPUT_DIM == 29 and len(set(SELECT_INDICES)) == 29
assert sorted(SELECT_INDICES + DROPPED_INDICES) == list(range(RAW_DIM))
assert OUTPUT_SLICES == dict(left_arm=(0, 7), right_arm=(7, 14), left_hand=(14, 20), right_hand=(20, 26), waist=(26, 29))


def indices_from_modality(modality: Mapping | str | Path, field: str = 'action') -> tuple[int, ...]:
    """Derive the 29 select indices from a dataset's meta/modality.json (must equal SELECT_INDICES)."""
    if not isinstance(modality, Mapping):
        modality = json.loads(Path(modality).read_text())
    groups = modality[field]
    expected_key = 'action' if field == 'action' else 'observation.state'
    out = []
    for g in GROUP_ORDER:
        row = groups[g]
        if row.get('original_key', expected_key) != expected_key:
            raise ValueError(f'{field}.{g}: unexpected original_key {row.get("original_key")}')
        lo, hi = int(row['start']), int(row['end'])
        if hi - lo != GROUP_DIMS[g]:
            raise ValueError(f'{field}.{g}: expected {GROUP_DIMS[g]} dims, got {hi - lo}')
        out.extend(range(lo, hi))
    covered = sorted(i for row in groups.values() for i in range(int(row['start']), int(row['end'])))
    if covered != list(range(RAW_DIM)):
        raise ValueError(f'{field}: modality slices do not tile 0..{RAW_DIM - 1}')
    return tuple(out)


def select(x, indices: Iterable[int] = SELECT_INDICES):
    """[..., 44] -> [..., 29] (numpy or torch)."""
    if x.shape[-1] != RAW_DIM:
        raise ValueError(f'Expected {RAW_DIM}-D GR1 vectors, got {tuple(x.shape)}')
    idx = list(indices)
    return x[..., idx]


def expand(x, fill: float = 0.0):
    """[..., 29] -> [..., 44]; dropped legs/neck are set to `fill` (0 in the data)."""
    if x.shape[-1] != OUTPUT_DIM:
        raise ValueError(f'Expected {OUTPUT_DIM}-D GR1 vectors, got {tuple(x.shape)}')
    if isinstance(x, torch.Tensor):
        out = torch.full((*x.shape[:-1], RAW_DIM), fill, dtype=x.dtype, device=x.device)
    else:
        out = np.full((*x.shape[:-1], RAW_DIM), fill, dtype=x.dtype)
    out[..., list(SELECT_INDICES)] = x
    return out


def split_groups(x) -> dict:
    """[..., 29] -> {'left_arm': [..., 7], ...} in GROUP_ORDER."""
    if x.shape[-1] != OUTPUT_DIM:
        raise ValueError(f'Expected {OUTPUT_DIM}-D GR1 vectors, got {tuple(x.shape)}')
    return {g: x[..., lo:hi] for g, (lo, hi) in OUTPUT_SLICES.items()}


def join_groups(groups: Mapping) -> np.ndarray:
    """{'left_arm': [..., 7], ...} -> [..., 29] in GROUP_ORDER (numpy)."""
    parts = []
    for g in GROUP_ORDER:
        v = np.asarray(groups[g])
        if v.shape[-1] != GROUP_DIMS[g]:
            raise ValueError(f'{g}: expected {GROUP_DIMS[g]} dims, got {v.shape}')
        parts.append(v)
    return np.concatenate(parts, axis=-1)


class GR1ArmsWaistSelect:
    """FastWAMProcessor `action_state_transforms` entry: select the 29 controllable GR1 dims.

    forward: batch['action'][k] / batch['state'][k] [..., 44] -> [..., 29] for every key in `keys`.
    backward: [..., 29] -> [..., 44] with legs/neck = 0 (identically zero in the source data).
    """

    def __init__(self, keys: Iterable[str] = ('default',)):
        self.keys = tuple(keys)

    def forward(self, batch):
        for field in ('action', 'state'):
            if field in batch:
                for k in self.keys:
                    batch[field][k] = select(batch[field][k])
        return batch

    def backward(self, batch):
        for field in ('action', 'state'):
            if field in batch:
                for k in self.keys:
                    batch[field][k] = expand(batch[field][k])
        return batch
