"""ChronoLoop data: decode-free full-LIBERO windows and the shared TBPTT stream schedule.

Windows are the parent's full-LIBERO windows (global index = ConcatDataset index,
which is also the latent-cache index). Non-video fields are rebuilt from in-memory
parquet columns with the exact LoopWAMLongDataset transform; video comes only from
the frozen latent cache. ``verify_against_reference`` checks equality with the
parent dataset on sampled indices.

The traversal schedule (plan section 3.2) visits every (demo, phase u) once per
epoch; traversal (d, u) visits windows u, u+10, ... of demo d. 32 streams pop
traversals in stream-id order and advance 4 windows per update. The schedule is
world-size independent: rank r owns streams r, r+R, ...
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fastwam.datasets.loopwam_long import DEFAULT_PROMPT, FULL_LIBERO_SUITES

ACTION_HORIZON = 32
VIDEO_OFFSETS = tuple(range(0, 33, 4))
STATE_OFFSETS = tuple(range(33))
REPLAN_STRIDE = 10


class FullLiberoWindows:
    """Index-addressable non-video window fields for the ConcatDataset from
    ``build_full_libero_datasets``; ``latents`` come from a LoopWAMLatentCache."""

    def __init__(self, concat_dataset):
        self.suites = list(FULL_LIBERO_SUITES)
        datasets = concat_dataset.datasets
        if len(datasets) != len(self.suites):
            raise ValueError('Expected one dataset per LIBERO suite')
        self.reference = concat_dataset
        self.normalizer = datasets[0].normalizer
        self._text = datasets[0]._text
        actions, states, ep_start, ep_end, prompts, episode_keys = [], [], [], [], [], []
        episodes = []  # (suite, local episode id, global start, length, prompt)
        offset = 0
        for suite, ds in zip(self.suites, datasets):
            reader = ds.reader
            table = reader.hf_dataset.data.table if hasattr(reader.hf_dataset.data, 'table') else reader.hf_dataset.data
            ep_col = np.asarray(table.column('episode_index').to_numpy(), dtype=np.int64)
            act = np.asarray(table.column('action').to_pylist(), dtype=np.float32)
            st = np.asarray(table.column('observation.state').to_pylist(), dtype=np.float32)
            task_idx = np.asarray(table.column('task_index').to_numpy(), dtype=np.int64)
            if len(ep_col) != len(ds):
                raise ValueError(f'{suite}: row count mismatch')
            starts = np.empty(len(ep_col), np.int64); ends = np.empty(len(ep_col), np.int64)
            for nested, ep in enumerate(reader.episodes):
                lo = int(reader.episode_data_index['from'][nested]); hi = int(reader.episode_data_index['to'][nested])
                if not (ep_col[lo:hi] == ep).all():
                    raise ValueError(f'{suite}: episode rows are not contiguous for episode {ep}')
                starts[lo:hi] = offset + lo; ends[lo:hi] = offset + hi
                task = reader.meta.tasks.iloc[int(task_idx[lo])].name
                episodes.append((suite, int(ep), offset + lo, hi - lo, DEFAULT_PROMPT.format(task=task)))
            actions.append(act); states.append(st); ep_start.append(starts); ep_end.append(ends)
            offset += len(ep_col)
        self.actions = torch.from_numpy(np.concatenate(actions))
        self.states = torch.from_numpy(np.concatenate(states))
        self.ep_start = np.concatenate(ep_start); self.ep_end = np.concatenate(ep_end)
        self.episodes = episodes
        self.episode_of = np.empty(offset, np.int64)
        for d, (_, _, start, length, _) in enumerate(episodes):
            self.episode_of[start:start + length] = d
        self.size = offset
        self.latent_cache = None

    def __len__(self):
        return self.size

    def suite_of_episode(self, d):
        return self.episodes[d][0]

    def fields(self, index: int) -> dict:
        """Exact non-video fields of LoopWAMLongDataset.__getitem__(index)."""
        start, end = int(self.ep_start[index]), int(self.ep_end[index])
        a_idx = index + np.arange(ACTION_HORIZON)
        action_pad = torch.from_numpy(a_idx >= end)
        action = self.actions[np.clip(a_idx, start, end - 1)].clone()
        action[action_pad, :6] = 0
        action = self.normalizer['action'].forward(action)
        s_idx = index + np.asarray(STATE_OFFSETS)
        proprio = self.normalizer['state'].forward(self.states[np.clip(s_idx, start, end - 1)])[:-1]
        proprio_pad = torch.from_numpy(s_idx >= end)[:-1]
        image_pad = torch.from_numpy(index + np.asarray(VIDEO_OFFSETS) >= end)
        prompt = self.episodes[self.episode_of[index]][4]
        context, context_mask = self._text(prompt)
        return {'action': action, 'proprio': proprio, 'action_is_pad': action_pad,
                'image_is_pad': image_pad, 'proprio_is_pad': proprio_pad,
                'context': context, 'context_mask': context_mask, 'training_index': index}

    def verify_against_reference(self, indices) -> int:
        for i in indices:
            ref = self.reference[int(i)]
            mine = self.fields(int(i))
            for key in ('action', 'proprio', 'action_is_pad', 'image_is_pad', 'proprio_is_pad', 'context', 'context_mask'):
                if not torch.equal(ref[key], mine[key]):
                    raise AssertionError(f'Window {i}: field {key} differs from the parent dataset')
        return len(indices)


# --------------------------------------------------------------------------- schedule

def build_traversals(episode_lengths, epochs: int, seed: int) -> np.ndarray:
    """[(epoch, demo, phase)] for every epoch, each epoch shuffled with seed+epoch."""
    rows = []
    for epoch in range(epochs):
        pairs = np.asarray([(d, u) for d in range(len(episode_lengths)) for u in range(REPLAN_STRIDE)
                            if u < episode_lengths[d]], dtype=np.int64)
        order = np.random.default_rng(seed + epoch).permutation(len(pairs))
        rows.append(np.column_stack([np.full(len(pairs), epoch), pairs[order]]))
    return np.concatenate(rows).astype(np.int64)


def traversal_length(length: int, phase: int) -> int:
    return max(0, (length - phase + REPLAN_STRIDE - 1) // REPLAN_STRIDE)


@dataclass
class StreamState:
    traversal: int = -1      # row in the traversal list, -1 = needs a new one
    position: int = 0        # next query index within the traversal


@dataclass
class StreamScheduler:
    """Deterministic, world-size independent stream schedule (plan 3.2)."""
    traversals: np.ndarray
    episode_starts: np.ndarray
    episode_lengths: np.ndarray
    streams: int = 32
    tbptt: int = 4
    cursor: int = 0
    states: list = field(default_factory=list)
    finished_per_epoch: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.states:
            self.states = [StreamState() for _ in range(self.streams)]
        self.epoch_sizes = np.bincount(self.traversals[:, 0])

    @property
    def exhausted(self):
        return self.cursor >= len(self.traversals) and all(s.traversal < 0 for s in self.states)

    def next_update(self):
        """Return (slots, resets) for one update.

        slots[k][w] = (global window index, traversal row, query index) or None;
        resets[k] = True if stream k starts a fresh traversal (s = 0).
        """
        slots, resets = [], []
        for st in self.states:
            reset = False
            if st.traversal < 0 and self.cursor < len(self.traversals):
                st.traversal, st.position, reset = self.cursor, 0, True
                self.cursor += 1
            row = []
            if st.traversal >= 0:
                epoch, d, u = (int(v) for v in self.traversals[st.traversal])
                n = traversal_length(int(self.episode_lengths[d]), u)
                for w in range(self.tbptt):
                    q = st.position + w
                    row.append((int(self.episode_starts[d]) + u + REPLAN_STRIDE * q, st.traversal, q) if q < n else None)
                st.position += self.tbptt
                if st.position >= n:
                    self.finished_per_epoch[epoch] = self.finished_per_epoch.get(epoch, 0) + 1
                    st.traversal = -1
            else:
                row = [None] * self.tbptt
            slots.append(row); resets.append(reset)
        return slots, resets

    def epochs_complete(self) -> int:
        """Number of leading epochs whose traversals have all finished."""
        done = 0
        for epoch, size in enumerate(self.epoch_sizes):
            if self.finished_per_epoch.get(epoch, 0) == size:
                done = epoch + 1
            else:
                break
        return done

    def state_dict(self):
        return dict(cursor=self.cursor, states=[(s.traversal, s.position) for s in self.states],
                    finished_per_epoch={int(k): int(v) for k, v in self.finished_per_epoch.items()})

    def load_state_dict(self, state):
        self.cursor = int(state['cursor'])
        self.states = [StreamState(int(t), int(p)) for t, p in state['states']]
        self.finished_per_epoch = {int(k): int(v) for k, v in state['finished_per_epoch'].items()}


def simulate_schedule(traversals, episode_starts, episode_lengths, streams=32, tbptt=4):
    """Total updates, valid windows, and the update at which each epoch completes."""
    sched = StreamScheduler(traversals, episode_starts, episode_lengths, streams, tbptt)
    updates, valid, epoch_done_at = 0, 0, {}
    while not sched.exhausted:
        slots, _ = sched.next_update()
        updates += 1
        valid += sum(x is not None for row in slots for x in row)
        done = sched.epochs_complete()
        for e in range(1, done + 1):
            epoch_done_at.setdefault(e, updates)
    return dict(updates=updates, valid_windows=valid, slots=updates * streams * tbptt,
                epoch_complete_update=epoch_done_at)


def write_schedule(path, windows: FullLiberoWindows, epochs=10, seed=42, streams=32, tbptt=4):
    """Create the shared schedule file once (refuses to change an existing one)."""
    path = Path(path)
    lengths = np.asarray([e[3] for e in windows.episodes], np.int64)
    starts = np.asarray([e[2] for e in windows.episodes], np.int64)
    traversals = build_traversals(lengths, epochs, seed)
    summary = simulate_schedule(traversals, starts, lengths, streams, tbptt)
    suites = np.asarray([FULL_LIBERO_SUITES.index(e[0]) for e in windows.episodes])
    suite_windows = {s: int(lengths[suites == i].sum()) for i, s in enumerate(FULL_LIBERO_SUITES)}
    payload = dict(format='chronoloop-schedule-v1', epochs=epochs, seed=seed, streams=streams, tbptt=tbptt,
                   replan_stride=REPLAN_STRIDE, episodes=[list(e[:4]) for e in windows.episodes],
                   traversals=traversals.tolist(), summary=summary, suite_windows_per_epoch=suite_windows)
    text = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(text.encode()).hexdigest()
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Existing schedule differs: {path}')
    else:
        tmp = path.with_suffix('.tmp'); tmp.write_text(text); tmp.replace(path)
    return digest, payload


def load_schedule(path):
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    payload['sha256'] = hashlib.sha256(raw).hexdigest()
    payload['traversals'] = np.asarray(payload['traversals'], np.int64)
    eps = payload['episodes']
    payload['episode_starts'] = np.asarray([e[2] for e in eps], np.int64)
    payload['episode_lengths'] = np.asarray([e[3] for e in eps], np.int64)
    return payload


def read_latents(cache, indices, device):
    """Exact cached FP32 latents; refuses misses (the cache must be fully populated)."""
    ids = np.asarray(indices, np.int64)
    if not (cache._valid[ids] == 1).all():
        raise RuntimeError('Latent cache miss; run scripts/chronoloop_precompute_latents.py first')
    bits = np.ascontiguousarray(cache._latents[ids]).view(np.int16)
    return torch.from_numpy(bits).view(torch.bfloat16).to(device=device, non_blocking=True).float()
