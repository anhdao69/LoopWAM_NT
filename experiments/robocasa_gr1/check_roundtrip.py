#!/usr/bin/env python3
"""Check (c): normalization / slicing round trip with the real training stats and dataset rows (FastWAM venv).

1. env-style state.* groups (44-D modality slices) concatenated in the server's order == the training 29-D select
2. min/max normalize -> de-normalize is the identity on real action/state rows; training rows land in [-1, 1]
3. the server path (LoopWAMGR1Policy.predict_action, real code, model replaced by an oracle returning the
   normalized ground-truth chunk) returns action.* groups equal to the dataset's 44-D slices
4. starVLA msgpack_numpy round trip of request and response is bit-exact
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT)); sys.path.insert(0, str(CODE_ROOT / 'src'))
from fastwam.datasets.lerobot.transforms import gr1  # noqa: E402
from fastwam.datasets.robocasa_gr1 import GR1_FOLDERS, build_normalizer  # noqa: E402
from experiments.robocasa_gr1 import loopwam_policy_server as server  # noqa: E402


class Oracle:
    def __init__(self, chunk_norm):
        self.chunk = chunk_norm

    def infer_action(self, input_image, proprio, context, context_mask, action_horizon, num_inference_steps, seed, rand_device):
        assert tuple(input_image.shape) == (3, 256, 256) and input_image.dtype == torch.bfloat16
        assert float(input_image.min()) >= -1 and float(input_image.max()) <= 1
        assert tuple(proprio.shape) == (29,) and action_horizon == 32 and num_inference_steps == 10
        self.proprio = proprio
        return {'action': self.chunk.clone()}


class Text:
    cache_hits = encoded = 0

    def __call__(self, prompt):
        self.cache_hits += 1
        m = torch.zeros(128, dtype=torch.bool); m[:4] = True
        return torch.zeros(128, 4096, dtype=torch.bfloat16), m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--stats', required=True); p.add_argument('--out', required=True)
    p.add_argument('--folders', type=int, default=24); p.add_argument('--episodes', type=int, default=3)
    a = p.parse_args()
    stats = json.loads(Path(a.stats).read_text()); norm = build_normalizer(stats)
    from deployment.model_server.tools import msgpack_numpy
    rep = dict(folders=0, rows=0, max_state_concat_err=0., max_norm_roundtrip_err={'action': 0., 'state': 0.},
               normalized_range={'action': [np.inf, -np.inf], 'state': [np.inf, -np.inf]}, max_server_group_err=0.,
               max_proprio_err=0., msgpack_bit_exact=True)
    for folder in GR1_FOLDERS[:a.folders]:
        rep['folders'] += 1
        root = Path(a.root) / folder
        info = json.loads((root / 'meta/info.json').read_text())
        for ep in range(a.episodes):
            t = pq.read_table(root / info['data_path'].format(episode_chunk=0, episode_index=ep), columns=['action', 'observation.state'])
            act44 = np.stack(t.column('action').to_numpy(zero_copy_only=False)).astype(np.float64)
            st44 = np.stack(t.column('observation.state').to_numpy(zero_copy_only=False)).astype(np.float64)
            rep['rows'] += len(act44)
            # 1. env groups -> server concat == training select
            env_state = {f'state.{g}': st44[:, lo:hi] for g, (lo, hi) in gr1.SOURCE_SLICES.items()}
            cat = np.concatenate([env_state[k] for k in server.STATE_KEYS], -1)
            rep['max_state_concat_err'] = max(rep['max_state_concat_err'], float(np.abs(cat - gr1.select(st44)).max()))
            # 2. normalizer round trip on real rows
            for kind, x in (('action', gr1.select(act44)), ('state', gr1.select(st44))):
                xt = torch.from_numpy(x).float(); n = norm[kind].forward(xt)
                rep['max_norm_roundtrip_err'][kind] = max(rep['max_norm_roundtrip_err'][kind], float((norm[kind].backward(n) - xt).abs().max()))
                r = rep['normalized_range'][kind]; r[0] = min(r[0], float(n.min())); r[1] = max(r[1], float(n.max()))
            # 3. server code path with an oracle model, at a mid-episode anchor
            k = len(act44) // 2
            target44 = act44[np.minimum(np.arange(k, k + 32), len(act44) - 1)]
            oracle = Oracle(norm['action'].forward(torch.from_numpy(gr1.select(target44)).float()))
            pol = object.__new__(server.LoopWAMGR1Policy)
            pol.model, pol.normalizer, pol.text, pol.device = oracle, norm, Text(), 'cpu'
            pol.seed, pol.nfe, pol.n_action_steps, pol.video_key = 42, 10, 16, server.VIDEO_KEY
            pol.queries, pol.latency, pol.prompts = 0, [], {}
            torch.cuda.synchronize = lambda *x: None
            obs = {server.VIDEO_KEY: np.random.default_rng(ep).integers(0, 256, (1, 1, 256, 256, 3), dtype=np.uint8),
                   server.TEXT_KEY: ['unlocked_waist: test']}
            obs.update({key: v[k][None, None] for key, v in env_state.items()})
            req = msgpack_numpy.unpackb(msgpack_numpy.Packer().pack({'type': 'infer', 'payload': {'observation': obs, 'n_action_steps': 24}}))
            rep['msgpack_bit_exact'] &= all(np.array_equal(req['payload']['observation'][key], obs[key]) for key in obs if key != server.TEXT_KEY)
            out = pol.predict_action(**req['payload'])
            out = msgpack_numpy.unpackb(msgpack_numpy.Packer().pack(out))
            for g, (lo, hi) in gr1.SOURCE_SLICES.items():
                got = np.asarray(out[f'action.{g}'])
                assert got.shape == (1, 24, hi - lo), got.shape
                rep['max_server_group_err'] = max(rep['max_server_group_err'], float(np.abs(got[0] - target44[:24, lo:hi]).max()))
            want_proprio = norm['state'].forward(torch.from_numpy(gr1.select(st44[k])).float()[None])[0]
            rep['max_proprio_err'] = max(rep['max_proprio_err'], float((oracle.proprio - want_proprio).abs().max()))
    rep['passed'] = (rep['max_state_concat_err'] == 0 and max(rep['max_norm_roundtrip_err'].values()) < 1e-4
                     and rep['max_server_group_err'] < 1e-4 and rep['max_proprio_err'] < 1e-6 and rep['msgpack_bit_exact']
                     and all(-1 - 1e-5 <= r[0] and r[1] <= 1 + 1e-5 for r in rep['normalized_range'].values()))
    Path(a.out).write_text(json.dumps(rep, indent=2)); print(json.dumps(rep), flush=True)
    if not rep['passed']:
        raise SystemExit('check (c) FAILED')


if __name__ == '__main__':
    main()
