#!/usr/bin/env python3
"""Build the RoboCasa-GR1 window index, 29-D min/max stats and data manifest (CPU only).

python scripts/gr1_prepare_data.py --shared-dir <run>/shared [--global-batch 256 --epochs 5]

Writes <shared>/data/{gr1_windows_<spec>.npz, dataset_stats_29d.json, data_manifest.json,
prompts.json, prepare_report.json}. Checks: every folder is LeRobot v2.x at 20 fps with
meta/stats.json and the expected modality layout; timestamps contiguous; task_index equals the
coarse-action annotation; no empty prompt; dropped legs/neck exactly zero; computed per-dim
min/max agree with the union of the folders' meta/stats.json min/max on the 29 kept dims.
All unique prompt strings are logged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from fastwam.datasets.lerobot.transforms import gr1 as gr1_select
from fastwam.datasets.loopwam_robot_batches import training_budget
from fastwam.datasets.robocasa_gr1 import GR1Windows, gr1_spec, stats_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shared-dir', required=True)
    p.add_argument('--root', default=None)
    p.add_argument('--text-cache-dir', default=None)
    p.add_argument('--threads', type=int, default=32)
    p.add_argument('--global-batch', type=int, default=256)
    p.add_argument('--epochs', type=int, default=5)
    a = p.parse_args()
    data = Path(a.shared_dir) / 'data'
    data.mkdir(parents=True, exist_ok=True)
    stats_path = data / 'dataset_stats_29d.json'
    spec = gr1_spec(root=a.root, text_cache_dir=a.text_cache_dir)
    packed = data / f'gr1_windows_{spec.sha256()[:16]}.npz'
    if packed.exists() or stats_path.exists():
        raise SystemExit(f'Refusing to overwrite existing preparation outputs in {data}')
    w = GR1Windows(spec, cache_dir=data, threads=a.threads, require_stats=False)
    stats = w._stats64
    stats_path.write_text(json.dumps(stats, indent=2))

    # Cross-check against the published per-folder meta/stats.json (min/max are exact aggregates).
    src_min = {k: np.full(44, np.inf) for k in ('action', 'observation.state')}
    src_max = {k: np.full(44, -np.inf) for k in ('action', 'observation.state')}
    for f in spec.folders:
        s = json.loads((Path(spec.root) / f / 'meta/stats.json').read_text())
        for k in src_min:
            src_min[k] = np.minimum(src_min[k], np.asarray(s[k]['min'], np.float64))
            src_max[k] = np.maximum(src_max[k], np.asarray(s[k]['max'], np.float64))
    agree = {}
    for kind, key in (('action', 'action'), ('state', 'observation.state')):
        mine = stats[kind]['default']
        dmin = np.abs(np.asarray(mine['global_min']) - gr1_select.select(src_min[key])).max()
        dmax = np.abs(np.asarray(mine['global_max']) - gr1_select.select(src_max[key])).max()
        agree[kind] = dict(max_abs_min_diff=float(dmin), max_abs_max_diff=float(dmax))
        if max(dmin, dmax) > 1e-5:
            raise SystemExit(f'{kind}: computed min/max disagree with meta/stats.json ({dmin}, {dmax})')
    if max(w.dropped_max_abs.values()) != 0.0:
        raise SystemExit(f'Dropped legs/neck are not identically zero: {w.dropped_max_abs}')

    stats_check = json.loads(stats_path.read_text())     # exactly what the trainer/server will load
    w.stats = stats_check
    manifest = w.manifest()
    (data / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    per_epoch, total, tail = training_budget(len(w), a.global_batch, a.epochs)
    prompt_counts = np.bincount(w.task_index, minlength=len(w.texts))
    prompts = [dict(prompt_id=i, text=t, prompt=pr, windows=int(prompt_counts[i]))
               for i, (t, pr) in enumerate(zip(w.texts, w.prompts()))]
    (data / 'prompts.json').write_text(json.dumps(prompts, indent=2))
    rng = {k: (np.asarray(stats[k]['default']['global_max']) - np.asarray(stats[k]['default']['global_min'])).tolist()
           for k in ('action', 'state')}
    report = dict(windows=len(w), episodes=len(w.episodes), folders=len(w.groups),
                  group_windows=manifest['group_windows'], unique_prompts=len(w.texts),
                  global_batch=a.global_batch, epochs=a.epochs, updates_per_epoch=per_epoch, total_updates=total,
                  epoch_tail_windows=tail, normalization_sha256=stats_sha256(stats_check),
                  stats_file_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
                  data_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  meta_stats_agreement=agree, dropped_dims_max_abs=w.dropped_max_abs,
                  ranges=rng, guarded_dims=stats['normalization']['guarded_dims'],
                  smallest_range=stats['normalization']['smallest_range'],
                  latent_cache_gb=round(len(w) * 16 * 3 * 32 * 32 * 2 / 1e9, 1))
    (data / 'prepare_report.json').write_text(json.dumps(report, indent=2))
    for row in prompts:
        print(json.dumps(dict(event='prompt', **row)), flush=True)
    print(json.dumps({k: v for k, v in report.items() if k not in ('ranges', 'group_windows')}), flush=True)
    assert math.ceil(len(w) / a.global_batch) == per_epoch


if __name__ == '__main__':
    main()
