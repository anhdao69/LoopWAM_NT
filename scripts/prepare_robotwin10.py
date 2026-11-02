#!/usr/bin/env python3
"""Write the RoboTwin-10 data manifest and check the T5 text cache before training.

python scripts/prepare_robotwin10.py --shared-dir DIR [--load-all-text]

Writes DIR/data/data_manifest.json and dataset_stats.json (or checks them if present), then checks that every
distinct anchor prompt of the 1,163,351 windows has a valid T5 cache entry (all of them are loaded with
--load-all-text; otherwise existence for all and a full load for a sample).
"""
import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from fastwam.datasets.chronoloop_robot import RobotWindows, robot_spec
from fastwam.datasets.chronoloop_robot_tasks import LILAWAM_10
from fastwam.datasets.loopwam_long import DEFAULT_PROMPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shared-dir', required=True)
    p.add_argument('--load-all-text', action='store_true')
    p.add_argument('--sample', type=int, default=2000)
    p.add_argument('--threads', type=int, default=16)
    a = p.parse_args()
    shared = Path(a.shared_dir)
    w = RobotWindows(robot_spec('robotwin10'), cache_dir=shared / 'data', text_threads=a.threads)
    m = w.manifest()
    assert len(w) == 1163351 and len(w.episodes) == 5500 and tuple(w.groups) == LILAWAM_10, (len(w), len(w.episodes))
    for name, payload in (('data_manifest.json', m), ('dataset_stats.json', w.stats)):
        path = shared / 'data' / name
        if path.exists():
            assert json.loads(path.read_text()) == payload, f'{path} differs from the current data'
        else:
            tmp = path.with_suffix('.json.tmp'); tmp.write_text(json.dumps(payload, indent=2)); tmp.replace(path)
    task_ids = np.unique(w.task_index)
    prompts = sorted({DEFAULT_PROMPT.format(task=w.task_text[int(t)]) for t in task_ids})
    first = np.unique(w.task_index, return_index=True)[1]
    assert all(w.prompt(int(i)) == DEFAULT_PROMPT.format(task=w.task_text[int(w.task_index[i])]) for i in first[:100])
    text_dir = Path(w.spec.text_cache_dir)
    paths = [text_dir / f'{hashlib.sha256(q.encode()).hexdigest()}.{w.spec.text_suffix}' for q in prompts]
    with ThreadPoolExecutor(a.threads) as pool:
        exists = list(pool.map(lambda x: x.exists(), paths))
    missing = [q for q, ok in zip(prompts, exists) if not ok]
    load = prompts if a.load_all_text else random.Random(42).sample(prompts, min(a.sample, len(prompts)))
    with ThreadPoolExecutor(a.threads) as pool:
        valid = sum(int(mask.sum()) > 0 for _, mask in pool.map(w._load_text, load))
    report = dict(windows=len(w), episodes=len(w.episodes), groups=w.groups, group_windows=m['group_windows'],
                  distinct_task_indices=len(task_ids), distinct_prompts=len(prompts), missing_text=len(missing),
                  missing_examples=missing[:5], loaded_text=len(load), loaded_valid=valid,
                  normalization_sha256=m['normalization_sha256'], spec_sha256=m['spec_sha256'],
                  data_manifest_sha256=hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest())
    print(json.dumps(report, indent=2))
    if missing or valid != len(load):
        raise SystemExit('Text cache incomplete')


if __name__ == '__main__':
    main()
