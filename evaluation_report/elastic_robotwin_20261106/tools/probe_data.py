"""Probe: RoboTwin-50 window/text facts and random first-frame latent read throughput from NFS."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from fastwam.datasets.chronoloop_robot import RobotWindows, robot_spec, LATENT_SHAPE
S = Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin50_20261025/shared/robotwin50')
t = time.time(); w = RobotWindows(robot_spec('robotwin50'), cache_dir=S / 'data', text_threads=4)
print(json.dumps(dict(windows=len(w), episodes=len(w.episodes), groups=len(w.groups), load_s=round(time.time() - t, 1),
                      task_indices=int(len(np.unique(w.task_index))), unique_prompts=len(set(w.task_text.values())))), flush=True)
d = Path(w.spec.text_cache_dir); files = sorted(d.glob('*.' + w.spec.text_suffix))[:5]
print(json.dumps(dict(text_dir=str(d), text_files=sum(1 for _ in d.glob('*.' + w.spec.text_suffix)), sizes=[f.stat().st_size for f in files])), flush=True)
lat = np.memmap(S / 'latents/latents.uint16', mode='r', dtype=np.uint16, shape=(len(w), *LATENT_SHAPE))
rng = np.random.default_rng(int(time.time()))
for threads in (8, 32):
    idx = rng.integers(0, len(w), 4000)
    def read(i): return np.array(lat[i, :, :1], copy=True).sum()
    t = time.time()
    with ThreadPoolExecutor(threads) as pool: list(pool.map(read, idx))
    dt = time.time() - t
    print(json.dumps(dict(threads=threads, windows=len(idx), seconds=round(dt, 2), windows_per_s=round(len(idx) / dt))), flush=True)
t = time.time(); n = 0
for i in rng.integers(0, len(w), 300):
    n += len(w._load_text(w.prompt(int(i)))[0])
print(json.dumps(dict(text_loads=300, ms_each=round((time.time() - t) / 300 * 1000, 2))), flush=True)
