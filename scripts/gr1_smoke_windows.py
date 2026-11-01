"""Global window ids touched by the GR1 timing smoke (first N updates of epoch 0), the epoch-0 tail batch and the
reload gate window, in the trainer's randperm(seed + epoch) order. Output: .npy for gr1_precompute_latents --windows-file."""
import argparse, json
from pathlib import Path

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--shared-dir', required=True); p.add_argument('--updates', type=int, default=200)
p.add_argument('--global-batch', type=int, default=256); p.add_argument('--seed', type=int, default=42); p.add_argument('--out', required=True)
a = p.parse_args()
rep = json.loads((Path(a.shared_dir) / 'data/prepare_report.json').read_text())
n, per_epoch = rep['windows'], rep['updates_per_epoch']
perm = torch.randperm(n, generator=torch.Generator().manual_seed(a.seed)).numpy()
ids = np.unique(np.concatenate([perm[:a.updates * a.global_batch], perm[(per_epoch - 1) * a.global_batch:], np.arange(0, 64)]))
np.save(a.out, ids.astype(np.int64))
print(json.dumps(dict(windows=int(len(ids)), of=n, updates=a.updates, tail=int(n - (per_epoch - 1) * a.global_batch))))
