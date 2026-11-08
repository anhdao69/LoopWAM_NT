"""Elastic (RT) distillation inputs on RoboCasa-GR1: the v0 GR1 training windows and their frozen-VAE latents.

Every window is the one the RoboCasa-GR1 v0 trainer (scripts/train_loopwam_gr1.py at 45ee0c0 / b748d92) used, through
the same GR1Windows.fields() (RobotWindows.fields) on the same packed columns, statistics and T5 text cache: 32
absolute 29-D joint actions (the final action repeated past the episode end and masked by action_is_pad), the anchor
state as proprio, and the anchor task prompt. Only the observed (first) latent frame [16, 1, 32, 32] of each cached
clip is read; the cache (the rsync-verified NFS backup of the training cache) is opened read-only after the same
metadata identity check LoopWAMLatentCache performs (which would open it read-write).

Text: GR1 has 186 distinct prompts, so each DataLoader worker memoizes RobotWindows._load_text (the function the
trainer's warm_text/text path calls) per prompt. The thread pool behind text() is never used here, because a forked
worker cannot rely on executor threads created in its parent.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

WINDOWS = 6_020_058
EPISODES = 24_000
GROUPS = 24


class TransportGR1Dataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, latent_dir, vae_sha256):
        from .robocasa_gr1 import GR1Windows, gr1_spec, LATENT_SHAPE, latent_provenance_gr1, stats_sha256
        from .loopwam_latent_cache import LoopWAMLatentCache
        manifest_path = Path(manifest_path)
        data_dir = manifest_path.parent                     # <run>/shared/data (stats, packed windows, manifest)
        stored = json.loads(manifest_path.read_text())
        # The GR1 v0 trainer's contract digest of this manifest (make_contract: data_manifest_sha256).
        self.data_manifest_sha256 = hashlib.sha256(json.dumps(stored, sort_keys=True).encode()).hexdigest()
        stats_path = data_dir / 'dataset_stats_29d.json'
        windows = GR1Windows(gr1_spec(stats_path=stats_path), cache_dir=data_dir, text_threads=1)
        if (len(windows), len(windows.episodes), len(windows.groups)) != (WINDOWS, EPISODES, GROUPS):
            raise ValueError('Unexpected RoboCasa-GR1 window/episode/task counts')
        if windows.manifest() != stored:
            raise ValueError('RoboCasa-GR1 data manifest differs from the windows built from the dataset')
        memo = {}

        def text(prompt):                                   # per-process memo of the trainer's loader; see the docstring
            if prompt not in memo:
                memo[prompt] = windows._load_text(prompt)
            return memo[prompt]
        windows.text = text
        self.windows = windows
        self.size = len(windows)
        directory = Path(latent_dir)
        metadata = dict(format=LoopWAMLatentCache.FORMAT, size=self.size, latent_shape=list(LATENT_SHAPE),
                        storage_dtype='uint16_bfloat16_bits', output_dtype='float32',
                        provenance=latent_provenance_gr1(windows, vae_sha256))
        text_ = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
        if (directory / 'metadata.json').read_text() != text_:
            raise ValueError('GR1 latent cache provenance, shape, size or format mismatch')
        # = the trainer's contract latent_cache_provenance_sha256 (LoopWAMLatentCache.provenance_sha256)
        self.latent_cache_provenance_sha256 = hashlib.sha256(text_.encode()).hexdigest()
        if (directory / 'latents.uint16').stat().st_size != self.size * int(np.prod(LATENT_SHAPE)) * 2:
            raise ValueError('GR1 latent cache payload has the wrong size')
        valid = np.memmap(directory / 'valid.uint8', mode='r', dtype=np.uint8, shape=(self.size,))
        if not np.all(valid == 1):
            raise ValueError('GR1 latent cache must be complete')
        del valid
        self.latents = np.memmap(directory / 'latents.uint16', mode='r', dtype=np.uint16, shape=(self.size, *LATENT_SHAPE))
        if stored['normalization_sha256'] != stats_sha256(windows.stats):
            raise ValueError('GR1 statistics file differs from the data manifest')
        self.manifest = dict(stored, dataset_scope='robocasa_gr1', normalization_path=str(stats_path), train_windows=self.size)
        self.transform = SimpleNamespace(normalizer=windows.normalizer)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        valid = index >= 0; real = max(int(index), 0)
        f = self.windows.fields(real)
        bits = np.array(self.latents[real, :, :1], copy=True).view(np.int16)
        latents = torch.from_numpy(bits).view(torch.bfloat16).float()
        return dict(first_frame_latents=latents, action=f['action'], proprio=f['proprio'][:1],
                    context=f['context'], context_mask=f['context_mask'], action_is_pad=f['action_is_pad'],
                    sample_valid=valid, training_index=index)
