"""Elastic (RT) distillation inputs on RoboTwin-50: the v0 robot training windows and their frozen-VAE latents.

Every window is the one the RoboTwin v0 trainer (scripts/train_loopwam_robotwin_full.py at 8326632) used, through the
same RobotWindows.fields() on the same packed columns, statistics and text cache: 32 absolute 14-D actions (the final
action repeated past the episode end and masked by action_is_pad), the anchor state as proprio, and the anchor task
prompt. Only the observed (first) latent frame of each cached clip is read; the cache is opened read-only after the
same metadata identity check LoopWAMLatentCache performs (which would open it read-write).

Text: RoboTwin-50 has one prompt per task_index (921,032 distinct), so DataLoader workers load each context directly
with RobotWindows._load_text, the function its cached text() path calls. The thread pool behind text() is never used
here, because a forked worker cannot rely on executor threads created in its parent.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

WINDOWS = 6_075_103
EPISODES = 27_500
GROUPS = 50


class TransportRobotWinDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, latent_dir, vae_sha256):
        from .chronoloop_robot import RobotWindows, robot_spec, LATENT_SHAPE, latent_provenance
        from .loopwam_latent_cache import LoopWAMLatentCache
        manifest_path = Path(manifest_path)
        stored = json.loads(manifest_path.read_text())
        # The RoboTwin v0 trainer's contract digest of this manifest.
        self.data_manifest_sha256 = hashlib.sha256(json.dumps(stored, sort_keys=True).encode()).hexdigest()
        windows = RobotWindows(robot_spec('robotwin50'), cache_dir=manifest_path.parent, text_threads=1)
        if (len(windows), len(windows.episodes), len(windows.groups)) != (WINDOWS, EPISODES, GROUPS):
            raise ValueError('Unexpected RoboTwin-50 window/episode/task counts')
        if windows.manifest() != stored:
            raise ValueError('RoboTwin-50 data manifest differs from the windows built from the dataset')
        windows.text = windows._load_text          # direct per-call load; see the module docstring
        self.windows = windows
        self.size = len(windows)
        directory = Path(latent_dir)
        metadata = dict(format=LoopWAMLatentCache.FORMAT, size=self.size, latent_shape=list(LATENT_SHAPE),
                        storage_dtype='uint16_bfloat16_bits', output_dtype='float32',
                        provenance=latent_provenance(windows, vae_sha256))
        text = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
        if (directory / 'metadata.json').read_text() != text:
            raise ValueError('RoboTwin latent cache provenance, shape, size or format mismatch')
        if (directory / 'latents.uint16').stat().st_size != self.size * int(np.prod(LATENT_SHAPE)) * 2:
            raise ValueError('RoboTwin latent cache payload has the wrong size')
        valid = np.memmap(directory / 'valid.uint8', mode='r', dtype=np.uint8, shape=(self.size,))
        if not np.all(valid == 1):
            raise ValueError('RoboTwin latent cache must be complete')
        del valid
        self.latents = np.memmap(directory / 'latents.uint16', mode='r', dtype=np.uint16, shape=(self.size, *LATENT_SHAPE))
        self.manifest = dict(stored, dataset_scope='robotwin50', normalization_path=windows.spec.stats_path,
                             train_windows=self.size)
        # Normalization statistics the trainer copies next to its outputs: the file the RoboTwin v0 policy uses.
        stats = json.loads(Path(windows.spec.stats_path).read_text())
        if stats != stored['normalization']:
            raise ValueError('RoboTwin statistics file differs from the data manifest')
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
