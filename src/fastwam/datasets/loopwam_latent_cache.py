"""Lazy disk cache of frozen, reset-per-clip Wan VAE latents only.

Rank zero opens with create=True, then all ranks synchronize before opening
with create=False. Every real training index must have at most one concurrent
writer and must always identify the same immutable preprocessed clip. Noise,
timesteps, policy activations, and validation indices do not belong here.

Payloads store BF16 bits in uint16 memmaps. Payload flush completes before a
uint8 validity marker is published, so an interrupted write stays a cache miss.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


class LoopWAMLatentCache:
    FORMAT = 'loopwam-frozen-clip-bfloat16-v1'

    def __init__(self, directory, size: int, provenance: dict, *, create=False,
                 device='cuda', latent_shape=(16, 3, 28, 56)):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError('Cache size must be a positive integer.')
        self.directory = Path(directory)
        self.size = size
        self.latent_shape = tuple(latent_shape)
        if not self.latent_shape or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
                                       for n in self.latent_shape):
            raise ValueError('Latent shape must contain positive integer dimensions.')
        self.device = torch.device(device)
        self._closed = False
        self._data_path = self.directory / 'latents.uint16'
        self._valid_path = self.directory / 'valid.uint8'
        metadata_path = self.directory / 'metadata.json'
        metadata = dict(format=self.FORMAT, size=size, latent_shape=list(self.latent_shape),
                        storage_dtype='uint16_bfloat16_bits', output_dtype='float32', provenance=provenance)
        metadata_text = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n'
        self.provenance_sha256 = hashlib.sha256(metadata_text.encode()).hexdigest()
        self.payload_bytes = size * int(np.prod(self.latent_shape, dtype=np.int64)) * 2
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
            if not metadata_path.exists():
                if self._data_path.exists() or self._valid_path.exists():
                    raise ValueError('Incomplete cache initialization: payload exists without metadata.')
                # Exclusive creation, sparse allocation, and zero-filled validity
                # bytes avoid allocating the full dataset in RAM at startup.
                with self._data_path.open('xb') as stream:
                    stream.truncate(self.payload_bytes)
                with self._valid_path.open('xb') as stream:
                    stream.truncate(size)
                temporary = self.directory / 'metadata.json.tmp'
                temporary.write_text(metadata_text)
                temporary.replace(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f'Cache metadata absent; rank zero must create it first: {metadata_path}')
        if metadata_path.read_text() != metadata_text:
            raise ValueError('Latent cache provenance, shape, size, or format mismatch.')
        if self._data_path.stat().st_size != self.payload_bytes or self._valid_path.stat().st_size != size:
            raise ValueError('Latent cache files have incorrect sizes.')
        self._latents = np.memmap(self._data_path, mode='r+', dtype=np.uint16,
                                  shape=(size, *self.latent_shape))
        self._valid = np.memmap(self._valid_path, mode='r+', dtype=np.uint8, shape=(size,))
        self.reset_stats()

    def reset_stats(self):
        """Reset local counters, e.g. at each epoch; sum counts across ranks."""
        self.stats = dict(hits=0, misses=0, dummy_encodes=0, encoder_samples=0, batches=0,
                          read_seconds=0., encode_seconds=0., write_seconds=0.)

    @torch.no_grad()
    def encode_batch(self, indices, video, encoder):
        """Return ordered FP32 latents, encoding only cache misses and -1 dummies.

        ``indices`` is a one-dimensional CPU integer tensor. ``video`` may
        reside on CPU or GPU; the callback receives only missing rows on that
        same device and owns any transfer needed by the frozen encoder. Its
        output must be FP32 with finite, exactly BF16-representable values.
        Timings are observed host wall times including cache transfers/flushes.
        """
        if self._closed:
            raise RuntimeError('Latent cache is closed.')
        if not isinstance(indices, torch.Tensor) or indices.device.type != 'cpu' or indices.ndim != 1:
            raise ValueError('Cache indices must be a one-dimensional CPU tensor.')
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError('Cache indices must be integer tensors.')
        if not isinstance(video, torch.Tensor) or video.ndim < 1 or video.shape[0] != len(indices):
            raise ValueError('Video batch size must match cache indices.')
        ids = indices.to(torch.int64).numpy()
        if np.any(ids < -1) or np.any(ids >= self.size):
            raise IndexError('Cache index must be -1 (dummy) or a valid training index.')
        real = ids >= 0
        hits = np.zeros(len(ids), dtype=bool)
        hits[real] = self._valid[ids[real]] == 1
        hit_positions = np.flatnonzero(hits)
        missing_positions = np.flatnonzero(~hits)
        result = torch.empty((len(ids), *self.latent_shape), dtype=torch.float32, device=self.device)
        self.stats['batches'] += 1
        if len(hit_positions):
            started = time.perf_counter()
            # Advanced indexing copies a coherent sample out of the file mapping.
            bits = np.ascontiguousarray(self._latents[ids[hit_positions]]).view(np.int16)
            values = torch.from_numpy(bits).view(torch.bfloat16).to(device=self.device, dtype=torch.float32)
            result.index_copy_(0, torch.as_tensor(hit_positions, device=self.device), values)
            self.stats['hits'] += len(hit_positions)
            self.stats['read_seconds'] += time.perf_counter() - started
        if len(missing_positions):
            started = time.perf_counter()
            rows = torch.as_tensor(missing_positions, device=video.device)
            encoded = encoder(video.index_select(0, rows))
            expected = (len(missing_positions), *self.latent_shape)
            if not isinstance(encoded, torch.Tensor) or tuple(encoded.shape) != expected:
                raise ValueError(f'Encoder must return latent shape {expected}.')
            if encoded.dtype != torch.float32:
                raise ValueError('Encoder must return FP32 casts of native BF16 latents.')
            bits_tensor = encoded.to(torch.bfloat16)
            if not torch.isfinite(encoded).all() or not torch.equal(encoded, bits_tensor.float()):
                raise ValueError('Encoder latents must be finite and losslessly representable as BF16.')
            self.stats['encode_seconds'] += time.perf_counter() - started
            self.stats['encoder_samples'] += len(missing_positions)
            dummy_count = int(np.count_nonzero(ids[missing_positions] == -1))
            self.stats['dummy_encodes'] += dummy_count
            self.stats['misses'] += len(missing_positions) - dummy_count
            result.index_copy_(0, torch.as_tensor(missing_positions, device=self.device),
                               encoded.to(device=self.device))
            store_rows = np.flatnonzero(ids[missing_positions] >= 0)
            if len(store_rows):
                started = time.perf_counter()
                cache_ids = ids[missing_positions[store_rows]]
                host_bits = bits_tensor.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
                self._latents[cache_ids] = host_bits[store_rows]
                self._latents.flush()  # Must finish before publishing any valid bit.
                self._valid[cache_ids] = 1
                self._valid.flush()
                self.stats['write_seconds'] += time.perf_counter() - started
        return result

    def close(self):
        if not self._closed:
            self._latents.flush()
            self._valid.flush()
            self._latents._mmap.close()
            self._valid._mmap.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def latent_cache_provenance(data_manifest: dict, vae_sha: str) -> dict:
    """Stable cache identity across benchmark/production output directories.

    Only the run-specific normalization_path is removed. The normalization hash,
    training-index split, source file hashes, and every other data-manifest field
    remain part of the identity. Implementation hashes bind the actual dataset
    transform and native VAE normalization/reset code as well as the weights.
    """
    if not isinstance(vae_sha, str) or not vae_sha:
        raise ValueError('The native VAE checkpoint hash is required.')
    data = json.loads(json.dumps(data_manifest, allow_nan=False))
    data.pop('normalization_path', None)
    dataset_path = Path(__file__).with_name('loopwam_long.py')
    vae_path = Path(__file__).parents[1] / 'models' / 'wan22' / 'wan_video_vae.py'
    return {
        'format': LoopWAMLatentCache.FORMAT,
        'data_manifest': data,
        'vae_sha256': vae_sha,
        'latent_shape': [16, 3, 28, 56],
        'latent_precision': 'native_bfloat16_exact_bits_returned_as_float32',
        'preprocessing': {
            'resize_hw': [224, 224], 'resize_antialias': True,
            'camera_concat': 'horizontal_in_data_manifest_camera_order',
            'rgb_range': [-1, 1], 'native_latent_normalization_applications': 1,
            'encoder_state_reset': 'each_clip',
        },
        'implementation_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (dataset_path, vae_path)},
    }
