#!/usr/bin/env python3
"""Populate the shared full-LIBERO frozen-VAE latent cache once, for every ChronoLoop run.

torchrun --standalone --nproc_per_node=4 scripts/chronoloop_precompute_latents.py --shared-dir DIR

Each episode's two camera streams are decoded once and resized once per frame
(the parent decodes 9 frames per window). Windows are assembled exactly as in
LoopWAMLongDataset (repeat-last-frame padding, horizontal camera concat, [-1, 1])
and encoded with the same per-clip-reset native VAE call as LoopWAM training.
Rank 0 checks assembled clips against the parent dataset bit-for-bit before writing.
Cache identity (provenance) is the parent's, so cached entries are interchangeable.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torchvision.transforms.functional import resize

from fastwam.datasets.chronoloop_data import FullLiberoWindows, VIDEO_OFFSETS
from fastwam.datasets.lerobot.lerobot.datasets.video_utils import decode_video_frames
from fastwam.datasets.loopwam_latent_cache import LoopWAMLatentCache, latent_cache_provenance
from fastwam.datasets.loopwam_long import CAMERAS, build_full_libero_datasets
from fastwam.models.wan22.loopwam_init import load_wan21_vae, sha256_file


class EpisodeFrames(torch.utils.data.Dataset):
    def __init__(self, concat, episodes):
        self.concat, self.episodes = concat, episodes
        self.readers = {suite: ds.reader for suite, ds in zip(
            ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'), concat.datasets)}

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, i):
        d, (suite, ep, start, length, _) = self.episodes[i]
        reader = self.readers[suite]
        nested = reader._episode_id_to_nested_id[ep]
        lo = int(reader.episode_data_index['from'][nested])
        ts = reader.hf_dataset.select(range(lo, lo + length))['timestamp']
        ts = torch.stack(list(ts)).tolist()
        meta = reader.meta.episodes[ep]
        views = []
        for key in CAMERAS:
            shifted = [meta[f'videos/{key}/from_timestamp'] + t for t in ts]
            frames = decode_video_frames(reader.root / reader.meta.get_video_file_path(ep, key), shifted,
                                         reader.tolerance_s, reader.video_backend)
            if frames.shape[0] != length:
                raise ValueError(f'{suite}:{ep}: decoded {frames.shape[0]} of {length} frames')
            views.append(torch.stack([resize(f, [224, 224], antialias=True) for f in frames]))
        return d, start, length, views[0], views[1]


def assemble(cam0, cam1, local_anchor, length):
    idx = torch.clamp(local_anchor[:, None] + torch.tensor(VIDEO_OFFSETS, device=cam0.device), max=length - 1)
    video = torch.cat((cam0[idx], cam1[idx]), dim=-1) * 2 - 1      # [B, 9, 3, 224, 448]
    return video.permute(0, 2, 1, 3, 4).contiguous()                # [B, 3, 9, 224, 448]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shared-dir', required=True)
    p.add_argument('--dataset-dir', default='data/lerobot_v30')
    p.add_argument('--text-cache-dir', default='data/text_embeds_cache/libero')
    p.add_argument('--vae-path', default='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--verify-windows', type=int, default=48)
    p.add_argument('--max-episodes', type=int, default=None, help='smoke only')
    args = p.parse_args()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    shared = Path(args.shared_dir)
    if rank == 0:
        train, _, manifest = build_full_libero_datasets(args.dataset_dir, args.text_cache_dir, str(shared / 'data'))
    if world > 1: dist.barrier()
    if rank != 0:
        train, _, manifest = build_full_libero_datasets(args.dataset_dir, args.text_cache_dir, str(shared / 'data'))
    windows = FullLiberoWindows(train)
    vae_sha = [sha256_file(args.vae_path) if rank == 0 else None]
    if world > 1: dist.broadcast_object_list(vae_sha, src=0)
    provenance = latent_cache_provenance(manifest, vae_sha[0])
    cache_dir = shared / 'latents'
    if rank == 0:
        LoopWAMLatentCache(cache_dir, len(train), provenance, create=True).close()
    if world > 1: dist.barrier()
    cache = LoopWAMLatentCache(cache_dir, len(train), provenance)
    vae = load_wan21_vae(args.vae_path, device=f'cuda:{local}', dtype=torch.bfloat16)

    def encode(video):
        latents = vae.model.encode(video.to(dtype=torch.bfloat16), vae.scale)
        vae.model.clear_cache()
        return latents.to(torch.float32)

    episodes = list(enumerate(windows.episodes))
    if args.max_episodes: episodes = episodes[:args.max_episodes]
    mine = [e for e in episodes[rank::world]
            if not (cache._valid[e[1][2]:e[1][2] + e[1][3]] == 1).all()]
    loader = torch.utils.data.DataLoader(EpisodeFrames(train, mine), batch_size=None, num_workers=args.workers,
                                         prefetch_factor=2 if args.workers else None)
    verified, done_windows, started = 0, 0, time.perf_counter()
    rng = np.random.default_rng(rank)
    for n, (d, start, length, cam0, cam1) in enumerate(loader):
        cam0, cam1 = cam0.cuda(non_blocking=True), cam1.cuda(non_blocking=True)
        for lo in range(0, length, args.batch):
            anchors = torch.arange(lo, min(length, lo + args.batch), device='cuda')
            video = assemble(cam0, cam1, anchors, length)
            if verified < args.verify_windows and rng.random() < 0.25:
                j = int(rng.integers(len(anchors)))
                ref = train[start + int(anchors[j])]['video']
                if not torch.equal(ref, video[j].cpu()):
                    raise AssertionError(f'Assembled clip differs from parent dataset at window {start + int(anchors[j])}')
                verified += 1
            with torch.no_grad():
                cache.encode_batch(torch.arange(start + lo, start + lo + len(anchors)), video, encode)
            done_windows += len(anchors)
        if n % 20 == 0:
            rate = done_windows / (time.perf_counter() - started)
            print(json.dumps(dict(rank=rank, episodes=n + 1, of=len(mine), windows=done_windows,
                                  windows_per_s=round(rate, 2), verified=verified)), flush=True)
    cache.close()
    if world > 1: dist.barrier()
    if rank == 0:
        valid = np.memmap(cache_dir / 'valid.uint8', mode='r', dtype=np.uint8)
        target = sum(e[1][3] for e in episodes)
        coverage = int(np.count_nonzero(valid == 1))
        print(json.dumps(dict(event='done', valid=coverage, total=len(train), target=target,
                              seconds=time.perf_counter() - started)), flush=True)
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    main()
