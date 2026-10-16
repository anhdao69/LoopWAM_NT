#!/usr/bin/env python3
"""Frozen-VAE latent cache for ChronoLoop robot datasets (RoboTwin subset, RoboDojo).

torchrun --standalone --nproc_per_node=2 scripts/chronoloop_robot_precompute.py --dataset robotwin10 \
    --shared-dir DIR [--windows-file needed.npy]

Each needed episode's three AV1 camera streams are decoded once (PyAV) up to the last frame a
needed window uses. Only the frames those windows read are preprocessed (CPU, exactly the
RoboTwin v0 adapter's resize chain), windows are assembled with repeat-last padding, and every
window is encoded with the per-clip-reset native Wan2.1 VAE, as in LoopWAM training.
With --windows-file only the listed windows are encoded (benchmarks); otherwise all of them.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import av
import numpy as np
import torch
import torch.distributed as dist

from fastwam.datasets.chronoloop_robot import (CAMERAS, LATENT_SHAPE, VIDEO_OFFSETS, RobotWindows,
                                               latent_provenance, preprocess_frames, preprocess_frames_cuda,
                                               robot_spec)
from fastwam.datasets.loopwam_latent_cache import LoopWAMLatentCache
from fastwam.models.wan22.loopwam_init import load_wan21_vae, sha256_file


def decode_rgb(path, last_frame, threads=2):
    """uint8 [T, 3, H, W] for frames 0..last_frame (sequential decode, every frame)."""
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.codec_context.thread_count = threads
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format='rgb24'))
            if len(frames) > last_frame:
                break
    if len(frames) <= last_frame:
        raise ValueError(f'{path}: decoded {len(frames)} frames, need {last_frame + 1}')
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)


class EpisodeFrames(torch.utils.data.Dataset):
    """Item: (episode d, global start, length, anchors, frame ids, bf16 frames [F, 3, 384, 320])."""

    def __init__(self, windows: RobotWindows, work, decode_threads=2):
        self.windows, self.work, self.decode_threads = windows, work, decode_threads
        root = Path(windows.spec.root)
        info = json.loads((root / 'meta/info.json').read_text())
        self.video_path = lambda ep, key: root / info['video_path'].format(
            episode_chunk=ep // int(info['chunks_size']), video_key=key, episode_index=ep)

    def __len__(self):
        return len(self.work)

    def __getitem__(self, i):
        d, anchors = self.work[i]
        _, ep, start, length = self.windows.episodes[d]
        need = np.unique(np.minimum(anchors[:, None] + np.asarray(VIDEO_OFFSETS), length - 1))
        views = [decode_rgb(self.video_path(ep, key), int(need[-1]), self.decode_threads)[torch.from_numpy(need)]
                 for key in CAMERAS]
        if self.windows.spec.preprocess == 'cuda':
            return d, start, length, anchors, need, views          # raw uint8; resized on the GPU
        return d, start, length, anchors, need, preprocess_frames(views)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--shared-dir', required=True)
    p.add_argument('--vae-path', default='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--windows-file', default=None, help='.npy of global window indices (default: all)')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--decode-threads', type=int, default=2)
    p.add_argument('--max-episodes', type=int, default=None, help='smoke only')
    args = p.parse_args()
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)))
    torch.cuda.set_device(local)
    torch.backends.cudnn.benchmark = True
    if world > 1:
        dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    shared = Path(args.shared_dir)
    spec = robot_spec(args.dataset)
    if rank == 0:
        windows = RobotWindows(spec, cache_dir=shared / 'data')
    if world > 1: dist.barrier()
    if rank != 0:
        windows = RobotWindows(spec, cache_dir=shared / 'data')
    vae_sha = [sha256_file(args.vae_path) if rank == 0 else None]
    if world > 1: dist.broadcast_object_list(vae_sha, src=0)
    provenance = latent_provenance(windows, vae_sha[0])
    cache_dir = shared / 'latents'
    if rank == 0:
        LoopWAMLatentCache(cache_dir, len(windows), provenance, create=True, latent_shape=LATENT_SHAPE).close()
    if world > 1: dist.barrier()
    cache = LoopWAMLatentCache(cache_dir, len(windows), provenance, latent_shape=LATENT_SHAPE)
    vae = load_wan21_vae(args.vae_path, device=f'cuda:{local}', dtype=torch.bfloat16)

    def encode(video):
        latents = vae.model.encode(video.to(dtype=torch.bfloat16), vae.scale)
        vae.model.clear_cache()
        return latents.to(torch.float32)

    wanted = (np.arange(len(windows)) if args.windows_file is None
              else np.unique(np.load(args.windows_file).astype(np.int64)))
    wanted = wanted[np.asarray(cache._valid[wanted]) != 1]
    by_episode = {}
    for w in wanted:
        by_episode.setdefault(int(windows.episode_of[w]), []).append(int(w))
    work = [(d, np.asarray(ws, np.int64) - windows.episodes[d][2]) for d, ws in sorted(by_episode.items())]
    if args.max_episodes: work = work[:args.max_episodes]
    # Balance by windows: largest episodes first, greedy assignment to the least-loaded rank.
    order = sorted(range(len(work)), key=lambda i: -len(work[i][1]))
    load = [0] * world; mine = []
    for i in order:
        r = int(np.argmin(load)); load[r] += len(work[i][1])
        if r == rank: mine.append(work[i])
    loader = torch.utils.data.DataLoader(EpisodeFrames(windows, mine, args.decode_threads), batch_size=None,
                                         num_workers=args.workers, prefetch_factor=2 if args.workers else None)
    offsets = torch.tensor(VIDEO_OFFSETS)
    done, started = 0, time.perf_counter()
    for n, (d, start, length, anchors, need, frames) in enumerate(loader):
        anchors, need = np.asarray(anchors), np.asarray(need)    # the loader converts numpy to tensors
        pos = torch.full((length,), -1, dtype=torch.long)
        pos[torch.from_numpy(need)] = torch.arange(len(need))
        if windows.spec.preprocess == 'cuda':
            frames = preprocess_frames_cuda(frames, torch.device('cuda', local))
        else:
            frames = frames.cuda(non_blocking=True)
        for lo in range(0, len(anchors), args.batch):
            a = torch.from_numpy(anchors[lo:lo + args.batch])
            idx = pos[torch.clamp(a[:, None] + offsets, max=length - 1)]
            if (idx < 0).any():
                raise AssertionError('window frame was not decoded')
            video = frames[idx.cuda()].permute(0, 2, 1, 3, 4).contiguous()     # [B, 3, 9, 384, 320]
            with torch.no_grad():
                cache.encode_batch(start + a, video, encode)
            done += len(a)
        if n % 10 == 0 or n == len(mine) - 1:
            el = time.perf_counter() - started
            print(json.dumps(dict(rank=rank, episodes=n + 1, of=len(mine), windows=done, of_windows=load[rank],
                                  windows_per_s=round(done / el, 2), elapsed_s=round(el, 1))), flush=True)
    cache.close()
    if world > 1: dist.barrier()
    if rank == 0:
        valid = np.asarray(np.memmap(cache_dir / 'valid.uint8', mode='r', dtype=np.uint8))
        print(json.dumps(dict(event='done', valid=int(np.count_nonzero(valid == 1)), total=len(windows),
                              wanted_valid=int(np.count_nonzero(valid[np.unique(np.load(args.windows_file))] == 1))
                              if args.windows_file else None,
                              seconds=round(time.perf_counter() - started, 1))), flush=True)
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    main()
