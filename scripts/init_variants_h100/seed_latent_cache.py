#!/usr/bin/env python3
"""Seed or top up a LoopWAM latent cache from another cache with identical provenance.

Rows are published in the cache's own order: payloads written and fsynced, then the valid bytes set.
The source is only read. Its valid flags are snapshotted before any payload is read, and its writer
flushes a payload before setting the flag, so every row copied as valid was complete.
  * new destination: the payload file is copied block by block and the snapshot flags become its flags.
  * existing destination (a trainer may be using it): the source is scanned sequentially and rows that are
    invalid in the destination are copied, skipping rows whose epoch-1 position is below
    --min-update * global_batch, so a trainer that is at most at that update can never be writing them.
Lustre random reads are ~10-40 ms per row, so both paths read the source sequentially.
    python seed_latent_cache.py SRC DST [--min-update U]
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch

ROW_BYTES = 16 * 3 * 28 * 56 * 2
BLOCK_ROWS = (64 << 20) // ROW_BYTES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('dst', type=Path)
    ap.add_argument('--min-update', type=int, default=0)
    ap.add_argument('--global-batch', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    meta = (args.src / 'metadata.json').read_text()
    n = json.loads(meta)['size']
    snapshot = np.fromfile(args.src / 'valid.uint8', dtype=np.uint8)
    assert snapshot.shape == (n,)
    started = time.time()
    if not (args.dst / 'metadata.json').exists():
        args.dst.mkdir(parents=True, exist_ok=True)
        if any(args.dst.iterdir()):
            raise SystemExit('Destination has files but no metadata; refusing to seed')
        # Rows still being written are copied too but stay invalid, so the trainer re-encodes them.
        size = n * ROW_BYTES
        with (args.src / 'latents.uint16').open('rb') as src, (args.dst / 'latents.uint16').open('xb') as dst:
            dst.truncate(size)
            for offset in range(0, size, BLOCK_ROWS * ROW_BYTES):
                block = src.read(min(BLOCK_ROWS * ROW_BYTES, size - offset))
                if block.count(0) != len(block):
                    dst.seek(offset)
                    dst.write(block)
            dst.flush()
            os.fsync(dst.fileno())
        snapshot.tofile(args.dst / 'valid.uint8.tmp')
        (args.dst / 'valid.uint8.tmp').replace(args.dst / 'valid.uint8')
        (args.dst / 'metadata.json.tmp').write_text(meta)
        (args.dst / 'metadata.json.tmp').replace(args.dst / 'metadata.json')
        copied = int(snapshot.sum())
    else:
        if (args.dst / 'metadata.json').read_text() != meta:
            raise SystemExit('Destination provenance differs from the source cache')
        order = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).numpy()
        position = np.empty(n, dtype=np.int64)
        position[order] = np.arange(n)
        dst_valid = np.memmap(args.dst / 'valid.uint8', mode='r+', dtype=np.uint8, shape=(n,))
        wanted = (snapshot == 1) & (np.array(dst_valid) == 0) & (position >= args.min_update * args.global_batch)
        rows = np.flatnonzero(wanted)
        with (args.src / 'latents.uint16').open('rb') as src, (args.dst / 'latents.uint16').open('r+b') as dst:
            for first in range(0, n, BLOCK_ROWS):
                block_rows = rows[(rows >= first) & (rows < first + BLOCK_ROWS)]
                if not len(block_rows):
                    continue
                src.seek(first * ROW_BYTES)
                block = src.read(min(BLOCK_ROWS, n - first) * ROW_BYTES)
                for row in block_rows:
                    local = (int(row) - first) * ROW_BYTES
                    dst.seek(int(row) * ROW_BYTES)
                    dst.write(block[local:local + ROW_BYTES])
            dst.flush()
            os.fsync(dst.fileno())
        dst_valid[rows] = 1
        dst_valid.flush()
        copied = int(len(rows))
    valid = np.fromfile(args.dst / 'valid.uint8', dtype=np.uint8)
    print(json.dumps(dict(event='seed_latent_cache', time=time.strftime('%Y-%m-%dT%H:%M:%S'), dst=str(args.dst),
                          copied=copied, src_valid=int(snapshot.sum()), dst_valid=int(valid.sum()),
                          min_update=args.min_update, seconds=round(time.time() - started, 1))), flush=True)


if __name__ == '__main__':
    main()
