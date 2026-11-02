#!/usr/bin/env python3
"""Parallel byte-exact file copy (pread/pwrite ranges) with sampled verification. No torch.

python pcopy.py SRC DST [--threads 8] [--chunk-mb 64] [--verify-samples 256]
"""
import argparse, hashlib, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor

def main():
    p = argparse.ArgumentParser(); p.add_argument('src'); p.add_argument('dst')
    p.add_argument('--threads', type=int, default=8); p.add_argument('--chunk-mb', type=int, default=64)
    p.add_argument('--verify-samples', type=int, default=256)
    a = p.parse_args()
    size = os.path.getsize(a.src); chunk = a.chunk_mb << 20
    tmp = a.dst + '.partial'
    fd_out = os.open(tmp, os.O_WRONLY | os.O_CREAT, 0o644); os.ftruncate(fd_out, size)
    fd_in = os.open(a.src, os.O_RDONLY)
    offsets = list(range(0, size, chunk)); done = [0]; t0 = time.time()
    def copy(off):
        n = min(chunk, size - off); pos = 0
        while pos < n:
            buf = os.pread(fd_in, n - pos, off + pos)
            if not buf: raise IOError(f'short read at {off + pos}')
            os.pwrite(fd_out, buf, off + pos); pos += len(buf)
        done[0] += n
        if len(offsets) > 20 and (off // chunk) % max(1, len(offsets) // 20) == 0:
            el = time.time() - t0
            print(f'{done[0] / 1e9:.1f}/{size / 1e9:.1f} GB {done[0] / 1e9 / max(el, 1e-9):.2f} GB/s', flush=True)
    with ThreadPoolExecutor(a.threads) as pool:
        list(pool.map(copy, offsets))
    os.fsync(fd_out); os.close(fd_out)
    rng = random.Random(0); fd_out = os.open(tmp, os.O_RDONLY)
    for off in [0, max(0, size - chunk)] + [rng.randrange(0, max(1, size - (1 << 22))) for _ in range(a.verify_samples)]:
        n = min(1 << 22, size - off)
        if hashlib.sha256(os.pread(fd_in, n, off)).digest() != hashlib.sha256(os.pread(fd_out, n, off)).digest():
            sys.exit(f'verification mismatch at {off}')
    os.close(fd_in); os.close(fd_out); os.replace(tmp, a.dst)
    print(f'copied {size / 1e9:.1f} GB in {time.time() - t0:.0f} s, {a.verify_samples + 2} samples verified', flush=True)

if __name__ == '__main__':
    main()
