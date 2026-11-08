#!/usr/bin/env python3
"""Stage the GR1 latent cache from its persistent backup (NFS) into worker-2 /tmp (RAM tmpfs), verified.

python stage_cache.py SRC_DIR DST_DIR [--threads 16]

latents.uint16 is copied in 256 MiB chunks (sha256 of each source chunk as read; every destination chunk is re-read and
compared), valid.uint8 is copied and byte-compared and must be all ones, metadata.json is written last (the trainer's
cache loader refuses a directory without it) and DST/STAGED.json is the completion marker. Idempotent: an existing
STAGED.json with matching sizes and metadata means already staged; a partial DST (no STAGED.json) is rebuilt.
"""
import argparse, hashlib, json, os, shutil, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CHUNK = 256 << 20
ALLOWED_DST = '/tmp/gr1_cont_latents_20261106/latents'


def main():
    p = argparse.ArgumentParser(); p.add_argument('src'); p.add_argument('dst'); p.add_argument('--threads', type=int, default=16)
    a = p.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    assert str(dst) == ALLOWED_DST, f'destination must be {ALLOWED_DST}'
    t0 = time.time(); meta_text = (src / 'metadata.json').read_text(); meta = json.loads(meta_text)
    size = meta['size']; row = int(np.prod(meta['latent_shape'])) * 2
    nbytes = (src / 'latents.uint16').stat().st_size
    assert nbytes == size * row and (src / 'valid.uint8').stat().st_size == size, 'backup sizes disagree with metadata'
    marker = dst / 'STAGED.json'
    if marker.exists():
        rec = json.loads(marker.read_text())
        if (rec.get('bytes') == nbytes and (dst / 'latents.uint16').stat().st_size == nbytes and
                (dst / 'metadata.json').read_text() == meta_text and (dst / 'valid.uint8').stat().st_size == size):
            print(json.dumps(dict(event='already_staged', **rec)), flush=True); return
        raise SystemExit(f'{marker} exists but does not match the backup: refusing to touch it')
    if dst.exists():
        print(json.dumps(dict(event='removing_partial', path=str(dst))), flush=True); shutil.rmtree(dst)
    free = shutil.disk_usage(dst.parent if dst.parent.exists() else '/tmp').free
    assert free > nbytes + (20 << 30), f'/tmp free {free} < needed {nbytes + (20 << 30)}'
    dst.mkdir(parents=True)
    fs = os.open(src / 'latents.uint16', os.O_RDONLY); fd = os.open(dst / 'latents.uint16', os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o664)
    os.ftruncate(fd, nbytes)
    jobs = [(o, min(CHUNK, nbytes - o)) for o in range(0, nbytes, CHUNK)]

    def copy(job):
        o, n = job; b = os.pread(fs, n, o)
        assert len(b) == n, f'short read at {o}'
        assert os.pwrite(fd, b, o) == n, f'short write at {o}'
        return o, hashlib.sha256(b).hexdigest()

    hashes = {}; t1 = time.time()
    with ThreadPoolExecutor(a.threads) as pool:
        for i, (o, h) in enumerate(pool.map(copy, jobs)):
            hashes[o] = h
            if i % 200 == 0:
                print(json.dumps(dict(event='copy', chunk=i + 1, of=len(jobs), gb_s=round((i + 1) * CHUNK / 1e9 / (time.time() - t1), 2))), flush=True)
    os.fsync(fd)

    def check(job):
        o, n = job
        return hashlib.sha256(os.pread(fd, n, o)).hexdigest() == hashes[o]
    with ThreadPoolExecutor(a.threads) as pool:
        ok = list(pool.map(check, jobs))
    assert all(ok), f'{ok.count(False)} destination chunks differ from the backup'
    os.close(fs); os.close(fd)
    shutil.copyfile(src / 'valid.uint8', dst / 'valid.uint8')
    v = np.fromfile(dst / 'valid.uint8', dtype=np.uint8)
    assert v.tobytes() == (src / 'valid.uint8').read_bytes() and int((v == 1).sum()) == size, 'valid mask incomplete or differs'
    tmp = dst / 'metadata.json.tmp'; tmp.write_text(meta_text); tmp.replace(dst / 'metadata.json')
    rec = dict(source=str(src), destination=str(dst), bytes=nbytes, windows=size, chunks=len(jobs), chunk_bytes=CHUNK,
               metadata_sha256=hashlib.sha256(meta_text.encode()).hexdigest(), chunk_sha256_verified=True, valid_all_ones=True,
               seconds=round(time.time() - t0, 1), host=os.uname().nodename, finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    marker.write_text(json.dumps(rec, indent=2))
    print(json.dumps(dict(event='done', **rec)), flush=True)


if __name__ == '__main__':
    main()
