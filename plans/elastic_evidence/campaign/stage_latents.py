#!/usr/bin/env python3
"""Copy the complete frozen-VAE latent cache to node-local RAM with per-chunk SHA-256 verification.

Lustre random memmap reads cost 30-40 ms/row; node-local reads cost ~0.04 ms/row.
The copy is byte-identical: every chunk is hashed when read from Lustre and re-hashed from the copy.
"""
import hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
CHUNK = 256 << 20
report = {}
for name in ("metadata.json", "valid.uint8"):
    data = (src / name).read_bytes(); (dst / name).write_bytes(data)
    report[name] = hashlib.sha256(data).hexdigest()
size = (src / "latents.uint16").stat().st_size
with open(dst / "latents.uint16.partial", "wb") as f: f.truncate(size)
offsets = list(range(0, size, CHUNK))
def copy(off):
    with open(src / "latents.uint16", "rb", buffering=0) as fi, open(dst / "latents.uint16.partial", "r+b", buffering=0) as fo:
        fi.seek(off); data = fi.read(min(CHUNK, size - off))
        fo.seek(off); fo.write(data)
        return off, hashlib.sha256(data).hexdigest()
t = time.time()
with ThreadPoolExecutor(8) as ex: hashes = dict(ex.map(copy, offsets))
read_seconds = time.time() - t
def check(off):
    with open(dst / "latents.uint16.partial", "rb", buffering=0) as f:
        f.seek(off); return hashlib.sha256(f.read(min(CHUNK, size - off))).hexdigest() == hashes[off]
with ThreadPoolExecutor(8) as ex: ok = all(ex.map(check, offsets))
if not ok: raise SystemExit("copy verification failed")
os.replace(dst / "latents.uint16.partial", dst / "latents.uint16")
report.update(source=str(src), bytes=size, chunks=len(offsets), chunk_bytes=CHUNK, read_seconds=read_seconds,
              chunk_sha256_digest=hashlib.sha256("".join(hashes[o] for o in offsets).encode()).hexdigest(), verified=True)
(dst / "STAGED.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
