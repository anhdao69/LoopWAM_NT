#!/usr/bin/env python3
"""sha256 + size of every file under robocasa/models/assets (relative paths, sorted) -> JSON lines."""
import hashlib, json, os, sys
root = sys.argv[1]; out = open(sys.argv[2], 'w')
for d, _, fs in sorted(os.walk(root)):
    for f in sorted(fs):
        p = os.path.join(d, f); h = hashlib.sha256()
        with open(p, 'rb') as fh:
            for c in iter(lambda: fh.read(1 << 22), b''): h.update(c)
        out.write(json.dumps([os.path.relpath(p, root), os.path.getsize(p), h.hexdigest()]) + '\n')
