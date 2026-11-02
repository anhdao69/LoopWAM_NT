#!/usr/bin/env bash
# Make /tmp/loopwam_rt10/shared/latents a complete, verified copy of the Lustre master cache (one stager per node).
set -euo pipefail
WK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_work
PY=/groups/yshang/an221229/envs/FastWAM/bin/python
M=$WK/shared/latents; L=/tmp/loopwam_rt10/shared/latents
mkdir -p "$L"
exec 9>/tmp/loopwam_rt10/stage.lock; flock 9
echo "$(date -Is) stage on $(hostname)"
if [ -f "$L/STAGED" ] && [ -f "$L/BUILT_HERE" ]; then
  echo "cache was built on this node"; exit 0
fi
if [ -f "$L/STAGED" ] && cmp -s "$L/metadata.json" "$M/metadata.json" && cmp -s "$L/valid.uint8" "$M/valid.uint8"; then
  echo "already staged"; exit 0
fi
[ -f "$M/COMPLETE" ] || { echo "master cache incomplete"; exit 1; }
need=$(stat -c %s "$M/latents.uint16"); free=$(df -B1 --output=avail /tmp | tail -1)
[ "$free" -gt $((need + 50000000000)) ] || { echo "not enough local disk: $free < $need"; exit 1; }
rm -f "$L/STAGED" "$L/metadata.json" "$L/valid.uint8"
$PY $WK/pcopy.py "$M/latents.uint16" "$L/latents.uint16" --threads 12 --chunk-mb 64 --verify-samples 256
cat "$M/valid.uint8" > "$L/valid.uint8"; cat "$M/metadata.json" > "$L/metadata.json"
$PY -c "
import numpy as np,sys
v=np.fromfile('$L/valid.uint8',dtype=np.uint8); ok=bool((v==1).all()); print('valid',int((v==1).sum()),'of',len(v)); sys.exit(0 if ok else 1)"
date -Is > "$L/STAGED"
echo "$(date -Is) staged"
