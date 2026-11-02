#!/usr/bin/env bash
# After the evc101 build: verify, mark the local cache staged (BUILT_HERE), copy it to the striped Lustre master, COMPLETE.
set -euo pipefail
WK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_work
PY=/groups/yshang/an221229/envs/FastWAM/bin/python
L=/tmp/loopwam_rt10/shared/latents; M=$WK/shared/latents
BUILDER=${1:?builder pid}
while kill -0 $BUILDER 2>/dev/null; do sleep 30; done
echo "$(date -Is) builder exited"; grep '"event": "done"' $WK/logs/cache_build_evc101.log | tail -1
$PY -c "
import numpy as np,sys
v=np.fromfile('$L/valid.uint8',dtype=np.uint8); n=int((v==1).sum()); print('valid',n,'of',len(v)); sys.exit(0 if n==len(v)==1163351 else 1)"
date -Is > $L/BUILT_HERE; date -Is > $L/STAGED
echo "$(date -Is) local cache staged on $(hostname)"
rm -f $M/COMPLETE
$PY $WK/pcopy.py $L/latents.uint16 $M/latents.uint16 --threads 16 --chunk-mb 64 --verify-samples 512
cat $L/valid.uint8 > $M/valid.uint8; cat $L/metadata.json > $M/metadata.json
cmp $L/valid.uint8 $M/valid.uint8; cmp $L/metadata.json $M/metadata.json
sha256sum $L/metadata.json $M/metadata.json
date -Is > $M/COMPLETE
echo "$(date -Is) master complete"
