#!/usr/bin/env bash
# Download Robocasa-GR1/epoch_0N.pt (+ dataset_stats.json, receipt) from HF anhdao69/LoopWAM_NT and verify sha256
# against the upload receipt written on the H100 side. Usage: ws_fetch_ckpt.sh N
set -euo pipefail
N=$(printf %02d "$1"); ROOT=/storage/anhdh35/robocasa_gr1_eval; D=$ROOT/models/Robocasa-GR1
PY=/storage/anhdh35/miniconda3/envs/loopwam-libero-eval/bin/python
mkdir -p "$D"
env -u HF_HUB_OFFLINE HF_HUB_ENABLE_HF_TRANSFER=0 "$PY" - "$N" "$ROOT/models" <<'PY'
import json, sys, hashlib
from huggingface_hub import hf_hub_download
n, root = sys.argv[1], sys.argv[2]
for f in (f'Robocasa-GR1/epoch_{n}.pt.uploaded.json', 'Robocasa-GR1/dataset_stats.json', 'Robocasa-GR1/manifest.json',
          f'Robocasa-GR1/epoch_{n}.pt'):
    p = hf_hub_download('anhdao69/LoopWAM_NT', f, local_dir=root)
    print('downloaded', p, flush=True)
r = json.load(open(f'{root}/Robocasa-GR1/epoch_{n}.pt.uploaded.json'))
h = hashlib.sha256()
with open(f'{root}/Robocasa-GR1/epoch_{n}.pt', 'rb') as fh:
    for c in iter(lambda: fh.read(1 << 24), b''): h.update(c)
ok = h.hexdigest() == r['sha256']
print(json.dumps(dict(file=f'epoch_{n}.pt', sha256=h.hexdigest(), receipt_sha256=r['sha256'], match=ok, step=r['step'])))
sys.exit(0 if ok else 1)
PY
