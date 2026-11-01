#!/usr/bin/env bash
# GR1 v0 unattended pipeline (one srun step holding the job's 4 GPUs), after the timing smoke selected the layout:
#   1. native gates (10-update smoke, strict reload, resume to 12 + drift vs timing run, epoch-tail smoke)
#   2. full frozen-VAE latent cache on node-local /tmp (resumable: valid windows are skipped)
#   3. completeness check, then a rate-limited backup of the cache to /mnt/data (valid.uint8 copied last)
#   4. production: 5 epochs, bounded auto-resume from latest.pt
set -euo pipefail
RUN=${GR1_RUN:-/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101}
source "$RUN/env.sh"
LAT=/tmp/gr1_v0_${SLURM_JOB_ID:?run inside the slurm allocation}/latents
stage() { echo "=== $(date -u +%FT%TZ) $*"; }

stage gates
[ -f "$RUN/train/gates_passed.json" ] || python scripts/run_loopwam_gr1.py --root "$RUN/train" --shared "$RUN/shared" --latent-dir "$LAT" --phase gates

stage latents
torchrun --standalone --nproc_per_node=4 scripts/gr1_precompute_latents.py --shared-dir "$RUN/shared" --latent-dir "$LAT" --workers 6 \
  >> "$RUN/logs/latents_full.log" 2>&1
python - "$LAT" <<'EOF'
import json, sys, numpy as np
from pathlib import Path
d = Path(sys.argv[1]); meta = json.loads((d / 'metadata.json').read_text())
valid = np.memmap(d / 'valid.uint8', mode='r', dtype=np.uint8)
n = int(np.count_nonzero(valid == 1))
print(json.dumps(dict(valid=n, total=meta['size'])), flush=True)
assert n == meta['size'], 'latent cache incomplete'
EOF

stage backup
( nice -n 19 rsync -a --bwlimit=300000 "$LAT/" "$RUN/shared/latents_backup/" && echo "backup ok $(date -u +%FT%TZ)" ) \
  >> "$RUN/logs/latents_backup.log" 2>&1 &

stage production
python scripts/run_loopwam_gr1.py --root "$RUN/train" --shared "$RUN/shared" --latent-dir "$LAT" --phase production
stage complete
wait
