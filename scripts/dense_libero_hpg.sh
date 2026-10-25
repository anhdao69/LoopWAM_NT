#!/usr/bin/env bash
# Dense-S12 / Dense-S30 on all four LIBERO suites on HiPerGator (2x B200), same contract as the four-suite v0 run
# (fresh canonical init, seed 42, global batch 128, ten epochs = 21,700 updates; scripts/run_full_libero_v0_job.py).
# Run from a pinned worktree (git rev-parse must work); every stage checks against the v0 four-suite release.
#   bash scripts/dense_libero_hpg.sh text                       T5 cache for the 40 prompts, compared to the reference
#   bash scripts/dense_libero_hpg.sh init                       canonical donor artifact, sha256 must match the reference
#   bash scripts/dense_libero_hpg.sh smoke VERSION MB WORKERS TAG   10 native updates (cold or warm latent cache)
#   bash scripts/dense_libero_hpg.sh train VERSION MB WORKERS   production; resumes from latest.pt if it exists
#   bash scripts/dense_libero_hpg.sh export VERSION             verify, export and upload to HF (DENSE_HF_TOKEN_FILE)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/chrono_env_hpg.sh
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
ROOT=$CHRONO_BLUE/checkpoints/LoopWAM/runs/dense_libero_all
REF=$CHRONO_BLUE/checkpoints/ChronoLoop/parent/libero-all-suites/v0-video4-action4
LAT=${DENSE_LATENTS:-/tmp/dense_libero_latents}            # node-local NVMe, shared by both models in one job
INIT=checkpoints/LoopWAM/wan21_compact_donors.pt
INIT_SHA=a3bf2097bafddac278148c1cb0321163f68cfe2b954e206436da0e3f0bd3c535
TORCHRUN=(torchrun --standalone --nproc_per_node=2)
mkdir -p "$ROOT"

train() {  # VERSION MB WORKERS OUTPUT [extra...]
  local version=$1 mb=$2 workers=$3 out=$4; shift 4
  "${TORCHRUN[@]}" scripts/train_loopwam.py --version "$version" --backend ddp --microbatch "$mb" \
    --global-batch 128 --epochs 10 --workers "$workers" --seed 42 --fused-optimizer --bucket-views \
    --structured-attention --smoke --dataset-scope full_libero --dataset-dir data/lerobot_v30 \
    --validation-samples 0 --output-dir "$out" --latent-cache-dir "$LAT" --save-every 2170 "$@"
}

case ${1:?stage} in
text)
  python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4 data=libero_2cam_lerobot_v30
  python - "$REF/data_manifest.json" data/text_embeds_cache/libero <<'E'
import hashlib, json, sys
from pathlib import Path
import torch
ref = json.loads(Path(sys.argv[1]).read_text())['text_cache_files']; d = Path(sys.argv[2])
rows = {n: dict(match=hashlib.sha256((d/n).read_bytes()).hexdigest() == r['sha256'],
                valid_tokens=int(torch.load(d/n, weights_only=True)['mask'].sum()) == r['valid_tokens'])
        for n, r in ref.items() if (d/n).exists()}
missing = sorted(set(ref) - set(rows))
print(json.dumps(dict(reference=len(ref), present=len(rows), missing=missing,
                      sha256_equal=sum(r['match'] for r in rows.values()),
                      valid_tokens_equal=sum(r['valid_tokens'] for r in rows.values()))))
if missing or not all(r['valid_tokens'] for r in rows.values()):
    sys.exit('text cache: missing prompts or different token counts')
E
  ;;
init)
  if [[ ! -f $INIT ]]; then
    python scripts/preprocess_loopwam.py --source-dir checkpoints/Wan-AI/Wan2.1-T2V-1.3B \
      --source-revision 37ec512624d61f7aa208f7ea8140a131f93afc9a --output "$INIT" --threads 12
  fi
  got=$(sha256sum "$INIT" | cut -c1-64); echo "init sha256 $got"
  [[ $got == "$INIT_SHA" ]] || { echo "init artifact differs from the reference $INIT_SHA" >&2; exit 1; }
  ;;
smoke)
  version=$2 mb=$3 workers=$4 tag=$5; out=$ROOT/smoke/${version}_mb${mb}_w${workers}_$tag
  [[ -e $out ]] && { echo "exists: $out" >&2; exit 1; }
  train "$version" "$mb" "$workers" "$out" --max-updates 10
  python - "$out" <<'E'
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); t = json.loads((d/'timing.json').read_text()); m = json.loads((d/'manifest.json').read_text())
assert t['status'] == 'smoke_complete' and t['completed_updates'] == 10 and t['windows_seen'] == 1280, t
assert m['train_windows'] == 277713 and m['planned_updates'] == 21700 and m['data']['available_episodes'] == 1712
rows = [json.loads(l) for l in (d/'metrics.jsonl').read_text().splitlines() if l.startswith('{"update"')]
print(json.dumps(dict(mean_s=t['measured_mean_update_seconds'], last5=sum(r['seconds'] for r in rows[-5:])/5,
                      peak_gb=max(r['peak_allocated_gb'] for r in rows), loss=[round(r['loss_video'] + r['loss_action'], 4) for r in rows])))
E
  ;;
train)
  version=$2 mb=$3 workers=$4; out=$ROOT/$version/train
  mkdir -p "$ROOT/$version"
  if [[ -f $out/latest.pt ]]; then train "$version" "$mb" "$workers" "$out" --resume "$out/latest.pt"
  else train "$version" "$mb" "$workers" "$out"; fi
  python scripts/dense_libero_hf_export.py verify --train "$out" --reference "$REF"
  ;;
export)
  version=$2
  python scripts/dense_libero_hf_export.py publish --train "$ROOT/$version/train" --reference "$REF" \
    --stage "$ROOT/$version/hf_stage" --token-file "${DENSE_HF_TOKEN_FILE:?}"
  ;;
*) echo "unknown stage $1" >&2; exit 2 ;;
esac
