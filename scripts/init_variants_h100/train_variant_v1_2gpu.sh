#!/usr/bin/env bash
# LoopWAM v0 (video 4 / action 4) on all four LIBERO suites with a student initialization variant, on 2 H100s:
# the `train` stage of the branch's scripts/v0_init_variants_hpg.sh (revision REVISION, unmodified worktree) with
# the same trainer arguments, except for this cluster's layout and paths:
#   * DDP on 2 GPUs, microbatch 8 x accumulation 8 = global batch 128 (the B200 runs used 16 x 4; an H100 80GB holds
#     microbatch 8), as for the other v0 runs on this cluster; 8 loader workers; resumes must keep 2 GPUs and mb 8;
#   * a checkpoint every 434 updates (the HPG stage saves once per epoch) so an interruption loses <= ~27 min;
#   * the latent cache is staged on node-local disk from the complete v0 4/2 cache (same provenance: data manifest,
#     VAE, loader and VAE code hashes); decoding is skipped for cached clips (--skip-cached-video-decode).
# Seed 42, 10 epochs = 21,700 updates, AdamW, LR schedule, precision and data are the release contract.
# Starts fresh, or resumes from latest.pt. Then runs the fairness check (h100_verify_variant.py run).
# usage: VARIANT=core_random GPUS=0,1 bash train_variant.sh
set -euo pipefail
VARIANT=${VARIANT:?set VARIANT}
WORK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work
RUN=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants/$VARIANT
TRAIN=$RUN/train
REF=$WORK/reference/libero-all-suites/v0-video4-action4
REVISION=$(cat "$WORK/REVISION")
SRC_CACHE=/groups/yshang/an221229/checkpoints/LoopWAM_NT/full_libero_v0_v4a2_20261026/latents
TAG=loopwam_iv_${VARIANT}_${SLURM_JOB_ID:-local}
CACHE=${CACHE:-/tmp/$TAG/latents}
INIT=checkpoints/LoopWAM/init_variants/$VARIANT.pt
source "$WORK/env.sh"
export TRITON_CACHE_DIR="/tmp/${TAG}_triton"
cd "$LOOPWAM_ROOT"
[ "$(git rev-parse HEAD)" = "$REVISION" ] || { echo "checkout is not $REVISION" >&2; exit 2; }
[ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "checkout is modified" >&2; exit 2; }
mkdir -p "$TRAIN" "$RUN/segments"
complete() { [ -f "$TRAIN/timing.json" ] && grep -q '"status": "complete"' "$TRAIN/timing.json"; }
verify() { python "$WORK/h100_verify_variant.py" run --train "$TRAIN" --reference "$REF" --init "$VARIANT"; }
if complete; then echo "training already complete"; verify; exit 0; fi
if [ ! -f "$TRAIN/manifest.json" ]; then
  WANT=$(python -c "import json; print(json.load(open('${INIT%.pt}.manifest.json'))['artifact_sha256'])")
  [ "$(sha256sum "$INIT" | cut -d' ' -f1)" = "$WANT" ] || { echo "$INIT hash differs from its manifest" >&2; exit 2; }
fi
# A staging left half done has no metadata.json and is redone; if local disk fails, stage on Lustre instead.
[ -d "$CACHE" ] && [ ! -f "$CACHE/metadata.json" ] && rm -rf "$CACHE"
if ! nice -n 10 python "$WORK/seed_latent_cache.py" "$SRC_CACHE" "$CACHE" | tee -a "$RUN/cache_log.jsonl"; then
  echo "local cache staging failed; staging on Lustre" >&2
  rm -rf "$CACHE"; CACHE=$RUN/latents_${SLURM_JOB_ID:-local}
  [ -d "$CACHE" ] && [ ! -f "$CACHE/metadata.json" ] && rm -rf "$CACHE"
  nice -n 10 python "$WORK/seed_latent_cache.py" "$SRC_CACHE" "$CACHE" | tee -a "$RUN/cache_log.jsonl"
fi
ARGS=(--version v0 --video-loops 4 --action-loops 4 --init-artifact "$INIT" --backend ddp --microbatch 8
      --global-batch 128 --epochs 10 --workers 8 --seed 42 --fused-optimizer --bucket-views --structured-attention
      --smoke --dataset-scope full_libero --dataset-dir data/lerobot_v30 --validation-samples 0
      --output-dir "$TRAIN" --latent-cache-dir "$CACHE" --save-every 434 --skip-cached-video-decode)
RESUME=false
if [ -f "$TRAIN/latest.pt" ]; then
  RESUME=true
  ARGS+=(--resume "$TRAIN/latest.pt")
  # Keep each segment's timing, and drop metric rows logged after the checkpoint (they are re-run).
  python - "$TRAIN" "$RUN/segments" <<'PY'
import json, shutil, sys, time, torch
from pathlib import Path
train, segments = Path(sys.argv[1]), Path(sys.argv[2])
step = torch.load(train / 'latest.pt', map_location='cpu', mmap=True, weights_only=False)['step']
stamp = time.strftime('%Y%m%dT%H%M%S')
for name in ('timing.json', 'metrics.jsonl', 'resume_manifest.json'):
    if (train / name).exists():
        shutil.copy2(train / name, segments / f'{stamp}_{name}')
lines, keep = (train / 'metrics.jsonl').read_text().splitlines(), []
for line in lines:
    row = json.loads(line)
    if 'update' in row and 'loss_video' in row and row['update'] > step:
        break
    keep.append(line)
(train / 'metrics.jsonl').write_text(''.join(l + '\n' for l in keep))
print(json.dumps(dict(event='resume_trim', checkpoint_step=step, kept=len(keep), dropped=len(lines) - len(keep))))
PY
elif [ -f "$TRAIN/metrics.jsonl" ]; then
  echo "metrics exist but no latest.pt; refusing to restart from scratch" >&2; exit 3
fi
echo "{\"event\": \"launch\", \"time\": \"$(date -Is)\", \"host\": \"$(hostname)\", \"slurm_job\": \"${SLURM_JOB_ID:-}\", \"slurm_step\": \"${SLURM_STEP_ID:-}\", \"variant\": \"$VARIANT\", \"gpus\": \"${GPUS:-0,1}\", \"resume\": $RESUME, \"revision\": \"$REVISION\", \"latent_cache\": \"$CACHE\", \"gpu_name\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPUS:-0,1}" 2>/dev/null | head -1)\"}" >> "$RUN/launches.jsonl"
CUDA_VISIBLE_DEVICES="${GPUS:-0,1}" nice -n "${NICE:-0}" torchrun --standalone --nproc_per_node=2 scripts/train_loopwam.py "${ARGS[@]}"
complete && verify
