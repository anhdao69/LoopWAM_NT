#!/usr/bin/env bash
# 10-update smoke of a v0 init-variant run with the production arguments (the HPG `smoke` stage, H100 layout), then
# the branch's smoke check. SKIP=0 decodes every clip instead of skipping cached ones (to compare against SKIP=1).
# usage: VARIANT=core_random GPUS=0,1 TAG=skip [NPROC=2 WORKERS=8 SKIP=1 UPDATES=10] CACHE=/tmp/.../latents bash smoke.sh
set -euo pipefail
WORK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work
OUT=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants/smoke/${VARIANT:?}_mb8_n${NPROC:-2}_w${WORKERS:-8}_${TAG:?}
source "$WORK/env.sh"
export TRITON_CACHE_DIR="/tmp/loopwam_iv_smoke_${VARIANT}_${TAG}_triton"
cd "$LOOPWAM_ROOT"
[ "$(git rev-parse HEAD)" = "$(cat "$WORK/REVISION")" ] && [ -z "$(git status --porcelain --untracked-files=all)" ] || { echo "checkout differs" >&2; exit 2; }
[ -f "${CACHE:?}/metadata.json" ] || { echo "cache not staged" >&2; exit 2; }
[ -e "$OUT" ] && { echo "exists: $OUT" >&2; exit 1; }
EXTRA=(); [ "${SKIP:-1}" = 1 ] && EXTRA=(--skip-cached-video-decode)
CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node="${NPROC:-2}" scripts/train_loopwam.py --version v0 --video-loops 4 \
  --action-loops 4 --init-artifact "checkpoints/LoopWAM/init_variants/$VARIANT.pt" --backend ddp --microbatch 8 \
  --global-batch 128 --epochs 10 --workers "${WORKERS:-8}" --seed 42 --fused-optimizer --bucket-views --structured-attention --smoke \
  --dataset-scope full_libero --dataset-dir data/lerobot_v30 --validation-samples 0 --output-dir "$OUT" \
  --latent-cache-dir "$CACHE" --save-every 434 --max-updates "${UPDATES:-10}" "${EXTRA[@]}"
rm -rf "$TRITON_CACHE_DIR"
[ "${UPDATES:-10}" = 10 ] && python "$WORK/h100_verify_variant.py" smoke --train "$OUT" \
  --reference "$WORK/reference/libero-all-suites/v0-video4-action4" --init "$VARIANT"
exit 0
