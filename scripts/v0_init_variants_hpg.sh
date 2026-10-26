#!/usr/bin/env bash
# LoopWAM v0 (video 4 / action 4) on all four LIBERO suites with alternative student initializations, on HiPerGator
# (2x B200). Everything except the initialization artifact is the four-suite v0 run's contract (seed 42, global batch
# 128, ten epochs = 21,700 updates, AdamW 0.9/0.95, wd 0.01, LR 1e-4 cosine with 5% warm-up, clip 1.0, FP32 master
# weights + BF16 compute, same data manifest and normalization). Run from a pinned worktree.
#   bash scripts/v0_init_variants_hpg.sh smoke INIT MB WORKERS TAG [trainer flags]   10 updates; INIT = canonical | VARIANT
#   bash scripts/v0_init_variants_hpg.sh train VARIANT MB WORKERS                    production (decode skipped for cached
#                                                                                   clips); resumes from latest.pt if present
#   bash scripts/v0_init_variants_hpg.sh verify VARIANT                              completion + fairness vs the v0 release
# Variant artifacts come from scripts/build_init_variant.py (checkpoints/LoopWAM/init_variants/VARIANT.pt).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/chrono_env_hpg.sh
export OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
ROOT=$CHRONO_BLUE/checkpoints/LoopWAM/runs/v0_init_variants
REF=$CHRONO_BLUE/checkpoints/ChronoLoop/parent/libero-all-suites/v0-video4-action4
LAT=${V0_LATENTS:-/tmp/v0_variant_latents}                 # node-local NVMe, shared by every run in one job
CANONICAL=checkpoints/LoopWAM/wan21_compact_donors.pt
mkdir -p "$ROOT"

init_path() {  # canonical | VARIANT
  if [[ $1 == canonical ]]; then echo "$CANONICAL"; else
    local p=checkpoints/LoopWAM/init_variants/$1.pt
    [[ -f $p && -f ${p%.pt}.manifest.json ]] || { echo "missing variant artifact $p" >&2; exit 1; }
    echo "$p"; fi
}

train() {  # INIT_PATH MB WORKERS OUTPUT [extra...]
  local init=$1 mb=$2 workers=$3 out=$4; shift 4
  torchrun --standalone --nproc_per_node=2 scripts/train_loopwam.py --version v0 --video-loops 4 --action-loops 4 \
    --init-artifact "$init" --backend ddp --microbatch "$mb" --global-batch 128 --epochs 10 --workers "$workers" \
    --seed 42 --fused-optimizer --bucket-views --structured-attention --smoke --dataset-scope full_libero \
    --dataset-dir data/lerobot_v30 --validation-samples 0 --output-dir "$out" --latent-cache-dir "$LAT" \
    --save-every 2170 "$@"
}

case ${1:?stage} in
smoke)
  init=$2 mb=$3 workers=$4 tag=$5; shift 5; out=$ROOT/smoke/${init}_mb${mb}_w${workers}_$tag
  [[ -e $out ]] && { echo "exists: $out" >&2; exit 1; }
  train "$(init_path "$init")" "$mb" "$workers" "$out" --max-updates 10 "$@"
  python scripts/v0_init_variant_verify.py smoke --train "$out" --reference "$REF" --init "$init"
  ;;
train)
  variant=$2 mb=$3 workers=$4; out=$ROOT/$variant/train
  [[ $variant != canonical && $variant != default ]] || { echo "v0 default is not retrained" >&2; exit 1; }
  init=$(init_path "$variant"); mkdir -p "$ROOT/$variant"
  if [[ -f $out/latest.pt ]]; then train "$init" "$mb" "$workers" "$out" --skip-cached-video-decode --resume "$out/latest.pt"
  else train "$init" "$mb" "$workers" "$out" --skip-cached-video-decode; fi
  python scripts/v0_init_variant_verify.py run --train "$out" --reference "$REF" --init "$variant"
  ;;
verify)
  python scripts/v0_init_variant_verify.py run --train "$ROOT/$2/train" --reference "$REF" --init "$2"
  ;;
*) echo "unknown stage $1" >&2; exit 2 ;;
esac
