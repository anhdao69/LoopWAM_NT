#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/vmo-ai-task/anhdh35/FastWAM
ASSETS="$ROOT/runs/loopwam_nt/robustness_assets_20261007"
cd "$ROOT/runs/loopwam_nt/source_robustness_prod_20261007"
source scripts/h100_env.sh
export PYTHONPATH="$ROOT/third_party/LIBERO-plus:$ASSETS/python_deps_offline:$PWD/src:$PWD:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="$ROOT/third_party/LIBERO-plus/.config"
export LD_LIBRARY_PATH="$ASSETS/magick/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export MAGICK_HOME="$ASSETS/magick/usr"
export MAGICK_CONFIGURE_PATH="$ASSETS/magick/etc/ImageMagick-6"
export OMP_NUM_THREADS=4
export LP_NUM_THREADS=4
python scripts/run_robustness_job.py --benchmark plus --world 2 --checkpoint "$ROOT/runs/loopwam_nt/v0_full_libero_job4659_20261006/train/latest.pt" --output "$ROOT/runs/loopwam_nt/libero_plus_job4728_after_pro_20261007"
