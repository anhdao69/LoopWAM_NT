#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/data/vmo-ai-task/anhdh35/FastWAM
SRC="$ROOT/runs/loopwam_nt/source_v0_v4a1_20261007"
OUT="$ROOT/runs/loopwam_nt/v0_v4a1_job4728_20261007"
cd "$SRC"
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=4
export LP_NUM_THREADS=4
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
python - <<'PY'
import torch
assert torch.cuda.device_count()==2
PY
python -m pytest tests/test_loopwam_asymmetric.py tests/test_loop_mot_structured_attention.py -q > "$OUT/gpu_tests.log" 2>&1
for trial in ddp_8 ddp_16 zero1_8 zero1_16 zero2_8 zero2_16; do
  backend="${trial%_*}"
  batch="${trial#*_}"
  if torchrun --standalone --nproc_per_node=2 scripts/benchmark_loopwam.py --version v0 --action-loops 1 --backend "$backend" --microbatch "$batch" --updates 7 --workers 4 --structured-attention --latent-cache-dir "$ROOT/runs/loopwam_nt/dense_s30_job4719_20261006/production_latents" --output-dir "$OUT/bench_r1_$trial" > "$OUT/bench_r1_$trial.log" 2>&1; then
    echo "$trial PASSED"
  elif grep -q 'CUDA out of memory' "$OUT/bench_r1_$trial.log"; then
    echo "$trial OOM rejected"
  else
    tail -n 60 "$OUT/bench_r1_$trial.log"
    exit 1
  fi
done
