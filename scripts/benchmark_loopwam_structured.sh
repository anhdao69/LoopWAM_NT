#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/h100_env.sh
root=runs/loopwam_nt/speed_structured
mkdir -p "$root"
python -m pytest -q tests/test_loop_mot.py tests/test_loop_mot_structured_attention.py tests/test_loopwam_vae_cache.py tests/test_loopwam_policy.py > "$root/gpu_tests.log" 2>&1 || exit 1
torchrun --standalone --nproc_per_node=2 tests/check_loopwam_backends_distributed.py > "$root/backend_tests.log" 2>&1 || exit 1
for trial in ddp:8 ddp:16 zero1:8 zero2:8; do
  backend=${trial%:*}; microbatch=${trial#*:}; name=${backend}_mb${microbatch}
  mkdir -p "$root/$name"
  torchrun --standalone --nproc_per_node=2 scripts/benchmark_loopwam.py --backend "$backend" --microbatch "$microbatch" --structured-attention --output-dir "$root/$name" > "$root/$name/launcher.log" 2>&1
  status=$?; echo "$name exit=$status"; echo "$status" > "$root/$name/exit_code"
done
