#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/h100_env.sh
for trial in ddp:2 ddp:4 ddp:8 ddp:16 zero1:8 zero1:16 zero2:8 zero2:16; do
  backend=${trial%:*}; microbatch=${trial#*:}
  name=${backend}_mb${microbatch}
  out=runs/loopwam_nt/speed_bench/$name
  mkdir -p "$out"
  torchrun --standalone --nproc_per_node=2 scripts/benchmark_loopwam.py --backend "$backend" --microbatch "$microbatch" --output-dir "$out" > "$out/launcher.log" 2>&1
  status=$?
  echo "$name exit=$status"
  echo "$status" > "$out/exit_code"
done
