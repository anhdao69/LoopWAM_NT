#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/h100_env.sh
root=${1:-runs/loopwam_nt/speed_cached_retry1}
cache=runs/loopwam_nt/latent_cache_v1
mkdir -p "$root"
for trial in fill:ddp:10 warm:ddp:10 warm:zero1:10 warm:zero2:10; do
  mode=${trial%%:*}; rest=${trial#*:}; backend=${rest%%:*}; updates=${rest##*:}
  name=${mode}_${backend}_mb8
  mkdir -p "$root/$name"
  torchrun --standalone --nproc_per_node=2 scripts/benchmark_loopwam.py --backend "$backend" --microbatch 8 --structured-attention --latent-cache-dir "$cache" --updates "$updates" --output-dir "$root/$name" > "$root/$name/launcher.log" 2>&1
  status=$?; echo "$name exit=$status"; echo "$status" > "$root/$name/exit_code"
  if [[ "$status" != 0 ]]; then exit "$status"; fi
done
