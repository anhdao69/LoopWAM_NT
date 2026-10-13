#!/usr/bin/env bash
# Resumable evaluation of the 14 missing RT conditions; safe to run concurrently on several allocations.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/rt_eval_env.sh"
cd "$RT_ROOT"
NGPU=$(nvidia-smi -L | wc -l)
GPUS=$(seq -s, 0 $((NGPU-1)))
WPG=${WPG:-$(( $(nproc) / NGPU ))}
exec python scripts/evaluate_rt_matched.py --tag final_rollout \
  --conditions RT-A:S4,RT-T1ft:S1,RT-T1ft:S2,RT-Pa:S1,RT-Pa:S2,RT-TF:S1,RT-TF:S2,RT-A2:S1,RT-A2:S2,RT-A2:S5,RT-A2:S6,RT-A2:S7,RT-A2:S8,RT-B2a:parent2 \
  --expected-sha "$(cat $RT_OUT/expected_sha.txt)" --output-root $RT_OUT --vae-path $RT_VAE --text-cache-dir $RT_TEXT \
  --gpus "${GPUS}" --workers-per-gpu "${WPG}" --no-videos
