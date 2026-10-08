#!/bin/bash
#SBATCH --job-name=test_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=120:00:00
set -euo pipefail
: "${KV_SOURCE:?Set immutable source checkout}"
: "${KV_OUTPUT:?Set fresh campaign output directory}"
: "${KV_PEER:?Set peer campaign output directory}"
: "${KV_QUEUE:?concat or mix}"
: "${KV_EVAL_WORKERS:?Set verified workers per GPU}"
: "${KV_EVAL_RENDER_THREADS:?Set verified render thread count}"
cd "$KV_SOURCE"
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LP_NUM_THREADS=1
python scripts/run_kv_campaign.py --queue "$KV_QUEUE" --output-root "$KV_OUTPUT" --peer-root "$KV_PEER" --phase prepare --eval-workers "$KV_EVAL_WORKERS" --eval-render-threads "$KV_EVAL_RENDER_THREADS"
# The release is written only after both CPU/GPU/native/fairness gates are audited.
for ((i=0;i<4320;i++)); do
  if [[ -f "$KV_OUTPUT/release.json" ]]; then
    exec python scripts/run_kv_campaign.py --queue "$KV_QUEUE" --output-root "$KV_OUTPUT" --peer-root "$KV_PEER" --phase run
  fi
  sleep 10
done
exit 1
