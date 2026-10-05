#!/usr/bin/env bash
# Reuse an existing interactive allocation; never cancels another workload.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 JOB_ID OUTPUT_DIR [extra train_loopwam.py arguments]" >&2
  exit 2
fi
LOOPWAM_JOB_ID=$1
LOOPWAM_OUTPUT=$2
shift 2
cd "$(dirname "$0")/.."
mkdir -p "$LOOPWAM_OUTPUT"
# Caller must first verify that both allocated GPUs are idle.
nohup srun --jobid="$LOOPWAM_JOB_ID" --overlap --exact -N1 -n1 -c12 --gres=gpu:2 \
  bash -c 'source scripts/h100_env.sh; exec torchrun --standalone --nproc_per_node=2 scripts/train_loopwam.py --config configs/loopwam_s_v0.yaml --output-dir "$@"' \
  _ "$LOOPWAM_OUTPUT" "$@" > "$LOOPWAM_OUTPUT/launcher.log" 2>&1 < /dev/null &
echo "$!" > "$LOOPWAM_OUTPUT/srun.pid"
echo "Started srun PID $! in allocation $LOOPWAM_JOB_ID; log: $LOOPWAM_OUTPUT/launcher.log"
