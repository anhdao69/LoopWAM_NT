#!/usr/bin/env bash
# Run inside a CPU-only Slurm step; child GPU stages survive controller restarts.
set -uo pipefail
if [[ $# != 3 ]]; then
  echo 'Usage: supervise_robotwin_queue.sh ENV CODE CONFIG' >&2
  exit 2
fi
queue_env=$1
queue_code=$2
queue_config=$3
source "$queue_env" || exit $?
for queue_attempt in 1 2 3; do
  python -u "$queue_code/scripts/run_robotwin_posttrain_queue.py" --config "$queue_config"
  queue_rc=$?
  if [[ $queue_rc == 0 ]]; then exit 0; fi
  echo "Controller attempt $queue_attempt exited $queue_rc at $(date -u +%FT%TZ)" >&2
  if [[ $queue_attempt != 3 ]]; then sleep 60; fi
done
exit "$queue_rc"
