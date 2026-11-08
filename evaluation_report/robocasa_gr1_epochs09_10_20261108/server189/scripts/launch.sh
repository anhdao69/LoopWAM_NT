#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
E=$1; PAR=$2; NS=$3; PORT=$4
exec 9>"$RUN_ROOT/tmp/epoch${E}.lock"
flock -n 9 || { echo 'Already running'; exit 1; }
if [[ ! -f "$RUN_ROOT/results/smoke_/epoch_$(printf '%02d' "$E")/VALIDATED.json" ]]; then
 "$FASTWAM_PY" "$RUN_ROOT/scripts/run.py" "$E" --parallel 1 --servers 1 --smoke --port "$((PORT+10))"
fi
exec "$FASTWAM_PY" "$RUN_ROOT/scripts/run.py" "$E" --parallel "$PAR" --servers "$NS" --port "$PORT"
