#!/usr/bin/env bash
set -uo pipefail
source "$(dirname "$0")/env.sh"
REPORT=/storage/anhdh35/LoopWAM_NT/evaluation_report/robocasa_gr1_epochs09_10_20261108
REMOTE=/media/newdisk/robocasa_gr1_epochs09_10_20261108
SSH='ssh -o BatchMode=yes -o ConnectTimeout=10 -S /tmp/loopwam-server189-20261104/control'
exec 8>"$RUN_ROOT/tmp/report_sync.lock"
flock -n 8 || exit 0
mkdir -p "$REPORT/local" "$REPORT/server189"
while :; do
  date -u +%FT%TZ > "$REPORT/sync_started_utc.txt"
  python3 "$RUN_ROOT/scripts/status.py" > "$REPORT/local_live_status.json.tmp" && mv "$REPORT/local_live_status.json.tmp" "$REPORT/local_live_status.json"
  rsync -a --exclude='*.pt' --exclude='*.pth' --exclude='__pycache__' "$RUN_ROOT/scripts" "$RUN_ROOT/evidence" "$REPORT/local/"
  rsync -a --exclude='videos' --exclude='partial_attempts' "$RUN_ROOT/results/" "$REPORT/local/results/"
  $SSH vm-perception-ws@10.148.183.189 "python3 '$REMOTE/scripts/status.py'" > "$REPORT/server189_live_status.json.tmp" && mv "$REPORT/server189_live_status.json.tmp" "$REPORT/server189_live_status.json"
  rsync -a -e "$SSH" --exclude='*.pt' --exclude='*.pth' --exclude='__pycache__' "vm-perception-ws@10.148.183.189:$REMOTE/scripts" "vm-perception-ws@10.148.183.189:$REMOTE/evidence" "$REPORT/server189/"
  rsync -a -e "$SSH" --exclude='videos' --exclude='partial_attempts' "vm-perception-ws@10.148.183.189:$REMOTE/results/" "$REPORT/server189/results/"
  date -u +%FT%TZ > "$REPORT/sync_completed_utc.txt"
  python3 "$RUN_ROOT/scripts/summarize_report.py"
  if [[ -f "$REPORT/local/results/official_/epoch_09/COMPLETE.json" && -f "$REPORT/server189/results/official_/epoch_10/COMPLETE.json" ]]; then break; fi
  sleep 60 8>&-
done
