#!/usr/bin/env bash
# ws: validate each official GR1 epoch as soon as its summary.json is complete; then write the per-task report.
R=/storage/anhdh35/robocasa_gr1_eval
log() { echo "[$(date -u +%FT%TZ)] ws-validate: $*"; }
complete() { python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['by_n_action_steps']['12']['complete'] else 1)" "$1/summary.json" 2>/dev/null; }
for E in 4 5; do
  O=$R/results/official_epoch0$E
  until complete "$O" || ! pgrep -f ws_gr1_eval_v2.sh > /dev/null; do sleep 120; done
  complete "$O" || { log "epoch $E: eval script ended without a complete summary"; exit 1; }
  sleep 30
  python3 $R/gr1_validate.py "$O" $((E * 23516)) --json "$O"_validation.json > "$O"_validation.txt 2>&1
  log "epoch $E validation rc=$?: $(head -1 "$O"_validation.txt | cut -c1-200) / $(tail -1 "$O"_validation.txt)"
done
python3 $R/gr1_report.py $R/results/official_results_ws.md "ws epoch 4=$R/results/official_epoch04_validation.json" \
  "ws epoch 5=$R/results/official_epoch05_validation.json" > /dev/null && log "report: $R/results/official_results_ws.md"
