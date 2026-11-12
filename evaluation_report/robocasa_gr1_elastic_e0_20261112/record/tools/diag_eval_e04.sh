#!/bin/bash
# Diagnostic evaluation requested by the user on 2026-11-10 after stopping GR1-E0 during epoch 5 (update 191,359,
# latest.pt resumable): epoch_04.pt (update 188,128) at flat_k4_n2 ([4,4]) and then flat_k4_n5 ([4]x5), with the
# orchestrator's own official protocol: the unchanged eval_gr1e.sh (24 tasks x 50 episodes, seed 0, n_action_steps 12,
# EGL, GPUs 4-7) and validate_elastic_eval.py, up to 3 attempts per schedule. Never overwrites an existing eval dir.
# Runs on a login node under setsid nohup. Usage: diag_eval_e04.sh
B=/mnt/data/vmo-ai-task/anhdh35/elastic_gr1; T=$B/tools; L=$B/logs; EV=$B/eval
ROOT=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_gr1
E=4; STEP=188128; CK=$B/runs/GR1-E0/epoch_04.pt
log() { echo "[$(date -u +%FT%TZ)] diag: $*" | tee -a "$L/diag_eval_e04.log" >> "$B/orchestrator.log"; }
status() { python3 -I -c 'import json,sys,time; json.dump(dict(phase=sys.argv[1], detail=sys.argv[2], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(sys.argv[3], "w"), indent=1)' \
  "$1" "$2" "$B/STATUS.json.tmp" && mv "$B/STATUS.json.tmp" "$B/STATUS.json"; log "STATUS $1: $2"; }
log "start: host $(hostname) pid $$"
SHA=$(sha256sum "$CK" | cut -d' ' -f1)
VSHA=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["export_sha256"])' "$EV/e04_flat_k1_n2_validation.json")
[ "$SHA" = "$VSHA" ] || { status stopped "diagnostic eval aborted: epoch_04.pt sha $SHA != the [1,1] evaluation's $VSHA"; exit 1; }
log "epoch_04.pt sha256 $SHA (same file as the [1,1] evaluation)"
for S in flat_k4_n2 flat_k4_n5; do
  OUT=$EV/e04_$S; RES=$EV/e04_${S}_validation.json
  if [ -e "$OUT" ] || [ -e "$RES" ]; then log "$OUT or its validation already exists: not re-running $S"; continue; fi
  for attempt in 1 2 3; do
    status diagnostic_eval "GR1-E0 stopped at update 191359 (epoch 5 partial, latest.pt resumable); evaluating epoch_04.pt schedule $S, attempt $attempt"
    timeout --signal=TERM --kill-after=300 8h srun --jobid=5758 --overlap -n1 -c120 --cpu-bind=none --gres=gpu:8 -J training_vln \
      bash "$T/eval_gr1e.sh" "$E" "$S" "$attempt" > "$L/eval_e04_${S}_a$attempt.log" 2>&1; rc=$?
    log "eval epoch $E $S attempt $attempt rc=$rc: $(tail -1 "$L/eval_e04_${S}_a$attempt.log" | cut -c1-200)"
    if [ "$rc" = 0 ]; then
      python3 "$ROOT/experiments/robocasa_gr1/validate_elastic_eval.py" "$OUT" "$STEP" --schedule "$S" --export-sha256 "$SHA" \
        --json "$RES" > "$L/validate_e04_$S.log" 2>&1
      log "validation epoch $E $S rc=$?: $(grep -E 'mean success' "$L/validate_e04_$S.log" | cut -c1-220) $(tail -1 "$L/validate_e04_$S.log" | cut -c1-120)"
      break
    fi
    sleep 120
  done
done
res=$(for S in flat_k1_n2 flat_k4_n2 flat_k4_n5; do python3 -I -c 'import json,sys; v=json.load(open(sys.argv[1])); print("%s %s/%s valid=%s" % (sys.argv[2], v["successes"], v["episodes"], v["valid"]), end="; ")' "$EV/e04_${S}_validation.json" "$S" 2>/dev/null || printf '%s missing; ' "$S"; done)
status stopped "GR1-E0 stopped by user at update 191359 (epoch 5 partial, latest.pt resumable); epoch 4 diagnostic evals done: $res"
log "done"
