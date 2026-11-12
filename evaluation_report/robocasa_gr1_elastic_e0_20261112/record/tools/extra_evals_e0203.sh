#!/bin/bash
# Requested by the user on 2026-11-11: "After epoch 4 finish then run eval [2, 2], [1, 1], [4, 4] on the epoch 2,3 for me".
# Epochs 2 and 3 (epoch_02.pt update 94,064; epoch_03.pt update 141,096) already have valid official [1,1] (flat_k1_n2)
# evaluations, so this runs the missing flat_k2_n2 and flat_k4_n2 for each epoch (an existing eval dir or validation is
# never re-run or overwritten; flat_k1_n2 is in the list only so that the log records it is present), with the same
# official protocol (eval_gr1e_flat9.sh: 24 tasks x 50 episodes, seed 0, n_action_steps 12, EGL, GPUs 4-7) and
# validate_elastic_eval.py, up to 3 attempts per evaluation.
# It first waits until the epoch-4 driver (extra_evals_e04.sh) reports phase "done" in EXTRA_E04_STATUS.json. It NEVER
# writes STATUS.json or EXTRA_E04_STATUS.json; its own state is EXTRA_E0203_STATUS.json and logs/extra_evals_e0203.log
# (also orchestrator.log). Runs on a login node under setsid nohup. Usage: extra_evals_e0203.sh
B=/mnt/data/vmo-ai-task/anhdh35/elastic_gr1; T=$B/tools; L=$B/logs; EV=$B/eval
ROOT=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_gr1
PER=47032
EPOCHS="2 3"; SCHEDULES="flat_k1_n2 flat_k2_n2 flat_k4_n2"
log() { echo "[$(date -u +%FT%TZ)] extra-e0203: $*" | tee -a "$L/extra_evals_e0203.log" >> "$B/orchestrator.log"; }
status() { python3 -I -c 'import json,sys,time; json.dump(dict(phase=sys.argv[1], detail=sys.argv[2], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(sys.argv[3], "w"), indent=1)' \
  "$1" "$2" "$B/EXTRA_E0203_STATUS.json.tmp" && mv "$B/EXTRA_E0203_STATUS.json.tmp" "$B/EXTRA_E0203_STATUS.json"; log "EXTRA STATUS $1: $2"; }
e04_phase() { python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "$B/EXTRA_E04_STATUS.json" 2> /dev/null; }
fresh() { if [ -e "$1" ]; then echo "${1%.log}_$(date -u +%Y%m%dT%H%M%SZ).log"; else echo "$1"; fi; }  # never overwrite an existing log
# one instance only: an atomic lock directory on the shared filesystem (the two login hosts cannot see each other's processes)
mkdir "$B/extra_evals_e0203.lock" 2> /dev/null || { echo "[$(date -u +%FT%TZ)] extra-e0203: lock $B/extra_evals_e0203.lock exists (another instance on some login host): exit" >> "$L/extra_evals_e0203.log"; exit 0; }
echo "host $(hostname) pid $$ $(date -u +%FT%TZ)" > "$B/extra_evals_e0203.lock/owner"
log "start: host $(hostname) pid $$; waiting for the epoch-4 driver (extra_evals_e04.sh) to finish"
status waiting "for the epoch-4 evaluations (extra_evals_e04.sh) to finish"
idle=0
while :; do
  phase=$(e04_phase)
  case "$phase" in
    done) break ;;
    failed|not_started)
      status not_started "the epoch-4 driver ended with phase '$phase': epoch 2/3 evaluations not started (inspect first)"; exit 1 ;;
  esac
  # while it evaluates, the epoch-4 driver holds one Slurm step named training_vln in job 5758 (gaps between its
  # evaluations are a few seconds, between failed attempts 2 min); none for 30 min of successful queries = it has died
  if steps=$(squeue -u "$(id -un)" -s -h -o '%i %j' 2> /dev/null); then
    if echo "$steps" | grep -qE '^5758\.[0-9]+ +training_vln'; then idle=0; else idle=$((idle + 1)); fi
  fi
  if [ "$idle" -ge 15 ]; then
    status not_started "the epoch-4 driver is in phase '$phase' but no evaluation step has run for 30 min: epoch 2/3 evaluations not started (inspect first)"; exit 1
  fi
  sleep 120
done
log "epoch-4 driver finished: $(tr -d '\n' < "$B/EXTRA_E04_STATUS.json" | cut -c1-400)"
sleep 60
problems=""
for E in $EPOCHS; do
  EE=$(printf %02d "$E"); STEP=$((E * PER)); CK=$B/runs/GR1-E0/epoch_$EE.pt
  if [ ! -f "$CK" ]; then log "missing $CK: epoch $E skipped"; problems="$problems epoch$E:missing_checkpoint"; continue; fi
  SHA=$(sha256sum "$CK" | cut -d' ' -f1)
  VSHA=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["export_sha256"])' "$EV/e${EE}_flat_k1_n2_validation.json" 2> /dev/null)
  if [ -z "$SHA" ] || [ "$SHA" != "$VSHA" ]; then
    log "epoch_$EE.pt sha '$SHA' != the [1,1] evaluation's '$VSHA': epoch $E skipped"; problems="$problems epoch$E:sha_mismatch"; continue
  fi
  log "epoch_$EE.pt sha256 $SHA (same file as the existing epoch-$E [1,1] evaluation), step $STEP"
  for S in $SCHEDULES; do
    OUT=$EV/e${EE}_$S; RES=$EV/e${EE}_${S}_validation.json
    if [ -e "$OUT" ] || [ -e "$RES" ]; then log "$OUT or its validation already exists: not re-running epoch $E $S"; continue; fi
    ok=0
    for attempt in 1 2 3; do
      status eval "epoch_$EE.pt (step $STEP) schedule $S, attempt $attempt"
      ELOG=$(fresh "$L/eval_e${EE}_${S}_a$attempt.log")
      timeout --signal=TERM --kill-after=300 8h srun --jobid=5758 --overlap -n1 -c120 --cpu-bind=none --gres=gpu:8 -J training_vln \
        bash "$T/eval_gr1e_flat9.sh" "$E" "$S" "$attempt" > "$ELOG" 2>&1; rc=$?
      log "eval epoch $E $S attempt $attempt rc=$rc: $(tail -1 "$ELOG" | cut -c1-200)"
      if [ "$rc" = 0 ]; then
        VLOG=$(fresh "$L/validate_e${EE}_$S.log")
        python3 "$ROOT/experiments/robocasa_gr1/validate_elastic_eval.py" "$OUT" "$STEP" --schedule "$S" --export-sha256 "$SHA" \
          --json "$RES" > "$VLOG" 2>&1
        log "validation epoch $E $S rc=$?: $(grep -E 'mean success' "$VLOG" | cut -c1-220) $(tail -1 "$VLOG" | cut -c1-120)"
        python3 -I "$T/report_gr1e.py" "$B" > /dev/null 2>&1 || log "report generation failed"
        ok=1; break
      fi
      sleep 120
    done
    [ "$ok" = 1 ] || problems="$problems epoch$E/$S:eval_failed"
  done
done
res=$(for E in $EPOCHS; do EE=$(printf %02d "$E"); for X in $SCHEDULES; do python3 -I -c 'import json,sys; v=json.load(open(sys.argv[1])); print("e%s %s %s/%s valid=%s" % (sys.argv[3], sys.argv[2], v["successes"], v["episodes"], v["valid"]), end="; ")' "$EV/e${EE}_${X}_validation.json" "$X" "$EE" 2>/dev/null || printf 'e%s %s missing; ' "$EE" "$X"; done; done)
status done "epochs 2 and 3 at [1,1], [2,2], [4,4]: $res${problems:+ problems:$problems}"
log "done"
