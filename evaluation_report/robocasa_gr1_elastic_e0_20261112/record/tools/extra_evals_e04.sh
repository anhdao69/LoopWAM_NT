#!/bin/bash
# Requested by the user on 2026-11-11: "After finish 9 runs above run 9 runs for epoch 4 for me". Epoch 4 (epoch_04.pt,
# update 188,128) already has valid official evaluations for flat_k1_n2, flat_k2_n2, flat_k4_n2 and flat_k4_n5, so this
# completes the FLAT9 set with the five missing schedules, with the same official protocol (eval_gr1e_flat9.sh: 24 tasks
# x 50 episodes, seed 0, n_action_steps 12, EGL, GPUs 4-7) and validate_elastic_eval.py, up to 3 attempts per schedule.
# It first waits until the epoch-5 orchestrator (orchestrate_gr1e_e5.sh) has reached done / done_with_problems, its
# heartbeat has stopped and its keeper has exited. It NEVER writes STATUS.json (the e5 keeper relaunches the orchestrator
# on a non-final phase); its own state is EXTRA_E04_STATUS.json and logs/extra_evals_e04.log (also orchestrator.log).
# Never overwrites an existing eval dir. Runs on a login node under setsid nohup. Usage: extra_evals_e04.sh
B=/mnt/data/vmo-ai-task/anhdh35/elastic_gr1; T=$B/tools; L=$B/logs; EV=$B/eval
ROOT=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_gr1
E=4; STEP=188128; CK=$B/runs/GR1-E0/epoch_04.pt
SCHEDULES="flat_k2_n5 flat_k1_n5 flat_k4_n10 flat_k2_n10 flat_k1_n10"
log() { echo "[$(date -u +%FT%TZ)] extra-e04: $*" | tee -a "$L/extra_evals_e04.log" >> "$B/orchestrator.log"; }
status() { python3 -I -c 'import json,sys,time; json.dump(dict(phase=sys.argv[1], detail=sys.argv[2], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(sys.argv[3], "w"), indent=1)' \
  "$1" "$2" "$B/EXTRA_E04_STATUS.json.tmp" && mv "$B/EXTRA_E04_STATUS.json.tmp" "$B/EXTRA_E04_STATUS.json"; log "EXTRA STATUS $1: $2"; }
phase_now() { python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "$B/STATUS.json" 2> /dev/null; }
# one instance only: an atomic lock directory on the shared filesystem (the two login hosts cannot see each other's processes)
mkdir "$B/extra_evals_e04.lock" 2> /dev/null || { echo "[$(date -u +%FT%TZ)] extra-e04: lock $B/extra_evals_e04.lock exists (another instance on some login host): exit" >> "$L/extra_evals_e04.log"; exit 0; }
echo "host $(hostname) pid $$ $(date -u +%FT%TZ)" > "$B/extra_evals_e04.lock/owner"
log "start: host $(hostname) pid $$; waiting for the epoch-5 orchestrator to finish"
status waiting "for the nine epoch-5 evaluations (orchestrate_gr1e_e5.sh) to finish"
while :; do
  phase=$(phase_now)
  case "$phase" in
    done|done_with_problems)
      age=$(( $(date +%s) - $(stat -c %Y "$B/heartbeat" 2> /dev/null || echo 0) ))
      [ "$age" -gt 180 ] && break ;;
    failed|failed_validation)
      status not_started "the epoch-5 orchestrator ended with phase '$phase': epoch-4 evaluations not started (inspect first)"; exit 1 ;;
  esac
  sleep 120
done
log "epoch-5 orchestrator finished: $(tr -d '\n' < "$B/STATUS.json" | cut -c1-300)"
for i in $(seq 50); do tail -1 "$B/keeper.log" | grep -qE 'keeper exits|keeper gives up' && break; sleep 30; done
log "keeper: $(tail -1 "$B/keeper.log" | cut -c1-160)"
SHA=$(sha256sum "$CK" | cut -d' ' -f1)
VSHA=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["export_sha256"])' "$EV/e04_flat_k1_n2_validation.json")
[ "$SHA" = "$VSHA" ] || { status failed "epoch_04.pt sha $SHA != the [1,1] evaluation's $VSHA"; exit 1; }
log "epoch_04.pt sha256 $SHA (same file as the four existing epoch-4 evaluations)"
for S in $SCHEDULES; do
  OUT=$EV/e04_$S; RES=$EV/e04_${S}_validation.json
  if [ -e "$OUT" ] || [ -e "$RES" ]; then log "$OUT or its validation already exists: not re-running $S"; continue; fi
  for attempt in 1 2 3; do
    status eval "epoch_04.pt (step $STEP) schedule $S, attempt $attempt"
    timeout --signal=TERM --kill-after=300 8h srun --jobid=5758 --overlap -n1 -c120 --cpu-bind=none --gres=gpu:8 -J training_vln \
      bash "$T/eval_gr1e_flat9.sh" "$E" "$S" "$attempt" > "$L/eval_e04_${S}_a$attempt.log" 2>&1; rc=$?
    log "eval epoch $E $S attempt $attempt rc=$rc: $(tail -1 "$L/eval_e04_${S}_a$attempt.log" | cut -c1-200)"
    if [ "$rc" = 0 ]; then
      python3 "$ROOT/experiments/robocasa_gr1/validate_elastic_eval.py" "$OUT" "$STEP" --schedule "$S" --export-sha256 "$SHA" \
        --json "$RES" > "$L/validate_e04_$S.log" 2>&1
      log "validation epoch $E $S rc=$?: $(grep -E 'mean success' "$L/validate_e04_$S.log" | cut -c1-220) $(tail -1 "$L/validate_e04_$S.log" | cut -c1-120)"
      python3 -I "$T/report_gr1e.py" "$B" > /dev/null 2>&1 || log "report generation failed"
      break
    fi
    sleep 120
  done
done
res=$(for X in flat_k1_n2 flat_k1_n5 flat_k1_n10 flat_k2_n2 flat_k2_n5 flat_k2_n10 flat_k4_n2 flat_k4_n5 flat_k4_n10; do python3 -I -c 'import json,sys; v=json.load(open(sys.argv[1])); print("%s %s/%s valid=%s" % (sys.argv[2], v["successes"], v["episodes"], v["valid"]), end="; ")' "$EV/e04_${X}_validation.json" "$X" 2>/dev/null || printf '%s missing; ' "$X"; done)
status done "epoch 4 FLAT9 evaluations: $res"
log "done"
