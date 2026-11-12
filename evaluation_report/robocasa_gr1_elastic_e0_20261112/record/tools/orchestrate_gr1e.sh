#!/usr/bin/env bash
# GR1-E0 x5 (elastic E0 on RoboCasa-GR1, 5 epochs, from the GR1 continuation's epoch-10 teacher if it beats epoch 9,
# else epoch 9) on GPUs 4-7 of job 5758, after the GR1 continuation (epochs 6-10 + evals) has finished.
# Runs as a 2-CPU step (-c2 --gres=none -J training_vln); idempotent: every decision comes from files, so a relaunch
# (keeper_gr1e.sh) continues where the previous process stopped.
#  1. wait until the GR1 continuation's STATUS is final (done/failed) and its orchestrator lock is free
#  2. teacher selection (select_gr1_teacher.py, once) and the common_v1 init from that teacher (elastic_init.py --gr1)
#  3. pre-launch validation (validate_gr1e.sh) once; training starts only if every check passed
#  4. for E = 1..5: train to the end of epoch E (train_gr1e.sh, --max-updates E*47032; up to 6 attempts per epoch with
#     resume from latest.pt; an attempt whose train.log/metrics.jsonl stop changing for 30 min is cancelled), keep the
#     epoch-end state as epoch_EE_recovery.pt, verify epoch_EE.pt against it (verify_gr1_export.py on GPU 4, incl. the
#     eval server for that epoch's schedules; epoch 1 also the update-2000 pair), then evaluate it with training
#     paused (eval_gr1e.sh: epochs 1-4 flat_k1_n2; epoch 5 flat_k1_n2, flat_k4_n2, flat_k4_n5; up to 3 attempts each,
#     then validate_elastic_eval.py), then the next epoch. A failed verification is retried once; if it still fails the
#     epoch is recorded unverified, not evaluated, and training continues. Results: RESULTS.md.
# Usage: orchestrate_gr1e.sh REVISION
set -u
REV=$1
D=$(cd "$(dirname "$0")" && pwd); source "$D/env_gr1e.sh"
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
B=$EGR_WORK; RUN=$GR1E_RUN; OUT=$B/runs/$RUN; V=$B/validation; L=$B/logs; EV=$B/eval
mkdir -p "$B/runs" "$V" "$L" "$EV" "$B/init"
source "$D/lib_gr1e.sh"
JOB=5758; PER=$GR1E_PER_EPOCH; EPOCHS=$GR1E_EPOCHS; MAX_ATTEMPTS=6; HANG=1800
exec 8>/tmp/egr_gr1e_orchestrator.lock
flock -n 8 || { echo "[$(date -u +%FT%TZ)] another orchestrate_gr1e is running on $(hostname)" >> "$B/orchestrator.log"; exit 0; }
# The lock (fd 8) is held by this shell only: every child is started with 8>&-, so a relaunch after this shell died is
# never locked out by a still-running training/eval step; the heartbeat stops with this shell ($$ in the subshell).
( exec 8>&-; while kill -0 $$ 2>/dev/null; do touch "$B/heartbeat"; sleep 60; done ) & BEAT=$!
trap 'kill $BEAT 2>/dev/null' EXIT
log "orchestrator start: host $(hostname) job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?} pid $$ revision $REV run $RUN epochs $EPOCHS"
cd "$ELASTIC_ROOT"
if [ "$(git rev-parse HEAD)" != "$REV" ] || [ -n "$(git status --porcelain -- scripts src experiments plans)" ]; then
  status failed "code at $ELASTIC_ROOT is not $REV with a clean tree"; exit 1
fi

# ---- 1. wait for the GR1 continuation ----
if ! cont_finished; then
  status waiting "for the GR1 continuation (epoch 10 training + eval) to finish on GPUs 4-7"
  n=0; until cont_finished; do sleep 60; n=$((n + 1)); [ $((n % 30)) = 0 ] && log "continuation: $(tr -d '\n' < "$GR1_CONT/STATUS.json" | cut -c1-200)"; done
  log "GR1 continuation finished: $(tr -d '\n' < "$GR1_CONT/STATUS.json")"
  sleep 120   # let its last step release GPUs 4-7
fi

# ---- 2. teacher selection and initialization ----
if [ ! -f "$ELASTIC_SELECTION" ]; then
  status selecting "teacher: epoch 10 if its official success count beats epoch 9's 626/1200, else epoch 9"
  python scripts/select_gr1_teacher.py --eval-dir "$GR1_EVAL" --out "$ELASTIC_SELECTION" > "$L/select_teacher.log" 2>&1 \
    || { status failed "teacher selection failed (logs/select_teacher.log)"; exit 1; }
fi
source "$D/env_gr1e.sh"   # ELASTIC_TEACHER from the selection record
log "teacher selection: $(python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); print("epoch", d["chosen_epoch"], d["teacher_sha256"][:12], "-", d["reason"])' "$ELASTIC_SELECTION")"
TSHA=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_sha256"])' "$ELASTIC_SELECTION")
IJ=${ELASTIC_INIT%.pt}.json
if [ ! -f "$ELASTIC_INIT" ] || [ ! -f "$IJ" ]; then   # elastic_init.py writes the .pt, then its .json (deterministic rebuild)
  status initializing "common_v1 init from the selected teacher (CPU)"
  python scripts/elastic_init.py common_v1 --gr1 --teacher "$ELASTIC_TEACHER" --vae-path "$ELASTIC_VAE" --out "$ELASTIC_INIT" \
    > "$L/elastic_init.log" 2>&1 8>&- || { status failed "elastic init failed (logs/elastic_init.log)"; exit 1; }
fi
python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["teacher_sha256"] == sys.argv[2] and d["recipe"] == "common_v1" and d["sha256"] == sys.argv[3] else 1)' \
  "$IJ" "$TSHA" "$(sha256sum "$ELASTIC_INIT" | cut -d' ' -f1)" || { status failed "init artifact was not built from the selected teacher (or its .json does not match the .pt)"; exit 1; }

# ---- 3. pre-launch validation ----
validated; vr=$?
try=0
while [ $vr = 2 ] && [ $try -lt 3 ]; do   # not run yet, or interrupted (e.g. GPUs 4-7 not free): up to 3 runs
  try=$((try + 1))
  [ -e "$B/prelaunch" ] && { mv "$B/prelaunch" "$B/prelaunch.incomplete_$(date -u +%Y%m%dT%H%M%SZ)"; log "kept an incomplete validation aside"; }
  status validating "pre-launch validation of $RUN x$EPOCHS on GPUs 4-7 (run $try)"
  srun --jobid=$JOB --overlap -n1 -c120 --cpu-bind=none --gres=gpu:8 -J training_vln bash "$D/validate_gr1e.sh" \
    >> "$L/validate.log" 2>&1 8>&-
  log "validation step rc=$?: $(tr '\n' ' ' < "$B/prelaunch/validate.rc" 2>/dev/null)"
  validated; vr=$?
  [ $vr = 2 ] && sleep 300
done
if [ $vr != 0 ]; then status failed_validation "pre-launch validation did not pass (see $B/prelaunch); nothing trained"; exit 1; fi
log "pre-launch validation PASSED"
# not required for training: the --finalize-only recovery path on a forged epoch-end copy (validate_gr1e.sh step 6)
fin=$(grep -E '^(finalize|verify_finalize) rc=' "$B/prelaunch/validate.rc" | tr '\n' ' ')
if [ "$fin" = "finalize rc=0 verify_finalize rc=0 " ] && passed "$B/prelaunch/verify_finalize.json"; then log "recovery path (--finalize-only) test passed"
else log "WARNING: recovery path (--finalize-only) test did not pass ($fin): an epoch whose export is lost would need a manual fix"; FIN_PROBLEM=" finalize_test"; fi

# ---- 4. epochs ----
watch() {  # PID (or empty when monitoring an orphan): wait for the attempt to end; cancel it on a hang
  local pid=$1 start; start=$(date +%s)
  while { [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; } || { [ -z "$pid" ] && alive; }; do
    sleep 60
    local now last; now=$(date +%s); last=$(last_write); last=${last:-$start}
    [ "$last" -lt "$start" ] && last=$start
    if [ $((now - last)) -gt "$HANG" ]; then
      local s; s=$(cat "$OUT/step_id" 2>/dev/null)
      log "$RUN: no training output for $((now - last)) s: cancelling step $JOB.${s:-?}"
      [ -n "$s" ] && scancel "$JOB.$s"
      sleep 120; start=$(date +%s)
    fi
  done
  [ -n "$pid" ] && wait "$pid"
}
verify() {  # E
  local e=$1 tag pairs; tag=$(printf '%s_e%02d' "$RUN" "$e")
  pairs=$(printf 'epoch_%02d.pt:epoch_%02d_recovery.pt' "$e" "$e")
  [ "$e" = 1 ] && pairs="$pairs,step_00002000.pt:step_00002000_recovery.pt"
  status verifying "$RUN epoch $e: export verification on GPU 4"
  rm -f "$V/verify_$tag.json"
  srun --jobid=$JOB --overlap -n1 -c120 --cpu-bind=none --gres=gpu:8 -J training_vln bash -c "
    source '$D/env_gr1e.sh'; cd \"\$ELASTIC_ROOT\"
    CUDA_VISIBLE_DEVICES=4 LOCAL_RANK=0 RANK=0 BIND_LOG='$L/verify_binding.log' BIND_TAG='$tag' bash plans/elastic_robotwin/bind_gpu.sh \
      scripts/verify_gr1_export.py --run-dir '$OUT' --pairs '$pairs' --server-schedules '$(gr1e_schedules "$e" | tr ' ' ,)' \
      --out '$V/verify_$tag.json'" >> "$L/verify_$tag.log" 2>&1 8>&-
  local rc=$?
  if verified "$tag"; then log "$RUN epoch $e: export VERIFIED ($pairs)"; else log "$RUN epoch $e: export verification FAILED rc=$rc (see $L/verify_$tag.log)"; fi
}
evaluate() {  # E S: official eval with training paused, then validation; 0 when a valid result exists
  local e=$1 s=$2 ee attempt rc res sha; ee=$(printf %02d "$e"); res=$EV/e${ee}_${s}_validation.json
  eval_valid "$res" && { log "epoch $e $s already evaluated"; return 0; }
  sha=$(sha256sum "$OUT/epoch_$ee.pt" | cut -d' ' -f1)
  for attempt in 1 2 3; do
    status eval "$RUN epoch $e (epoch_$ee.pt, step $((e * PER))) schedule $s, attempt $attempt"
    timeout --signal=TERM --kill-after=300 8h srun --jobid=$JOB --overlap -n1 -c120 --cpu-bind=none --gres=gpu:8 -J training_vln \
      bash "$D/eval_gr1e.sh" "$e" "$s" "$attempt" > "$L/eval_e${ee}_${s}_a$attempt.log" 2>&1 8>&-; rc=$?
    log "eval epoch $e $s attempt $attempt rc=$rc: $(tail -1 "$L/eval_e${ee}_${s}_a$attempt.log" | cut -c1-200)"
    if [ "$rc" = 0 ]; then
      python3 "$ELASTIC_ROOT/experiments/robocasa_gr1/validate_elastic_eval.py" "$EV/e${ee}_$s" "$((e * PER))" --schedule "$s" \
        --export-sha256 "$sha" --json "$res" > "$L/validate_e${ee}_$s.log" 2>&1
      log "validation epoch $e $s rc=$?: $(grep -E 'mean success' "$L/validate_e${ee}_$s.log" | cut -c1-220) $(tail -1 "$L/validate_e${ee}_$s.log" | cut -c1-120)"
      python3 -I "$D/report_gr1e.py" "$B" > /dev/null 2>&1 || log "report generation failed"
      eval_valid "$res" && return 0
      return 1
    fi
    sleep 120
  done
  return 1
}
problems="${FIN_PROBLEM:-}"
for E in $(seq 1 "$EPOCHS"); do
  tag=$(printf '%s_e%02d' "$RUN" "$E")
  if ! verified "$tag" && [ ! -f "$B/epoch_$E.finished" ]; then
    if alive; then
      status training "$RUN epoch $E: monitoring an attempt started by an earlier orchestrator process"
      watch ""
    fi
    epoch_complete "$E"; c=$?
    while [ $c != 0 ]; do
      if [ $c = 3 ]; then status failed "$RUN trained past the end of epoch $E: stopping for inspection"; exit 1; fi
      n=$(attempts "e$E")
      [ "$n" -ge "$MAX_ATTEMPTS" ] && break
      n=$((n + 1)); echo "$n" > "$B/attempts_e$E"
      status training "$RUN epoch $E/$EPOCHS (to update $((E * PER))): attempt $n/$MAX_ATTEMPTS ($( [ -f "$OUT/latest.pt" ] && echo resume || echo fresh ))"
      srun --jobid=$JOB --overlap -n1 -c120 --cpu-bind=none --gres=gpu:8 -J training_vln \
        bash "$D/train_gr1e.sh" "$REV" "$E" >> "$L/train_e$E.log" 2>&1 8>&- &
      watch $!; rc=$?
      log "$RUN epoch $E: attempt $n ended rc=$rc; status $(tr -d '\n ' < "$OUT/status.json" 2>/dev/null | cut -c1-200)"
      epoch_complete "$E"; c=$?
      [ $c = 0 ] || sleep 60
    done
    if [ $c != 0 ]; then status failed "$RUN epoch $E incomplete after $(attempts "e$E") attempts"; exit 1; fi
    log "$RUN epoch $E: training COMPLETE (update $((E * PER)))"
    if ! keep_state "$E"; then status failed "$RUN epoch $E: could not keep the epoch-end state"; exit 1; fi
    verify "$E"
    verified "$tag" || { log "$RUN epoch $E: retrying the verification once"; verify "$E"; }
    verified "$tag" || echo "unverified $(date -u +%FT%TZ)" > "$B/epoch_$E.finished"
  fi
  if ! verified "$tag"; then log "epoch $E: export UNVERIFIED, not evaluated"; problems="$problems epoch$E:unverified"; continue; fi
  for S in $(gr1e_schedules "$E"); do
    evaluate "$E" "$S" || problems="$problems epoch$E/$S:eval"
  done
done
python3 -I "$D/report_gr1e.py" "$B" > /dev/null 2>&1 || log "report generation failed"
if [ -n "$problems" ]; then status done_with_problems "$RUN x$EPOCHS trained and evaluated;${problems} (RESULTS.md)"
else status done "$RUN x$EPOCHS trained, verified and evaluated (RESULTS.md)"; fi
