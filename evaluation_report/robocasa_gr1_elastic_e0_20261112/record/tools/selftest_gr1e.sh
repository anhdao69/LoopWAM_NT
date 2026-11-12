#!/usr/bin/env bash
# Self-test of lib_gr1e.sh (the orchestrator's helpers) against the real GR1 continuation state (read only) and
# synthetic cases in a scratch dir; run on worker-2 (srun --overlap -c1) while the continuation trains epoch 10.
set -u
T=$(cd "$(dirname "$0")" && pwd); source "$T/env_gr1e.sh"
S=$EGR_WORK/selftest; rm -rf "$S"; mkdir -p "$S"
B=$S; V=$S/validation; EV=$S/eval; mkdir -p "$V" "$EV"
source "$T/lib_gr1e.sh"
fails=0
expect() { local want=$1 name=$2; shift 2; "$@"; local got=$?; if [ "$got" = "$want" ]; then echo "PASS $name (rc $got)"; else echo "FAIL $name (rc $got, want $want)"; fails=$((fails + 1)); fi; }
# real continuation state (epoch 10 training now; its orchestrator holds the lock)
echo "continuation phase now: $(cont_phase)"
expect 1 "cont_finished while the continuation trains" cont_finished
expect 1 "continuation lock held" lock_free /tmp/gr1cont_orchestrator_5758.lock
REAL=$GR1_CONT
GR1_CONT=$S/cont; mkdir -p "$GR1_CONT"
echo '{"phase": "done"}' > "$GR1_CONT/STATUS.json"; expect 1 "done but lock still held" cont_finished
LOCK=/tmp/gr1cont_orchestrator_5758.lock
cont_finished_test() { case "$(cont_phase)" in done|failed) lock_free "$S/free.lock" ;; *) return 1 ;; esac; }
expect 0 "done and lock free" cont_finished_test
echo '{"phase": "failed"}' > "$GR1_CONT/STATUS.json"; expect 0 "failed and lock free" cont_finished_test
echo '{"phase": "eval"}' > "$GR1_CONT/STATUS.json"; expect 1 "eval phase" cont_finished_test
echo '{"phase": "eval_failed"}' > "$GR1_CONT/STATUS.json"; expect 1 "eval_failed is not final" cont_finished_test
GR1_CONT=$REAL
# synthetic run dir
OUT=$S/run; mkdir -p "$OUT"; PER=47032; EPOCHS=5
st() { printf '{"epoch":%s,"next_micro":%s,"update":%s,"windows_seen":0,"status":"%s"}' "$1" "$2" "$3" "$4" > "$OUT/status.json"; }
touch "$OUT/epoch_01.pt"
st 1 0 47032 running; expect 1 "epoch 1 without the update-2000 pair" epoch_complete 1
touch "$OUT/step_00002000.pt" "$OUT/step_00002000_recovery.pt"
expect 0 "epoch 1 of 5 complete" epoch_complete 1
st 1 0 47032 complete; expect 1 "epoch 1 marked complete in a 5-epoch run" epoch_complete 1
st 1 3200 47000 running; expect 1 "mid epoch 1" epoch_complete 1
st 2 0 94064 running; expect 3 "past the end of epoch 1" epoch_complete 1
expect 1 "epoch 2 without epoch_02.pt" epoch_complete 2
touch "$OUT/epoch_02.pt"; expect 0 "epoch 2 complete" epoch_complete 2
touch "$OUT/epoch_05.pt"; st 5 0 235160 complete; expect 0 "epoch 5 complete (status complete)" epoch_complete 5
st 5 0 235160 running; expect 1 "epoch 5 'running'" epoch_complete 5
# keep_state
head -c 1000000 /dev/urandom > "$OUT/latest.pt"
expect 0 "keep_state copies latest.pt" keep_state 3
expect 0 "copy is byte-identical" cmp -s "$OUT/latest.pt" "$OUT/epoch_03_recovery.pt"
expect 0 "keep_state idempotent" keep_state 3
head -c 1000000 /dev/urandom > "$OUT/latest.pt"
expect 1 "keep_state never overwrites a different copy" keep_state 3
expect 1 "no .tmp left" test -e "$OUT/epoch_03_recovery.pt.tmp"
# verified / eval_valid / alive / attempts
echo '{"passed": true}' > "$V/verify_X.json"; echo '{"passed": false}' > "$V/verify_Y.json"
expect 0 "verified passed json" verified X; expect 1 "verified failed json" verified Y; expect 1 "verified missing json" verified Z
echo '{"valid": true}' > "$EV/a.json"; echo '{"valid": false}' > "$EV/b.json"
expect 0 "eval_valid true" eval_valid "$EV/a.json"; expect 1 "eval_valid false" eval_valid "$EV/b.json"; expect 1 "eval_valid missing" eval_valid "$EV/c.json"
expect 1 "alive: no holder of run.lock" alive
( exec 9>>"$OUT/run.lock"; flock 9; sleep 5 ) & sleep 1; expect 0 "alive: run.lock held" alive; wait
[ "$(attempts e4)" = 0 ] && echo "PASS attempts default 0" || { echo "FAIL attempts default"; fails=$((fails + 1)); }
# validated
B=$S/missing; expect 2 "validation not run" validated
B=$S/vok; mkdir -p "$B/prelaunch"
for n in preflight_core preflight_speed_E0 resume_A resume_B1 resume_B2 resume_compare schedule_contract verify_early verify_epoch eval_smoke; do echo "$n rc=0"; done > "$B/prelaunch/validate.rc"
for f in preflight_core preflight_speed_E0 verify_early verify_epoch eval_smoke; do echo '{"passed": true}' > "$B/prelaunch/$f.json"; done
expect 2 "validation without VALIDATE_DONE" validated
echo VALIDATE_DONE >> "$B/prelaunch/validate.rc"; expect 0 "validation all passed" validated
sed -i 's/^eval_smoke rc=0/eval_smoke rc=9/' "$B/prelaunch/validate.rc"; expect 1 "eval smoke skipped" validated
sed -i 's/^eval_smoke rc=9/eval_smoke rc=0/' "$B/prelaunch/validate.rc"; echo '{"passed": false}' > "$B/prelaunch/verify_epoch.json"
expect 1 "a report passed=false" validated
# schedules per epoch
[ "$(gr1e_schedules 1)" = flat_k1_n2 ] && [ "$(gr1e_schedules 4)" = flat_k1_n2 ] && [ "$(gr1e_schedules 5)" = "flat_k1_n2 flat_k4_n2 flat_k4_n5" ] \
  && echo "PASS eval schedules per epoch" || { echo "FAIL eval schedules"; fails=$((fails + 1)); }
# status / log write into the scratch dir only; report on an empty record dir
B=$S; status testing "selftest"; python3 -I -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["phase"]=="testing" else 1)' "$S/STATUS.json" \
  && grep -q "STATUS testing: selftest" "$S/orchestrator.log" && echo "PASS status/log" || { echo "FAIL status/log"; fails=$((fails + 1)); }
python3 -I "$T/report_gr1e.py" "$S" > /dev/null && [ -f "$S/RESULTS.md" ] && echo "PASS report" || { echo "FAIL report"; fails=$((fails + 1)); }
echo "SELFTEST fails=$fails"
