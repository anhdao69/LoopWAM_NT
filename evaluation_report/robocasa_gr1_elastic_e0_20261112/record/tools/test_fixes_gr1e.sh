#!/usr/bin/env bash
# CPU test of the review fixes (scratch dirs only): trainer --finalize-only guard, train_gr1e.sh decisions with stubbed
# nvidia-smi/torchrun, validate_elastic_eval.py across retry attempts, eval_gr1e.sh server-log move.
set -u
T=/mnt/data/vmo-ai-task/anhdh35/elastic_gr1/tools; source "$T/env_gr1e.sh"
X=$EGR_WORK/test_fixes; rm -rf "$X"; mkdir -p "$X/bin" "$X/tools"; fails=0
pass() { echo "PASS $1"; }; fail() { echo "FAIL $1"; fails=$((fails + 1)); }
cd "$ELASTIC_ROOT"
python scripts/train_transport.py --run-name GR1-E0 --output-dir "$X/x" --init-artifact x --teacher-selection x --finalize-only > "$X/a.log" 2>&1
[ $? = 2 ] && grep -q -- "--finalize-only needs --resume" "$X/a.log" && pass "trainer: --finalize-only needs --resume" || { fail "trainer guard"; tail -3 "$X/a.log"; }
# train_gr1e.sh with stubs
cat > "$X/bin/nvidia-smi" <<'S'
#!/bin/bash
case "$*" in *memory.used,utilization*) echo "4, 0 MiB, 0 %";; *) for i in 1 2 3 4; do echo 0; done;; esac
S
cat > "$X/bin/torchrun" <<'S'
#!/bin/bash
echo "TORCHRUN $*" >> "$TEST_X/torchrun.calls"
S
chmod +x "$X/bin/"*
cp "$T/train_gr1e.sh" "$X/tools/"
cat > "$X/tools/env_gr1e.sh" <<S
source $T/env_gr1e.sh
export EGR_WORK=$X/work ELASTIC_SELECTION=$X/sel.json ELASTIC_TEACHER=$X/teacher.pt ELASTIC_INIT=$X/init.pt TEST_X=$X PATH=$X/bin:\$PATH
S
touch "$X/sel.json" "$X/init.pt"
REV=$(git rev-parse HEAD); OUT=$X/work/runs/GR1-E0
case_() {  # name target latest_update status_update epoch_file(0/1) want_rc want_call(none|train|resume|finalize)
  local name=$1 tgt=$2 lu=$3 su=$4 ef=$5 wrc=$6 wcall=$7
  rm -rf "$X/work"; mkdir -p "$OUT"; : > "$X/torchrun.calls"
  [ "$lu" != none ] && python -c "import torch,sys; torch.save(dict(update=int(sys.argv[1])), sys.argv[2])" "$lu" "$OUT/latest.pt"
  [ "$su" != none ] && printf '{"epoch": %s, "next_micro": 0, "update": %s, "status": "running"}' "$((su / 47032))" "$su" > "$OUT/status.json"
  [ "$ef" = 1 ] && touch "$OUT/epoch_$(printf %02d "$tgt").pt"
  bash "$X/tools/train_gr1e.sh" "$REV" "$tgt" > "$X/$name.log" 2>&1; local rc=$?
  local call=none; c=$(cat "$X/torchrun.calls")
  if [ -n "$c" ]; then call=train; [[ $c == *--resume* ]] && call=resume; [[ $c == *--finalize-only* ]] && call=finalize; fi
  [[ $c == *"--max-updates $((tgt * 47032)) "* || -z $c ]] || call="$call(bad max-updates)"
  [ "$rc" = "$wrc" ] && [ "$call" = "$wcall" ] && pass "train_gr1e $name (rc $rc, $call)" || { fail "train_gr1e $name (rc $rc want $wrc, $call want $wcall)"; tail -3 "$X/$name.log"; }
}
case_ fresh 1 none none 0 0 train
case_ mid_epoch 2 50000 50000 0 0 resume
case_ complete 2 94064 94064 1 0 none
case_ export_missing 2 94064 94064 0 0 finalize
case_ status_stale 2 94064 93800 1 0 finalize
case_ status_missing 2 94064 none 1 0 finalize
case_ past_end 2 94065 94065 1 4 none
# validate_elastic_eval.py across attempts (structure of the epoch-9 official eval, scratch logs)
E9=$GR1_EVAL/official_epoch09; O=$X/eval; mkdir -p "$O/logs" "$O/partial_attempts/a1/logs"
for f in n12 videos summary.json; do ln -s "$E9/$f" "$O/$f"; done
one=$(ls "$E9"/logs/server_*.log | head -1); cp "$one" "$O/logs/server_7000.log"
printf 'x\nRuntimeError: boom in an earlier attempt\n' > "$O/partial_attempts/a1/logs/server_6990.log"
python experiments/robocasa_gr1/validate_elastic_eval.py "$O" 211644 --schedule flat_k1_n2 --export-sha256 abc --json "$X/v1.json" > "$X/v1.log" 2>&1
python3 -I -c 'import json,sys; p=json.load(open(sys.argv[1]))["problems"]; sys.exit(0 if not any("boom" in x or "server_6990" in x for x in p) else 1)' "$X/v1.json" \
  && grep -q "(+1 of earlier attempts)" "$X/v1.log" && pass "validator: an earlier attempt's server error is not a problem" || fail "validator old errors"
printf 'prompt pick up the can -> t5\n' >> "$O/partial_attempts/a1/logs/server_6990.log"
python experiments/robocasa_gr1/validate_elastic_eval.py "$O" 211644 --schedule flat_k1_n2 --export-sha256 abc --json "$X/v2.json" > "$X/v2.log" 2>&1
python3 -I -c 'import json,sys; p=json.load(open(sys.argv[1]))["problems"]; sys.exit(0 if any("prompt sources" in x and "t5" in x for x in p) else 1)' "$X/v2.json" \
  && pass "validator: an earlier attempt's non-cache prompt is still a problem" || fail "validator old prompts"
printf 'RuntimeError: boom now\n' >> "$O/logs/server_7000.log"
python experiments/robocasa_gr1/validate_elastic_eval.py "$O" 211644 --schedule flat_k1_n2 --export-sha256 abc --json "$X/v3.json" > "$X/v3.log" 2>&1
python3 -I -c 'import json,sys; p=json.load(open(sys.argv[1]))["problems"]; sys.exit(0 if any("server_7000" in x for x in p) else 1)' "$X/v3.json" \
  && pass "validator: a current-attempt server error is a problem" || fail "validator current errors"
# eval_gr1e.sh server-log move (the exact lines of the script)
EO=$X/evalmove; mkdir -p "$EO/logs"; echo a > "$EO/logs/server_7200.log"; echo b > "$EO/logs/n12_task.log"
OUT=$EO; log() { :; }
eval "$(sed -n '/^if compgen -G "\$OUT\/logs\/server_\*.log"/,/^fi$/p' "$T/eval_gr1e.sh")"
[ ! -e "$EO/logs/server_7200.log" ] && [ -f "$EO/logs/n12_task.log" ] && ls "$EO"/partial_attempts/*/logs/server_7200.log > /dev/null 2>&1 \
  && pass "eval_gr1e: earlier server logs moved aside, client logs kept" || fail "eval log move"
eval "$(sed -n '/^if compgen -G "\$OUT\/logs\/server_\*.log"/,/^fi$/p' "$T/eval_gr1e.sh")"
[ "$(ls -d "$EO"/partial_attempts/* | wc -l)" = 1 ] && pass "eval_gr1e: nothing moved when no server logs" || fail "eval log move noop"
bash -n "$T/orchestrate_gr1e.sh" && bash -n "$T/validate_gr1e.sh" && pass "syntax orchestrate/validate" || fail syntax
echo "TEST_FIXES fails=$fails"
