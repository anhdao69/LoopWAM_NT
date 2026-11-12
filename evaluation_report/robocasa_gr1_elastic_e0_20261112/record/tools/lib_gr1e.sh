#!/usr/bin/env bash
# Helper functions of orchestrate_gr1e.sh (sourced; also by selftest_gr1e.sh). They read the variables the orchestrator
# sets: B (record dir), RUN, OUT (run dir), V (validation dir), EV (eval dir), PER, EPOCHS, GR1_CONT.
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$B/orchestrator.log"; }
status() {
  python3 -I -c 'import json,sys,time; json.dump(dict(phase=sys.argv[1], detail=sys.argv[2], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(sys.argv[3], "w"), indent=1)' \
    "$1" "$2" "$B/STATUS.json.tmp" && mv "$B/STATUS.json.tmp" "$B/STATUS.json"
  log "STATUS $1: $2"
}
lock_free() { [ ! -e "$1" ] || ( flock -n 9 || exit 1; exit 0 ) 9>>"$1"; }
cont_phase() { python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "$GR1_CONT/STATUS.json" 2>/dev/null; }
# The GR1 continuation (epochs 6-10 + evals) is over: final phase, and its orchestrator no longer holds its lock.
cont_finished() { case "$(cont_phase)" in done|failed) lock_free /tmp/gr1cont_orchestrator_5758.lock ;; *) return 1 ;; esac; }
validated() {
  python3 -I - "$B/prelaunch" <<'PY'
import json, sys
from pathlib import Path
v = Path(sys.argv[1])
try: lines = (v / "validate.rc").read_text().split()
except Exception: sys.exit(2)                       # not run yet
if "VALIDATE_DONE" not in lines: sys.exit(2)
rcs = dict(l.split(" rc=") for l in (v / "validate.rc").read_text().splitlines() if " rc=" in l)
need = ["preflight_core", "preflight_speed_E0", "resume_A", "resume_B1", "resume_B2", "resume_compare", "schedule_contract",
        "verify_early", "verify_epoch", "eval_smoke"]
ok = all(rcs.get(n) == "0" for n in need)
for f in ("preflight_core.json", "preflight_speed_E0.json", "verify_early.json", "verify_epoch.json", "eval_smoke.json"):
    try: ok &= json.loads((v / f).read_text()).get("passed") is True
    except Exception: ok = False
sys.exit(0 if ok else 1)
PY
}
epoch_complete() {  # E: status.json at the end of epoch E with its export (and the update-2000 pair for epoch 1)
  python3 -I - "$OUT" "$1" "$PER" "$EPOCHS" <<'PY'
import json, sys
from pathlib import Path
out, e, per, epochs = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
try: s = json.loads((out / "status.json").read_text())
except Exception: sys.exit(1)
if s.get("update", 0) > e * per: sys.exit(3)        # trained past the end of epoch E
ok = s.get("update") == e * per and s.get("epoch") == e and s.get("next_micro") == 0 \
    and s.get("status") == ("complete" if e == epochs else "running") and (out / f"epoch_{e:02d}.pt").is_file()
if e == 1: ok &= all((out / f).is_file() for f in ("step_00002000.pt", "step_00002000_recovery.pt"))
sys.exit(0 if ok else 1)
PY
}
passed() { python3 -I -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("passed") is True else 1)' "$1" 2>/dev/null; }
verified() { passed "$V/verify_$1.json"; }
eval_valid() { python3 -I -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("valid") is True else 1)' "$1" 2>/dev/null; }
alive() { [ -d "$OUT" ] || return 1; ( flock -n 9 || exit 0; exit 1 ) 9>>"$OUT/run.lock"; }
last_write() { stat -c %Y "$OUT/metrics.jsonl" "$OUT/train.log" 2>/dev/null | sort -n | tail -1; }
attempts() { cat "$B/attempts_$1" 2>/dev/null || echo 0; }
keep_state() {  # E: epoch_EE_recovery.pt = byte-identical copy of the epoch-end latest.pt
  local e=$1 f; f=$(printf 'epoch_%02d_recovery.pt' "$e")
  if [ -f "$OUT/$f" ] && cmp -s "$OUT/$f" "$OUT/latest.pt"; then return 0; fi
  [ -f "$OUT/$f" ] && return 1                       # an existing different copy is never overwritten
  cp "$OUT/latest.pt" "$OUT/$f.tmp" && cmp -s "$OUT/$f.tmp" "$OUT/latest.pt" && mv "$OUT/$f.tmp" "$OUT/$f"
}
