#!/usr/bin/env bash
# launcher_gr1e_e5.sh: launcher_gr1e.sh for orchestrate_gr1e_e5.sh (2026-11-10); otherwise identical.
# Waiter in front of orchestrate_gr1e.sh (2-CPU step of job 5758, -J training_vln; relaunched by keeper_gr1e.sh).
# Waits until the GR1 continuation is finished (STATUS done/failed, its orchestrator lock free), then execs the
# orchestrator with the revision pinned in $EGR_WORK/REVISION. Nothing of the orchestrator is read before then, so its
# script and the pinned revision can still be corrected while this waits (never edit a script that is running).
set -u
B=/mnt/data/vmo-ai-task/anhdh35/elastic_gr1; T=$B/tools
GR1_CONT=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101/continuation_20261106
exec 7>/tmp/egr_gr1e_launcher.lock
flock -n 7 || { echo "[$(date -u +%FT%TZ)] launcher: another launcher is running on $(hostname)" >> "$B/orchestrator.log"; exit 0; }
log() { echo "[$(date -u +%FT%TZ)] launcher: $*" >> "$B/orchestrator.log"; }
lock_free() { [ ! -e "$1" ] || ( flock -n 9 || exit 1; exit 0 ) 9>>"$1"; }
cont_finished() {
  case "$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["phase"])' "$GR1_CONT/STATUS.json" 2>/dev/null)" in
    done|failed) lock_free /tmp/gr1cont_orchestrator_5758.lock ;; *) return 1 ;; esac
}
status_waiting() {
  python3 -I -c 'import json,sys,time; json.dump(dict(phase="waiting", detail=sys.argv[1], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(sys.argv[2], "w"), indent=1)' \
    "$1" "$B/STATUS.json.tmp" && mv "$B/STATUS.json.tmp" "$B/STATUS.json"
}
( exec 7>&-; while true; do touch "$B/heartbeat"; sleep 60; done ) & BEAT=$!
log "launcher start: host $(hostname) job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?} pid $$"
if ! cont_finished; then
  status_waiting "for the GR1 continuation (epoch 10 training + eval) to finish on GPUs 4-7"
  until cont_finished; do sleep 60; done
  log "GR1 continuation finished: $(tr -d '\n' < "$GR1_CONT/STATUS.json")"
fi
REV=$(tr -d ' \n' < "$B/REVISION")
log "starting orchestrate_gr1e_e5.sh with revision $REV (sha256 $(sha256sum "$T/orchestrate_gr1e_e5.sh" | cut -c1-16))"
kill "$BEAT" 2>/dev/null; wait "$BEAT" 2>/dev/null
exec 7>&-
exec bash "$T/orchestrate_gr1e_e5.sh" "$REV"
