#!/usr/bin/env bash
# RAM guard for the 2026-11-01 matched runs on the shared host (elastic lanes wam-elastic-* and the matched-harness
# run wam-matched-loopwam_v0_v4a2). When MemAvailable stays below GUARD_GB for three checks 10 s apart, the newest
# active unit is paused: an elastic lane is stopped as a whole (systemctl stop; autoscale.sh starts lanes again when
# RAM returns and releases the dead runner's claims), the harness run by SIGINT to its run_matched.py runner (its own
# cleanup; its wrapper waits for RAM and resumes). Complete groups are kept; interrupted groups rerun from episode 0 with
# the same seed. At most one pause per 10 minutes.
set -uo pipefail
ROOT=/data/anh/code/wam_baselines
LOG=$ROOT/logs/ram_guard_20261101.log
GUARD_GB=${GUARD_GB:-3}
GUARD_MB=${GUARD_MB:-$((GUARD_GB * 1024))}  # decision threshold in MB (whole-GB readings flap around the line)
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
avail_gb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
newest_unit() {
  local u t best= bt=0
  for u in $(systemctl --user list-units 'wam-elastic-*' 'wam-matched-*' --state=active --no-legend --plain | awk '{print $1}'); do
    t=$(systemctl --user show "$u" -p ActiveEnterTimestampMonotonic --value)
    [ "${t:-0}" -gt "$bt" ] && { bt=$t; best=$u; }
  done
  echo "$best"
}
pause() {
  local unit=$1 cg pid n=0
  cg=$(systemctl --user show "$unit" -p ControlGroup --value); [ -n "$cg" ] || return 1
  for pid in $(cat "/sys/fs/cgroup$cg/cgroup.procs" 2>/dev/null); do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)
    case "$unit" in
      wam-matched-*) echo "$cmd" | grep -q '/run_matched\.py --family ' && { kill -INT "$pid"; n=$((n + 1)); } ;;
    esac
  done
  case "$unit" in wam-elastic-*) systemctl --user stop "$unit" && n=1 ;; esac
  [ "$n" -gt 0 ]
}
say "ram guard started (pause the newest run when MemAvailable < ${GUARD_MB} MB for 30 s)"
low=0
while true; do
  if [ "$(avail_mb)" -lt "$GUARD_MB" ]; then low=$((low + 1)); else low=0; fi
  if [ "$low" -ge 3 ]; then
    unit=$(newest_unit)
    if [ -n "$unit" ] && pause "$unit"; then
      say "ram guard: MemAvailable $(avail_gb) GB for 30 s; paused $unit"
      sleep 600
    fi
    low=0
  fi
  sleep 10
done
