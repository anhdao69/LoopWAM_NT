#!/bin/bash
# RoboCasa-GR1 LoopWAM v0: epochs 6-10 continuation + official eval of every epoch, fully automatic on worker-2 in job 5758
# (GPUs 4-7 and their NUMA-1 CPUs only; GPUs 0-3 are never used). Requested 2026-11-06: start after the RoboTwin-10
# v0-video4-action2 run finishes; strictly 4 GPUs; pause training at each epoch end for the eval; lr cosine 3e-5 -> 3e-7.
#   1. wait until RoboTwin-10 v0-video4-action2 (GPUs 4-7) is COMPLETE and its supervisor has exited
#   2. stage the GR1 latent cache backup into /tmp (verified, stage_cache.py) once /tmp space and job memory allow
#   3. gates, once: 20-update continuation from the parent + 10-update resume (gate_check.py); stop if they fail
#   4. for E in 6..10: train to the end of epoch E (--max-updates E*23516 saves epoch_E + latest and exits; epoch 10
#      runs to COMPLETE), up to 4 attempts resuming from latest.pt; then the official eval of epoch_E on the same GPUs
#      (eval_epoch.sh, up to 3 attempts), gr1_validate.py, results table eval/official_results_continuation.md
#   5. remove the RAM caches it no longer needs (RoboTwin-10 once both RoboTwin-10 runs are COMPLETE; GR1 after training)
# Every phase is decided from files, so relaunching the same command continues where it stopped.
# Usage (2-CPU step):  orchestrate.sh   |   orchestrate.sh --check   (read-only preflight: prints the plan, changes nothing)
set -u
G=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101
CR=$G/continuation_20261106; OUT=$G/train/continuation_e06_10; EV=$G/eval
RT=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin10_v0_20261105
LAT=/tmp/gr1_cont_latents_20261106/latents; BK=$G/shared/latents_backup
RT10_CACHE=/tmp/rt10_latents_20261105; RT10_VERIFY=/tmp/rt10_verify_20261105
T=/mnt/data/vmo-ai-task/anhdh35/gr1_eval/tools
JOB=5758; U=23516; P=117580
CHECK=0; [ "${1:-}" = --check ] && CHECK=1
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
CGJ=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | sed -E 's#(/job_[0-9]+).*#\1#')

log() { if [ "$CHECK" = 1 ]; then echo "$*"; else echo "[$(date -u +%FT%TZ)] $*" >> "$CR/orchestrator.log"; fi; }
status() {  # PHASE DETAIL
  [ "$CHECK" = 1 ] && { echo "STATUS $1: $2"; return; }
  python3 -I -c 'import json,sys,time; json.dump(dict(phase=sys.argv[1], detail=sys.argv[2], utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(sys.argv[3], "w"), indent=1)' \
    "$1" "$2" "$CR/STATUS.json.tmp" && mv "$CR/STATUS.json.tmp" "$CR/STATUS.json"
  log "STATUS $1: $2"
}
pyenv() { ( source "$CR/env.sh" > /dev/null 2>&1; "$@" ); }     # FastWAM venv + continuation code on PYTHONPATH
step() { srun --jobid="$JOB" --overlap --ntasks=1 --job-name=training_vln "$@"; }
numa1_cpus() {
  python3 -I - "$(cat "$CGJ/cpuset.cpus.effective")" /sys/devices/system/node/node1/cpulist <<'PY'
import sys
def parse(s):
    out = []
    for part in s.strip().split(','):
        a, _, b = part.partition('-'); out += range(int(a), int(b or a) + 1)
    return out
print(','.join(map(str, sorted(set(parse(sys.argv[1])) & set(parse(open(sys.argv[2]).read()))))))
PY
}
rt_done() { [ -e "$RT/$1/train/COMPLETE" ] && grep -q "$1: training complete" "$RT/$1/supervisor.log" 2>/dev/null; }
ck_step() { pyenv python "$CR/tools/ckpt_info.py" "$1" 2> /dev/null | awk 'NF == 6 {print $1}'; }

maybe_cleanup_rt10() {
  [ -d "$RT10_CACHE" ] || [ -d "$RT10_VERIFY" ] || return 0
  rt_done v0-video4-action4 && rt_done v0-video4-action2 || return 0
  pgrep -f train_loopwam_robotwin10.py > /dev/null && return 0
  [ "$CHECK" = 1 ] && { echo "would remove $RT10_CACHE $RT10_VERIFY"; return 0; }
  log "both RoboTwin-10 runs COMPLETE: removing their RAM caches $RT10_CACHE $RT10_VERIFY (re-derivable: $RT/tools/derive_rt10_cache.py)"
  rm -rf --one-file-system "$RT10_CACHE" "$RT10_VERIFY"
}

wait_rt42() {
  local n=0
  until rt_done v0-video4-action2; do
    [ "$CHECK" = 1 ] && { echo "RoboTwin-10 v0-video4-action2 not complete yet: $(tr -d '\n ' < "$RT/v0-video4-action2/train/timing.json" | cut -c1-120)"; return 0; }
    [ $((n % 30)) = 0 ] && status waiting_for_robotwin10 "v0-video4-action2 $(tr -d '\n ' < "$RT/v0-video4-action2/train/timing.json" 2>/dev/null | cut -c1-120) | supervisor: $(tail -1 "$RT/v0-video4-action2/supervisor.log" | cut -c1-160)"
    n=$((n + 1)); sleep 120
  done
  log "RoboTwin-10 v0-video4-action2 complete: $(tr -d '\n ' < "$RT/v0-video4-action2/train/COMPLETE")"
}

stage() {
  local need free cur max attempt n=0
  need=$(( $(stat -c %s "$BK/latents.uint16") + $(stat -c %s "$BK/valid.uint8") ))
  # already staged (relaunch): the trainer re-checks the cache metadata, sizes and validity mask on every launch
  if [ -f "$LAT/STAGED.json" ]; then log "GR1 latent cache already staged ($LAT): $(tr -d '\n ' < "$LAT/STAGED.json" | cut -c1-200)"; return 0; fi
  while :; do
    maybe_cleanup_rt10
    free=$(df -B1 --output=avail /tmp | tail -1); cur=$(cat "$CGJ/memory.current"); max=$(cat "$CGJ/memory.max")
    [ "$max" = max ] && max=$((1 << 50))
    if [ "$free" -gt $((need + (40 << 30))) ] && [ $((cur + need + (160 << 30))) -lt "$max" ]; then
      log "staging check ok: /tmp free $((free >> 30)) GiB, job memory $((cur >> 30))/$((max >> 30)) GiB, cache $((need >> 30)) GiB"; break
    fi
    [ "$CHECK" = 1 ] && { echo "staging would WAIT: /tmp free $((free >> 30)) GiB, job memory $((cur >> 30))/$((max >> 30)) GiB, cache $((need >> 30)) GiB"; return 0; }
    [ $((n % 12)) = 0 ] && status waiting_for_memory "/tmp free $((free >> 30)) GiB, job memory $((cur >> 30))/$((max >> 30)) GiB, cache $((need >> 30)) GiB"
    n=$((n + 1)); sleep 300
  done
  [ "$CHECK" = 1 ] && return 0
  for attempt in 1 2 3; do
    status staging_cache "attempt $attempt: $BK -> $LAT"
    # -c120: the step cpuset must contain the NUMA-1 CPUs taskset pins to (a -c16 step cpuset does not: taskset EINVAL)
    step -c120 --cpu-bind=none --gres=none bash -c "source $CR/env.sh > /dev/null 2>&1; taskset -c $(numa1_cpus) python -u $CR/tools/stage_cache.py $BK $LAT --threads 16" \
      >> "$CR/logs/stage_cache.log" 2>&1
    [ -f "$LAT/STAGED.json" ] && { log "cache staged: $(tr -d '\n ' < "$LAT/STAGED.json" | cut -c1-300)"; return 0; }
    sleep 300
  done
  status failed "latent cache staging failed 3 times (logs/stage_cache.log)"; exit 1
}

run_train() {  # OUTDIR TAG [--resume] [--max-updates N]
  local out=$1 tag=$2 A; shift 2
  mapfile -t A < <(pyenv python "$CR/tools/gr1cont_args.py" "$out" "$LAT" "$@")
  [ "${#A[@]}" -gt 30 ] || { log "argument generation failed for $tag"; return 9; }
  if [ "$CHECK" = 1 ]; then echo "PLAN $tag: launch_train.sh $tag ${A[*]}"; return 0; fi
  log "launch $tag: ${A[*]}"
  step -c120 --cpu-bind=none --gres=gpu:8 --kill-on-bad-exit=1 bash "$CR/tools/launch_train.sh" "$tag" "${A[@]}" > "$CR/logs/$tag.log" 2>&1
}

gates() {
  local GO=$CR/gates/cont_gate/train try rc
  [ -f "$CR/gates/PASSED.json" ] && { log "gates already passed"; return 0; }
  if [ "$CHECK" = 1 ]; then run_train "$GO" gate1 --max-updates $((P + 20)); run_train "$GO" gate2 --resume --max-updates $((P + 30)); return 0; fi
  for try in 1 2; do
    [ -e "$GO" ] && mv "$GO" "$CR/gates/cont_gate/train.try_$(date -u +%Y%m%dT%H%M%SZ)"
    status gates "try $try: 20-update continuation from the parent, then a 10-update resume"
    run_train "$GO" gate1_t$try --max-updates $((P + 20)); rc=$?; log "gate1 try $try rc=$rc"
    [ "$rc" = 0 ] || { sleep 120; continue; }
    pyenv python "$CR/tools/gate_check.py" "$GO" $((P + 20)) --report "$CR/gates/gate1.json" >> "$CR/logs/gate_check.log" 2>&1 \
      || { status failed "gate 1 (continuation start) failed: gates/gate1.json"; exit 1; }
    run_train "$GO" gate2_t$try --resume --max-updates $((P + 30)); rc=$?; log "gate2 try $try rc=$rc"
    [ "$rc" = 0 ] || { sleep 120; continue; }
    pyenv python "$CR/tools/gate_check.py" "$GO" $((P + 30)) --resume-from $((P + 20)) --report "$CR/gates/gate2.json" >> "$CR/logs/gate_check.log" 2>&1 \
      || { status failed "gate 2 (resume) failed: gates/gate2.json"; exit 1; }
    rm -f "$GO/latest.pt"
    python3 -I -c 'import json,sys; json.dump(dict(passed=True, gate1=json.load(open(sys.argv[1])), gate2=json.load(open(sys.argv[2]))), open(sys.argv[3], "w"), indent=1)' \
      "$CR/gates/gate1.json" "$CR/gates/gate2.json" "$CR/gates/PASSED.json"
    log "gates passed: $(python3 -I -c 'import json,sys; g=json.load(open(sys.argv[1])); print(json.dumps({k: g[k] for k in ("loss_mean","grad_max","first_lr","median_seconds_after5")}))' "$CR/gates/gate2.json")"
    return 0
  done
  status failed "gate training launches failed twice (logs/gate1_t*.log, gate2_t*.log)"; exit 1
}

epoch_trained() {  # E: epoch_E.pt at step E*U, and the run paused at/after it (latest.pt) or COMPLETE for epoch 10
  local E=$1 EE target; EE=$(printf %02d "$1"); target=$(($1 * U))
  [ -f "$OUT/checkpoints/epoch_$EE.pt" ] && [ "$(ck_step "$OUT/checkpoints/epoch_$EE.pt")" = "$target" ] || return 1
  if [ "$E" = 10 ]; then [ -e "$OUT/COMPLETE" ]; else local s; s=$(ck_step "$OUT/latest.pt"); [ -n "$s" ] && [ "$s" -ge "$target" ]; fi
}

train_epoch() {
  local E=$1 EE target attempt rc R M; EE=$(printf %02d "$1"); target=$(($1 * U))
  for attempt in 1 2 3 4; do
    epoch_trained "$E" && { log "epoch $E trained (epoch_$EE.pt at update $target)"; return 0; }
    if [ -e "$OUT/latest.pt" ]; then R=(--resume)
    elif [ -e "$OUT/manifest.json" ] || [ -e "$OUT/metrics.jsonl" ]; then status failed "records in $OUT without latest.pt: stopped for inspection"; exit 2
    else R=(); fi
    M=(); [ "$E" -lt 10 ] && M=(--max-updates "$target")
    if [ "$CHECK" = 1 ]; then [ "$E" -gt 6 ] && R=(--resume); run_train "$OUT" "train_e${EE}" ${R[@]+"${R[@]}"} ${M[@]+"${M[@]}"}; return 0; fi
    status training "epoch $E (to update $target), attempt $attempt ${R[*]:-fresh continuation}"
    run_train "$OUT" "train_e${EE}_a$attempt" ${R[@]+"${R[@]}"} ${M[@]+"${M[@]}"}; rc=$?
    log "epoch $E attempt $attempt rc=$rc timing $(tr -d '\n ' < "$OUT/timing.json" 2>/dev/null | cut -c1-200)"
    epoch_trained "$E" && { log "epoch $E trained (epoch_$EE.pt at update $target)"; return 0; }
    [ "$attempt" -lt 4 ] && sleep 300
  done
  return 1
}

report() {
  local args=() e
  for e in 04 05; do [ -f "$EV/official_epoch${e}_validation.json" ] && args+=("H100 ep${e#0}=$EV/official_epoch${e}_validation.json"); done
  for e in 06 07 08 09 10; do [ -f "$EV/official_epoch${e}_validation.json" ] && args+=("ep${e#0}=$EV/official_epoch${e}_validation.json"); done
  python3 "$T/gr1_report.py" "$EV/official_results_continuation.md" ${args[@]+"${args[@]}"} > /dev/null 2>&1 || log "report generation failed"
}

eval_epoch() {
  local E=$1 EE target attempt rc; EE=$(printf %02d "$1"); target=$(($1 * U))
  [ -f "$EV/official_epoch${EE}_validation.json" ] && { log "epoch $E already evaluated"; return 0; }
  [ "$CHECK" = 1 ] && { echo "PLAN eval_e$EE: eval_epoch.sh $E 1 -> $EV/official_epoch$EE, then gr1_validate.py $EV/official_epoch$EE $target"; return 0; }
  for attempt in 1 2 3; do
    status eval "epoch $E (epoch_$EE.pt, step $target), attempt $attempt"
    timeout --signal=TERM --kill-after=300 8h srun --jobid="$JOB" --overlap --ntasks=1 --job-name=training_vln -c120 --cpu-bind=none --gres=gpu:8 \
      bash "$CR/tools/eval_epoch.sh" "$E" "$attempt" > "$CR/logs/eval_e${EE}_a$attempt.log" 2>&1; rc=$?
    log "eval epoch $E attempt $attempt rc=$rc: $(tail -1 "$CR/logs/eval_e${EE}_a$attempt.log" | cut -c1-200)"
    if [ "$rc" = 0 ]; then
      python3 "$T/gr1_validate.py" "$EV/official_epoch$EE" "$target" --json "$EV/official_epoch${EE}_validation.json" > "$CR/logs/validate_e$EE.log" 2>&1
      log "validation epoch $E rc=$?: $(grep -E 'mean success' "$CR/logs/validate_e$EE.log" | cut -c1-200) $(tail -1 "$CR/logs/validate_e$EE.log" | cut -c1-120)"
      report; return 0
    fi
    sleep 120
  done
  status eval_failed "epoch $E eval failed 3 times (logs/eval_e${EE}_a*.log); training continues"
  return 1
}

preflight() {
  echo "== preflight $(date -u +%FT%TZ) host $(hostname) job cgroup $CGJ"
  echo "numa1 cpus: $(numa1_cpus)"
  for f in env.sh tools/gr1cont_args.py tools/gate_check.py tools/ckpt_info.py tools/stage_cache.py tools/launch_train.sh tools/bind_gpu.sh tools/eval_epoch.sh; do
    [ -f "$CR/$f" ] && echo "ok $f" || echo "MISSING $f"; done
  for f in gr1_validate.py gr1_report.py; do [ -f "$T/$f" ] && echo "ok $T/$f" || echo "MISSING $T/$f"; done
  echo "parent latest.pt: $(pyenv python "$CR/tools/ckpt_info.py" "$G/train/production/latest.pt")"
  echo "eval code: $(git -C /mnt/data/vmo-ai-task/anhdh35/LoopWAM_gr1_eval_code rev-parse --short HEAD); train code: $(git -C /mnt/data/vmo-ai-task/anhdh35/LoopWAM_gr1cont_20261106 rev-parse --short HEAD)"
  echo "rt10 4/2 done: $(rt_done v0-video4-action2 && echo yes || echo no); rt10 4/4 done: $(rt_done v0-video4-action4 && echo yes || echo no)"
  echo "output dir: $(ls "$OUT" 2>&1 | tr '\n' ' ')"
}

main() {
  if [ "$CHECK" = 1 ]; then
    preflight; wait_rt42; stage; gates
    for E in 6 7 8 9 10; do train_epoch "$E"; eval_epoch "$E"; done
    maybe_cleanup_rt10; return 0
  fi
  exec 9> /tmp/gr1cont_orchestrator_$JOB.lock
  flock -n 9 || { echo "another orchestrator is running"; exit 1; }
  ( while kill -0 $$ 2> /dev/null; do touch "$CR/heartbeat"; sleep 60; done ) 9>&- &   # liveness for keeper.sh (login node)
  HB=$!; trap 'kill $HB 2> /dev/null' EXIT
  log "orchestrator start: job $JOB step ${SLURM_STEP_ID:-?} host $(hostname) pid $$"
  wait_rt42
  stage
  gates
  for E in 6 7 8 9 10; do
    train_epoch "$E" || { status failed "training epoch $E failed 4 attempts (logs/train_e$(printf %02d "$E")_a*.log)"; exit 1; }
    if [ "$E" = 10 ] && [ -e "$OUT/COMPLETE" ]; then
      log "training COMPLETE: $(tr -d '\n ' < "$OUT/COMPLETE"); removing the GR1 RAM cache $LAT (backup stays in $BK)"
      rm -rf --one-file-system /tmp/gr1_cont_latents_20261106
    fi
    eval_epoch "$E"
    maybe_cleanup_rt10
  done
  report
  status done "epochs 6-10 trained and evaluated; results: $EV/official_results_continuation.md"
}

main "$@"
