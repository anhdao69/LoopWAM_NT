#!/usr/bin/env bash
# ws_anhdh35: RoboCasa-GR1 official evaluation of LoopWAM v0 epoch_04, then epoch_05, on the RTX PRO 6000 (GPU 0).
# Starts only after the current RoboTwin evaluation (loopwam_robotwin10_v0_2x2_20261104, incl. its finalizer) has
# finished: no process of that run directory for 5 consecutive minutes. Same code (d937167) and protocol as H100:
# starVLA Robocasa_tabletop batch_eval_args.sh settings (24 tasks x 50 episodes, n_envs 1, max_episode_steps 720,
# n_action_steps 12, videos), seed 0, EGL rendering. Policy servers under a private CUDA MPS daemon (fallback: plain).
# Inputs verified before use: epoch_0N.pt sha256 = HF upload receipt, VAE sha256 = training contract, prompt cache =
# H100 training cache (187 files, sha256 list), code revision.
set -uo pipefail
ROOT=/storage/anhdh35/robocasa_gr1_eval
Q=/data14t/anhdh35/loopwam_robotwin10_v0_2x2_20261104
log() { echo "[$(date -u +%FT%TZ)] ws-gr1: $*"; }
log "start pid $$; waiting for the RoboTwin evaluation in $Q to finish"
quiet=0
while [ "$quiet" -lt 5 ]; do
  if pgrep -u anhdh35 -f "$Q/" > /dev/null; then quiet=0; else quiet=$((quiet + 1)); fi
  sleep 60
done
log "RoboTwin evaluation finished: $(python3 -c "import json; d=json.load(open('$Q/results/summary.json')); print({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in d.items() if k in ('finished', 'running', 'failed', 'pending')})" 2>&1); COMPLETE.json: $(ls $Q/results/COMPLETE.json 2>&1)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

# inputs
cd "$ROOT/models/text_cache" && sha256sum -c --quiet "$ROOT/text_cache.sha256" && [ "$(ls | wc -l)" = 187 ] \
  || { log "prompt cache incomplete or differs from H100"; exit 1; }
cd "$ROOT"
[ "$(sha256sum models/vae/Wan2.1_VAE.pth | cut -d' ' -f1)" = 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981 ] || { log "VAE sha mismatch"; exit 1; }
[ "$(sha256sum models/Robocasa-GR1/epoch_04.pt | cut -d' ' -f1)" = b62483a91bcaec1fd5586e11ad36590043f276d6c82e5bd50c4fa55a6b018ba6 ] || { log "epoch_04 sha mismatch"; exit 1; }
[ "$(sha256sum models/Robocasa-GR1/epoch_05.pt | cut -d' ' -f1)" = 1597816e9cc1496bb9ca71a89ef02292edd80cde688198805cd978ba9949fbfa ] || { log "epoch_05 sha mismatch"; exit 1; }
[ "$(git -C code rev-parse HEAD)" = d937167ba61a0df4a9818d1917de6c7155400dff ] || { log "code revision mismatch"; exit 1; }
log "inputs verified (prompt cache 187/187, VAE, epoch_04.pt, epoch_05.pt, code d937167)"

export GR1_EVAL_ROOT=$ROOT FASTWAM_PY=/storage/anhdh35/miniconda3/envs/loopwam-libero-eval/bin/python
export ROBOCASA_GR1_TEXT_CACHE=$ROOT/models/text_cache PYTHONPATH=$ROOT/code/src:$ROOT/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true DIFFSYNTH_MODEL_BASE_PATH=$ROOT/code/checkpoints
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# EGL render smoke on this GPU (one seeded env, 60 hold steps)
( cd third_party/starVLA && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 NUMBA_DISABLE_JIT=1 \
  OMP_NUM_THREADS=2 PYTHONPATH=$PWD "$ROOT/envs/robocasa/bin/python" "$ROOT/render_bench.py" "$ROOT/logs/render_egl" PnPCanToDrawerClose ) \
  > logs/render_smoke.log 2>&1
grep BENCH logs/render_smoke.log || { log "EGL render smoke failed (logs/render_smoke.log)"; exit 1; }

# layout from what is free now: ~4.6 GB GPU per policy server, ~1.3 GB GPU + ~4 GB RAM per EGL sim client
free_mib=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
avail_gb=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
PAR=$(( (avail_gb - 24) / 4 )); [ $PAR -gt 24 ] && PAR=24
NS=$(( (free_mib - PAR * 1300 - 4096) / 4600 )); [ $NS -gt 6 ] && NS=6
log "free GPU ${free_mib} MiB, MemAvailable ${avail_gb} GB -> parallel $PAR, policy servers $NS"
[ $PAR -ge 8 ] && [ $NS -ge 2 ] || { log "not enough free memory for the evaluation"; exit 1; }

MPS=/dev/shm/gr1_mps_anhdh35_ws; rm -rf "$MPS"; mkdir -p "$MPS/pipe" "$MPS/log"
export CUDA_MPS_PIPE_DIRECTORY=$MPS/pipe CUDA_MPS_LOG_DIRECTORY=$MPS/log
if CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d; then log "MPS daemon started ($MPS)"
else log "MPS daemon failed to start: servers run time-sliced"; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; fi
stop_mps() { [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] && { echo quit | nvidia-cuda-mps-control > /dev/null 2>&1; sleep 3; rm -rf "$MPS"; }; }
trap stop_mps EXIT

clean_partial() {  # OUT: move videos of unfinished task cells aside before a (re)try
  local OUT=$1 d t dest=$1/partial_attempts/$(date -u +%Y%m%dT%H%M%SZ)
  for d in "$OUT"/videos/n12/*/; do
    [ -d "$d" ] || continue; t=$(basename "$d")
    [ -f "$OUT/n12/$t.json" ] || { mkdir -p "$dest"; mv "$d" "$dest/"; log "moved unfinished videos of $t to $dest"; }
  done
}
( while sleep 600; do log "gpu $(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader), load $(cut -d' ' -f1-3 /proc/loadavg), MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB"; done ) &
mon=$!
run_epoch() {  # E
  local E=$1 OUT=$ROOT/results/official_epoch0$1 attempt
  for attempt in 1 2 3; do
    clean_partial "$OUT"
    log "epoch $E attempt $attempt -> $OUT"
    cd "$ROOT/code"
    nice -n 5 "$FASTWAM_PY" experiments/robocasa_gr1/run_eval.py --checkpoint "$ROOT/models/Robocasa-GR1/epoch_0$E.pt" \
      --stats "$ROOT/models/Robocasa-GR1/dataset_stats.json" --out "$OUT" --steps 12 --episodes 50 --n-envs 1 \
      --max-episode-steps 720 --seed 0 --gpus 0 --servers-per-gpu "$NS" --base-port $((6500 + 20 * (E - 4))) --parallel "$PAR" \
      --server-threads 4 --sim-threads 2 --video --render egl --vae-path "$ROOT/models/vae/Wan2.1_VAE.pth" \
      --expected-step $((E * 23516)) > "$ROOT/logs/official_epoch0${E}_driver.log" 2>&1
    log "epoch $E attempt $attempt rc=$?"
    python3 -c "import json,sys; d=json.load(open(sys.argv[1]))['by_n_action_steps']['12']; print(json.dumps(dict(tasks=d['tasks'], mean=d['mean_success']))); sys.exit(0 if d['complete'] else 1)" "$OUT/summary.json" && { log "epoch $E complete"; return 0; }
    sleep 90
  done
  log "epoch $E not complete after 3 attempts"; return 1
}
run_epoch 4; r4=$?
run_epoch 5; r5=$?
kill $mon 2>/dev/null
log "done: epoch 4 rc=$r4, epoch 5 rc=$r5"
