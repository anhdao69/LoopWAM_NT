#!/usr/bin/env bash
# ws_anhdh35: RoboCasa-GR1 epoch_04 sweep over n_action_steps (env steps executed per policy query) 16, 24, 32, 8.
# The model always predicts its trained 32-action chunk; only the executed prefix length changes. Everything else is
# the official protocol of ws_gr1_eval_v2.sh / ws_gr1_eval_e3.sh (code d937167, 24 tasks x 50 episodes, n_envs 1,
# max_episode_steps 720, seed 0, videos, EGL, policy servers under a private MPS daemon, layout from free memory).
# n_action_steps 12 is the existing official run results/official_epoch04 (46.25%). Shares the GPU with the user's
# SpatialForcing-VLN evaluation. Each n is validated with gr1_validate_n.py.
set -uo pipefail
ROOT=/storage/anhdh35/robocasa_gr1_eval
STEPS=16,24,32,8
OUT=$ROOT/results/sweep_epoch04
log() { echo "[$(date -u +%FT%TZ)] ws-gr1-sweep-e4: $*"; }
log "start pid $$, steps $STEPS -> $OUT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

cd "$ROOT/models/text_cache" && sha256sum -c --quiet "$ROOT/text_cache.sha256" && [ "$(ls | wc -l)" = 187 ] \
  || { log "prompt cache incomplete or differs from H100"; exit 1; }
cd "$ROOT"
[ "$(sha256sum models/vae/Wan2.1_VAE.pth | cut -d' ' -f1)" = 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981 ] || { log "VAE sha mismatch"; exit 1; }
[ "$(sha256sum models/Robocasa-GR1/epoch_04.pt | cut -d' ' -f1)" = b62483a91bcaec1fd5586e11ad36590043f276d6c82e5bd50c4fa55a6b018ba6 ] || { log "epoch_04 sha mismatch"; exit 1; }
[ "$(git -C code rev-parse HEAD)" = d937167ba61a0df4a9818d1917de6c7155400dff ] || { log "code revision mismatch"; exit 1; }
log "inputs verified (prompt cache 187/187, VAE, epoch_04.pt, code d937167)"

export GR1_EVAL_ROOT=$ROOT FASTWAM_PY=/storage/anhdh35/miniconda3/envs/loopwam-libero-eval/bin/python
export ROBOCASA_GR1_TEXT_CACHE=$ROOT/models/text_cache PYTHONPATH=$ROOT/code/src:$ROOT/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true DIFFSYNTH_MODEL_BASE_PATH=$ROOT/code/checkpoints
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

( cd third_party/starVLA && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 NUMBA_DISABLE_JIT=1 \
  OMP_NUM_THREADS=2 PYTHONPATH=$PWD "$ROOT/envs/robocasa/bin/python" "$ROOT/render_bench.py" "$ROOT/logs/render_egl_sweep" PnPCanToDrawerClose ) \
  > logs/render_smoke_sweep.log 2>&1
grep BENCH logs/render_smoke_sweep.log || { log "EGL render smoke failed (logs/render_smoke_sweep.log)"; exit 1; }

free_mib=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
avail_gb=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
PAR=$(( (avail_gb - 24) / 4 )); [ $PAR -gt 24 ] && PAR=24
NS=$(( (free_mib - PAR * 1300 - 4096) / 4600 )); [ $NS -gt 6 ] && NS=6
log "free GPU ${free_mib} MiB, MemAvailable ${avail_gb} GB -> parallel $PAR, policy servers $NS"
[ $PAR -ge 8 ] && [ $NS -ge 2 ] || { log "not enough free memory for the evaluation"; exit 1; }

MPS=/dev/shm/gr1_mps_anhdh35_ws_sweep; rm -rf "$MPS"; mkdir -p "$MPS/pipe" "$MPS/log"
export CUDA_MPS_PIPE_DIRECTORY=$MPS/pipe CUDA_MPS_LOG_DIRECTORY=$MPS/log
if CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d; then log "MPS daemon started ($MPS)"
else log "MPS daemon failed to start: servers run time-sliced"; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; fi
stop_mps() { [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] && { echo quit | nvidia-cuda-mps-control > /dev/null 2>&1; sleep 3; rm -rf "$MPS"; }; }
trap stop_mps EXIT

clean_partial() {  # move videos of unfinished (n, task) cells aside before a (re)try
  local d t n dest=$OUT/partial_attempts/$(date -u +%Y%m%dT%H%M%SZ)
  for d in "$OUT"/videos/n*/*/; do
    [ -d "$d" ] || continue; t=$(basename "$d"); n=$(basename "$(dirname "$d")")
    [ -f "$OUT/$n/$t.json" ] || { mkdir -p "$dest/$n"; mv "$d" "$dest/$n/"; log "moved unfinished videos of $n/$t to $dest/$n"; }
  done
}
( while sleep 600; do log "cells done $(ls "$OUT"/n*/*.json 2>/dev/null | wc -l)/96, gpu $(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader), load $(cut -d' ' -f1-3 /proc/loadavg), MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GB"; done ) &
mon=$!
rc=1
for attempt in 1 2 3; do
  clean_partial
  log "attempt $attempt"
  cd "$ROOT/code"
  nice -n 5 "$FASTWAM_PY" experiments/robocasa_gr1/run_eval.py --checkpoint "$ROOT/models/Robocasa-GR1/epoch_04.pt" \
    --stats "$ROOT/models/Robocasa-GR1/dataset_stats.json" --out "$OUT" --steps $STEPS --episodes 50 --n-envs 1 \
    --max-episode-steps 720 --seed 0 --gpus 0 --servers-per-gpu "$NS" --base-port 6560 --parallel "$PAR" \
    --server-threads 4 --sim-threads 2 --video --render egl --vae-path "$ROOT/models/vae/Wan2.1_VAE.pth" \
    --expected-step 94064 >> "$ROOT/logs/sweep_epoch04_driver.log" 2>&1
  log "attempt $attempt rc=$?"
  python3 -c "import json,sys; t=json.load(open(sys.argv[1]))['by_n_action_steps']; print(json.dumps({k: (v['tasks'], v['mean_success']) for k, v in t.items()})); sys.exit(0 if all(v['complete'] for v in t.values()) and len(t) == 4 else 1)" "$OUT/summary.json" && { log "all cells complete"; rc=0; break; }
  sleep 90
done
kill $mon 2>/dev/null
[ $rc -eq 0 ] || { log "sweep not complete after 3 attempts"; exit 1; }
cd "$ROOT"
for n in ${STEPS//,/ }; do
  python3 gr1_validate_n.py "$OUT" 94064 --n $n --json "$OUT/validation_n$n.json" > "$OUT/validation_n$n.txt" 2>&1
  log "n_action_steps $n validation rc=$?: $(head -1 "$OUT/validation_n$n.txt" | sed 's/.*mean success/mean success/') / $(tail -1 "$OUT/validation_n$n.txt")"
done
log "done"
