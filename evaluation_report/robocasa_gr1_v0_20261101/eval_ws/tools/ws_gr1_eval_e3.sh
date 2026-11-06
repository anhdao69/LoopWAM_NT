#!/usr/bin/env bash
# ws_anhdh35: RoboCasa-GR1 official evaluation of LoopWAM v0 epoch_03 (update 70,548) on the RTX PRO 6000 (GPU 0),
# started 2026-11-04 next to a running SpatialForcing-VLN evaluation (user's choice: share the GPU).
# Same code (d937167), protocol, layout rule and retry logic as ws_gr1_eval_v2.sh (epochs 4/5): starVLA
# Robocasa_tabletop settings (24 tasks x 50 episodes, n_envs 1, max_episode_steps 720, n_action_steps 12, videos),
# seed 0, EGL rendering, policy servers under a private CUDA MPS daemon. Validated afterwards with gr1_validate.py.
set -uo pipefail
ROOT=/storage/anhdh35/robocasa_gr1_eval
E3_SHA=a8d7ec33913d1b93a0205c14b3031ee1bb373a836e3385e3bd7f30994b2d76a6
log() { echo "[$(date -u +%FT%TZ)] ws-gr1-e3: $*"; }
log "start pid $$"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

# inputs
cd "$ROOT/models/text_cache" && sha256sum -c --quiet "$ROOT/text_cache.sha256" && [ "$(ls | wc -l)" = 187 ] \
  || { log "prompt cache incomplete or differs from H100"; exit 1; }
cd "$ROOT"
[ "$(sha256sum models/vae/Wan2.1_VAE.pth | cut -d' ' -f1)" = 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981 ] || { log "VAE sha mismatch"; exit 1; }
[ "$(sha256sum models/Robocasa-GR1/epoch_03.pt | cut -d' ' -f1)" = $E3_SHA ] || { log "epoch_03 sha mismatch"; exit 1; }
[ "$(python3 -c "import json;print(json.load(open('models/Robocasa-GR1/epoch_03.pt.uploaded.json'))['sha256'])")" = $E3_SHA ] || { log "epoch_03 receipt mismatch"; exit 1; }
[ "$(git -C code rev-parse HEAD)" = d937167ba61a0df4a9818d1917de6c7155400dff ] || { log "code revision mismatch"; exit 1; }
log "inputs verified (prompt cache 187/187, VAE, epoch_03.pt = HF receipt, code d937167)"

export GR1_EVAL_ROOT=$ROOT FASTWAM_PY=/storage/anhdh35/miniconda3/envs/loopwam-libero-eval/bin/python
export ROBOCASA_GR1_TEXT_CACHE=$ROOT/models/text_cache PYTHONPATH=$ROOT/code/src:$ROOT/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true DIFFSYNTH_MODEL_BASE_PATH=$ROOT/code/checkpoints
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# EGL render smoke on this GPU (one seeded env, 60 hold steps)
( cd third_party/starVLA && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 NUMBA_DISABLE_JIT=1 \
  OMP_NUM_THREADS=2 PYTHONPATH=$PWD "$ROOT/envs/robocasa/bin/python" "$ROOT/render_bench.py" "$ROOT/logs/render_egl_e3" PnPCanToDrawerClose ) \
  > logs/render_smoke_e3.log 2>&1
grep BENCH logs/render_smoke_e3.log || { log "EGL render smoke failed (logs/render_smoke_e3.log)"; exit 1; }

# layout from what is free now: ~4.6 GB GPU per policy server, ~1.3 GB GPU + ~4 GB RAM per EGL sim client
free_mib=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
avail_gb=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
PAR=$(( (avail_gb - 24) / 4 )); [ $PAR -gt 24 ] && PAR=24
NS=$(( (free_mib - PAR * 1300 - 4096) / 4600 )); [ $NS -gt 6 ] && NS=6
log "free GPU ${free_mib} MiB, MemAvailable ${avail_gb} GB -> parallel $PAR, policy servers $NS"
[ $PAR -ge 8 ] && [ $NS -ge 2 ] || { log "not enough free memory for the evaluation"; exit 1; }

MPS=/dev/shm/gr1_mps_anhdh35_ws_e3; rm -rf "$MPS"; mkdir -p "$MPS/pipe" "$MPS/log"
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
E=3; OUT=$ROOT/results/official_epoch03; rc=1
for attempt in 1 2 3; do
  clean_partial "$OUT"
  log "epoch $E attempt $attempt -> $OUT"
  cd "$ROOT/code"
  nice -n 5 "$FASTWAM_PY" experiments/robocasa_gr1/run_eval.py --checkpoint "$ROOT/models/Robocasa-GR1/epoch_03.pt" \
    --stats "$ROOT/models/Robocasa-GR1/dataset_stats.json" --out "$OUT" --steps 12 --episodes 50 --n-envs 1 \
    --max-episode-steps 720 --seed 0 --gpus 0 --servers-per-gpu "$NS" --base-port 6540 --parallel "$PAR" \
    --server-threads 4 --sim-threads 2 --video --render egl --vae-path "$ROOT/models/vae/Wan2.1_VAE.pth" \
    --expected-step 70548 > "$ROOT/logs/official_epoch03_driver.log" 2>&1
  log "epoch $E attempt $attempt rc=$?"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1]))['by_n_action_steps']['12']; print(json.dumps(dict(tasks=d['tasks'], mean=d['mean_success']))); sys.exit(0 if d['complete'] else 1)" "$OUT/summary.json" && { log "epoch $E complete"; rc=0; break; }
  sleep 90
done
kill $mon 2>/dev/null
[ $rc -eq 0 ] || { log "epoch 3 not complete after 3 attempts"; exit 1; }
cd "$ROOT"
python3 gr1_validate.py "$OUT" 70548 --json results/official_epoch03_validation.json > results/official_epoch03_validation.txt 2>&1
log "epoch 3 validation rc=$?: $(head -1 results/official_epoch03_validation.txt) / $(tail -1 results/official_epoch03_validation.txt)"
log "done"
