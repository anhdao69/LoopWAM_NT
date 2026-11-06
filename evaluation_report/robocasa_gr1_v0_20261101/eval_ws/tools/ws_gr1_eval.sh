#!/usr/bin/env bash
# ws_anhdh35: RoboCasa-GR1 official evaluation of LoopWAM v0 epoch_04 on the RTX PRO 6000 (GPU 0).
# Starts only after the current RoboTwin evaluation (loopwam_robotwin10_v0_2x2_20261104) has finished: no process
# of that run directory for 5 consecutive minutes. Same protocol and code as the H100 run (d937167):
# starVLA Robocasa_tabletop batch_eval_args.sh settings (24 tasks x 50 episodes, n_envs 1, max_episode_steps 720,
# n_action_steps 12, videos), seed 0, EGL rendering on the GPU of the policy server.
# Inputs verified before use: epoch_04.pt sha256 = HF receipt (ws_fetch_ckpt.sh), VAE sha256 = training contract,
# prompt cache = H100 training cache (187 files, sha256 list), assets = H100 asset tree (manifest compare).
set -uo pipefail
ROOT=/storage/anhdh35/robocasa_gr1_eval
Q=/data14t/anhdh35/loopwam_robotwin10_v0_2x2_20261104
NSERVERS=${NSERVERS:-3}
log() { echo "[$(date -u +%FT%TZ)] ws-gr1: $*"; }
log "start pid $$; waiting for the RoboTwin evaluation in $Q to finish"
quiet=0
while [ "$quiet" -lt 5 ]; do
  if pgrep -u anhdh35 -f "$Q/" > /dev/null; then quiet=0; else quiet=$((quiet + 1)); fi
  sleep 60
done
log "RoboTwin evaluation finished: $(python3 -c "import json; d=json.load(open('$Q/results/summary.json')); print({k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in d.items() if k in ('finished', 'running', 'failed', 'pending')})" 2>&1)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

# inputs
cd "$ROOT/models/text_cache" && sha256sum -c --quiet "$ROOT/text_cache.sha256" && [ "$(ls | wc -l)" = 187 ] \
  || { log "prompt cache incomplete or differs from H100"; exit 1; }
cd "$ROOT"
[ "$(sha256sum models/vae/Wan2.1_VAE.pth | cut -d' ' -f1)" = 38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981 ] || { log "VAE sha mismatch"; exit 1; }
[ "$(sha256sum models/Robocasa-GR1/epoch_04.pt | cut -d' ' -f1)" = b62483a91bcaec1fd5586e11ad36590043f276d6c82e5bd50c4fa55a6b018ba6 ] || { log "checkpoint sha mismatch"; exit 1; }
[ "$(git -C code rev-parse HEAD)" = d937167ba61a0df4a9818d1917de6c7155400dff ] || { log "code revision mismatch"; exit 1; }
log "inputs verified (prompt cache 187/187, VAE, epoch_04.pt, code d937167)"

export GR1_EVAL_ROOT=$ROOT FASTWAM_PY=/storage/anhdh35/miniconda3/envs/loopwam-libero-eval/bin/python
export ROBOCASA_GR1_TEXT_CACHE=$ROOT/models/text_cache PYTHONPATH=$ROOT/code/src:$ROOT/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true DIFFSYNTH_MODEL_BASE_PATH=$ROOT/code/checkpoints
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# EGL render smoke on this GPU (one seeded env, 60 hold steps)
( cd third_party/starVLA && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 NUMBA_DISABLE_JIT=1 \
  OMP_NUM_THREADS=2 PYTHONPATH=$PWD "$ROOT/envs/robocasa/bin/python" "$ROOT/render_bench.py" "$ROOT/logs/render_egl" PnPCanToDrawerClose ) \
  > logs/render_smoke.log 2>&1
grep BENCH logs/render_smoke.log || { log "EGL render smoke failed (logs/render_smoke.log)"; exit 1; }

OUT=$ROOT/results/official_epoch04
for attempt in 1 2 3; do
  log "epoch 4 attempt $attempt -> $OUT ($NSERVERS policy servers on GPU 0)"
  cd "$ROOT/code"
  nice -n 5 "$FASTWAM_PY" experiments/robocasa_gr1/run_eval.py --checkpoint "$ROOT/models/Robocasa-GR1/epoch_04.pt" \
    --stats "$ROOT/models/Robocasa-GR1/dataset_stats.json" --out "$OUT" --steps 12 --episodes 50 --n-envs 1 \
    --max-episode-steps 720 --seed 0 --gpus 0 --servers-per-gpu "$NSERVERS" --base-port 6500 --parallel 24 \
    --server-threads 4 --sim-threads 2 --video --render egl --vae-path "$ROOT/models/vae/Wan2.1_VAE.pth" \
    --expected-step 94064 > "$ROOT/logs/official_epoch04_driver.log" 2>&1
  log "attempt $attempt rc=$?"
  python3 -c "import json,sys; d=json.load(open(sys.argv[1]))['by_n_action_steps']['12']; print(json.dumps(dict(tasks=d['tasks'], mean=d['mean_success']))); sys.exit(0 if d['complete'] else 1)" "$OUT/summary.json" && { log "complete"; exit 0; }
  sleep 60
done
log "not complete after 3 attempts"; exit 1
