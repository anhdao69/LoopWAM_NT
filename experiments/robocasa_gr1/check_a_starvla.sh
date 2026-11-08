#!/usr/bin/env bash
# Check (a): the released StarVLA/Qwen3-VL-OFT-Robocasa (steps_90000) through starVLA's own server + client
# (PolicyWarper, 224x224, --no-send-state, n_action_steps 12, max_episode_steps 720) on 2 tasks, to validate
# that this sim install reproduces the published per-task success (README: CanToDrawer 76%, PlateToPan 66%, 50 rollouts).
# Usage (inside the slurm allocation): bash check_a_starvla.sh OUT_DIR
set -euo pipefail
R=${GR1_EVAL_ROOT:-/mnt/data/vmo-ai-task/anhdh35/gr1_eval}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${1:?out dir}; GPU=${GPU:-0}; PORT=${PORT:-6600}; EPISODES=${EPISODES:-50}; STEPS=${STEPS:-12}; NENVS=${NENVS:-5}
TASKS=${TASKS:-gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env}
CK=$R/models/Qwen3-VL-OFT-Robocasa/checkpoints/steps_90000_pytorch_model.pt
mkdir -p "$OUT"
if (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then echo "port $PORT in use"; exit 1; fi
cd "$R/third_party/starVLA"
CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONPATH=$PWD \
  "$R/envs/starvla/bin/python" deployment/model_server/server_policy.py --ckpt_path "$CK" --port "$PORT" --use_bf16 --seed 42 \
  --idle_timeout -1 --config_override "framework.qwenvl.base_vlm=$R/models/Qwen3-VL-4B-Instruct" > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server died"; tail -30 "$OUT/server.log"; exit 1; }
  sleep 5
done
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=${OSMESA_LIB:-/mnt/data/vmo-ai-task/anhdh35/FastWAM/third_party/system_libs/usr/lib/x86_64-linux-gnu}:${LD_LIBRARY_PATH:-}
unset MUJOCO_EGL_DEVICE_ID
pids=()
for t in ${TASKS//,/ }; do
  n=${t#gr1_unified/}; n=${n%_GR1ArmsAndWaistFourierHands_Env}
  PYTHONPATH=$PWD "$R/envs/robocasa/bin/python" "$CODE/experiments/robocasa_gr1/sim_client.py" --policy starvla --no-send-state \
    --env-name "$t" --port "$PORT" --n-episodes "$EPISODES" --n-envs "$NENVS" --n-action-steps "$STEPS" --max-episode-steps 720 \
    --seed 0 --out "$OUT/$n.json" > "$OUT/$n.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
for f in "$OUT"/*.json; do python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print(r['env_name'],r['success_rate'],r['successes_first_n'],'/',r['n_episodes'],r['wall_seconds'],'s')" "$f"; done
exit $rc
