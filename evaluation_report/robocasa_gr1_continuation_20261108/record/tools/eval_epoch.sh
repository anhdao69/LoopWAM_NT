#!/usr/bin/env bash
# usage: eval_epoch.sh E ATTEMPT   (Slurm step in job 5758: -c120 --cpu-bind=none --gres=gpu:8; started by orchestrate.sh)
# Official RoboCasa-GR1 evaluation of continuation epoch E while training is paused, on GPUs 4-7 only (policy servers
# CUDA_VISIBLE_DEVICES=4..7, EGL rendering on the same GPUs: EGL id k = physical GPU k, checked by egl_map_check.sh) and
# only the job's NUMA-1 CPUs (taskset), so GPUs 0-3 / NUMA-0 CPUs are never used.
# Protocol and code exactly as the epoch 4/5 H100 evals (gr1_eval/tools/gr1_eval_step.sh, clone LoopWAM_gr1_eval_code
# @d937167): 24 tasks x 50 episodes, n_envs 1, max_episode_steps 720, n_action_steps 12, seed 0, videos, EGL, state
# sent, 2 servers per GPU, 24 parallel sim clients. No MPS (keeps the MPS daemon off the other GPUs).
# Overrides for the smoke test only: EVAL_CK EVAL_STEP EVAL_OUT EVAL_TASKS EVAL_EPISODES EVAL_PORT EVAL_ALLOW_BUSY.
set -uo pipefail
E=$1; ATTEMPT=${2:-1}; EE=$(printf %02d "$E")
RUN=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101; EV=$RUN/eval
CK=${EVAL_CK:-$RUN/train/continuation_e06_10/checkpoints/epoch_$EE.pt}
STEP=${EVAL_STEP:-$((E * 23516))}
OUT=${EVAL_OUT:-$EV/official_epoch$EE}
PORT=${EVAL_PORT:-$((6700 + 40 * (E - 6) + 10 * (ATTEMPT - 1)))}
CODE=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_gr1_eval_code
export PATH=/mnt/data/vmo-ai-task/anhdh35/FastWAM/.venv/bin:$PATH
export PYTHONPATH=$CODE/src:$CODE:$CODE/scripts
export ROBOCASA_GR1_ROOT=/mnt/data/vmo-ai-task/anhdh35/datasets/robocasa_gr1
export ROBOCASA_GR1_TEXT_CACHE=/mnt/data/vmo-ai-task/anhdh35/datasets/LoopWAM_NT/text_embeds_cache/robocasa_gr1
export HF_HOME=/mnt/data/vmo-ai-task/anhdh35/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_MODEL_BASE_PATH=/mnt/data/vmo-ai-task/anhdh35/FastWAM/checkpoints DIFFSYNTH_DOWNLOAD_SOURCE=huggingface DIFFSYNTH_SKIP_DOWNLOAD=true
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/tmp/gr1_eval_${SLURM_JOB_ID:-x}
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
mkdir -p "$TMPDIR" "$OUT"
STATS=$RUN/shared/data/dataset_stats_29d.json
VAE=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_NT/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
EGL=/mnt/data/vmo-ai-task/anhdh35/elastic_cl0/eval/egl_libs
cd "$CODE"
log() { echo "[$(date -u +%FT%TZ)] eval-e$EE-a$ATTEMPT: $*"; }
[ "$(git rev-parse HEAD)" = d937167ba61a0df4a9818d1917de6c7155400dff ] || { log 'unexpected eval code revision'; exit 2; }
[ -f "$CK" ] || { log "missing checkpoint $CK"; exit 2; }
for i in $(seq 90); do
  busy=$(nvidia-smi -i 4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
  [ "$busy" = 0 ] && break; [ -n "${EVAL_ALLOW_BUSY:-}" ] && break; sleep 10
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'; echo
[ "$busy" = 0 ] || [ -n "${EVAL_ALLOW_BUSY:-}" ] || { log 'GPUs 4-7 still in use after 15 min'; exit 3; }
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | sed -E 's#(/job_[0-9]+).*#\1#')
CPUS=$(python3 -I - "$(cat $CG/cpuset.cpus.effective)" /sys/devices/system/node/node1/cpulist <<'PY'
import sys
def parse(s):
    out = []
    for part in s.strip().split(','):
        a, _, b = part.partition('-'); out += range(int(a), int(b or a) + 1)
    return out
cpus = sorted(set(parse(sys.argv[1])) & set(parse(open(sys.argv[2]).read())))
assert len(cpus) >= 32, cpus
print(','.join(map(str, cpus)))
PY
) || { log 'cannot determine NUMA-1 CPUs'; exit 2; }
log "checkpoint $CK step $STEP -> $OUT ports from $PORT, cpus $CPUS"
# a retry keeps finished task cells (run_eval skips them) but moves unfinished cells' videos aside
for d in "$OUT"/videos/n12/*/; do
  [ -d "$d" ] || continue; t=$(basename "$d")
  [ -f "$OUT/n12/$t.json" ] || { dest=$OUT/partial_attempts/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$dest"; mv "$d" "$dest/"; log "moved unfinished videos of $t to $dest"; }
done
TASKS=(); [ -n "${EVAL_TASKS:-}" ] && TASKS=(--tasks "$EVAL_TASKS")
taskset -c "$CPUS" python experiments/robocasa_gr1/run_eval.py --checkpoint "$CK" --out "$OUT" --base-port "$PORT"   --expected-step "$STEP" --stats "$STATS" --steps 12 --episodes "${EVAL_EPISODES:-50}" --n-envs 1 --max-episode-steps 720 --seed 0   --gpus 4,5,6,7 --servers-per-gpu 2 --parallel 24 --server-threads 4 --sim-threads 2 --video --render egl --egl-libs "$EGL"   --vae-path "$VAE" "${TASKS[@]}" >> "$OUT/driver.log" 2>&1
rc=$?; log "run_eval rc=$rc"
python3 -c "import json,sys; d=json.load(open(sys.argv[1]))['by_n_action_steps']['12']; print(json.dumps(dict(tasks=d['tasks'], complete=d['complete'], mean=d['mean_success']))); sys.exit(0 if d['complete'] else 1)" "$OUT/summary.json"
