#!/usr/bin/env bash
# usage: eval_gr1e_k2.sh E SCHEDULE ATTEMPT  [sibling of eval_gr1e.sh made 2026-11-10: identical except that the schedule list also accepts flat_k2_n2 (port slot 3)]   (Slurm step in job 5758: -c120 --cpu-bind=none --gres=gpu:8; started by
# orchestrate_gr1e.sh while GR1-E0 training is paused at the end of epoch E)
# Official RoboCasa-GR1 evaluation of GR1-E0 epoch_EE.pt with one elastic schedule, exactly the GR1 continuation's
# protocol and layout (continuation_20261106/tools/eval_epoch.sh): 24 tasks x 50 episodes, n_envs 1, max_episode_steps
# 720, n_action_steps 12, seed 0, videos, EGL on the policy servers' GPUs 4-7 (EGL id k = physical GPU k), state sent,
# 2 servers per GPU, 24 parallel sim clients, only the job's NUMA-1 CPUs; the server is elastic_policy_server.py
# (run_eval.py --elastic-schedule SCHEDULE). Overrides for the pre-launch smoke only: EVAL_CK EVAL_STEP EVAL_OUT
# EVAL_TASKS EVAL_EPISODES EVAL_PORT.
set -uo pipefail
E=$1; S=$2; ATTEMPT=${3:-1}; EE=$(printf %02d "$E")
source "$(dirname "$0")/env_gr1e.sh"
case "$S" in flat_k1_n2) SI=0 ;; flat_k4_n2) SI=1 ;; flat_k4_n5) SI=2 ;; flat_k2_n2) SI=3 ;; *) echo "unexpected schedule $S"; exit 2 ;; esac
CK=${EVAL_CK:-$EGR_WORK/runs/$GR1E_RUN/epoch_$EE.pt}
STEP=${EVAL_STEP:-$((E * GR1E_PER_EPOCH))}
OUT=${EVAL_OUT:-$EGR_WORK/eval/e${EE}_$S}
PORT=${EVAL_PORT:-$((7000 + 200 * E + 50 * SI + 10 * (ATTEMPT - 1)))}
CODE=$ELASTIC_ROOT
export PYTHONPATH=$CODE/src:$CODE:$CODE/scripts
export HF_HOME=/mnt/data/vmo-ai-task/anhdh35/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface DIFFSYNTH_SKIP_DOWNLOAD=true
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/tmp/gr1e_eval_${SLURM_JOB_ID:-x}
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
mkdir -p "$TMPDIR" "$OUT"
cd "$CODE"
log() { echo "[$(date -u +%FT%TZ)] eval-e$EE-$S-a$ATTEMPT: $*"; }
[ -z "$(git status --porcelain -- scripts src experiments plans)" ] || { log 'eval code is not clean'; exit 2; }
[ -f "$CK" ] || { log "missing checkpoint $CK"; exit 2; }
for i in $(seq 90); do
  busy=$(nvidia-smi -i "$GR1E_GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
  [ "$busy" = 0 ] && break; sleep 10
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'; echo
[ "$busy" = 0 ] || { log "GPUs $GR1E_GPUS still in use after 15 min"; exit 3; }
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
log "checkpoint $CK step $STEP schedule $S -> $OUT ports from $PORT, cpus $CPUS, code $(git rev-parse HEAD)"
# a retry keeps finished task cells (run_eval skips them) but moves unfinished cells' videos aside
for d in "$OUT"/videos/n12/*/; do
  [ -d "$d" ] || continue; t=$(basename "$d")
  [ -f "$OUT/n12/$t.json" ] || { dest=$OUT/partial_attempts/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$dest"; mv "$d" "$dest/"; log "moved unfinished videos of $t to $dest"; }
done
# an earlier attempt's server logs (whose errors ended it) go aside too: validate_elastic_eval.py checks server errors in
# this attempt's logs and prompt sources in every attempt's (partial_attempts/*/logs)
if compgen -G "$OUT/logs/server_*.log" > /dev/null; then
  dest=$OUT/partial_attempts/$(date -u +%Y%m%dT%H%M%SZ)/logs; mkdir -p "$dest"; mv "$OUT"/logs/server_*.log "$dest/"; log "moved earlier server logs to $dest"
fi
TASKS=(); [ -n "${EVAL_TASKS:-}" ] && TASKS=(--tasks "$EVAL_TASKS")
taskset -c "$CPUS" python experiments/robocasa_gr1/run_eval.py --checkpoint "$CK" --elastic-schedule "$S" --out "$OUT" \
  --base-port "$PORT" --expected-step "$STEP" --stats "$ELASTIC_STATS" --steps 12 --episodes "${EVAL_EPISODES:-50}" --n-envs 1 \
  --max-episode-steps 720 --seed 0 --gpus "$GR1E_GPUS" --servers-per-gpu 2 --parallel 24 --server-threads 4 --sim-threads 2 \
  --video --render egl --egl-libs "$GR1E_EGL" --vae-path "$ELASTIC_VAE" "${TASKS[@]}" >> "$OUT/driver.log" 2>&1
rc=$?; log "run_eval rc=$rc"
python3 -c "import json,sys; d=json.load(open(sys.argv[1]))['by_n_action_steps']['12']; print(json.dumps(dict(tasks=d['tasks'], complete=d['complete'], mean=d['mean_success']))); sys.exit(0 if d['complete'] else 1)" "$OUT/summary.json"
