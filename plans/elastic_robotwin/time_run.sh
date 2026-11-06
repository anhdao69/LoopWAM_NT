#!/usr/bin/env bash
# One timing run of the real elastic trainer (scripts/train_transport.py) on the given physical GPUs.
# Usage: time_run.sh <RT50-E0|RT50-E1|RT50-E2> <gpus e.g. 0,1,2,3,4,5,6,7> <microbatch> <updates> <tag> [workers]
# Fresh output dir per run; --epochs 1 (planned 47,462 updates; only the LR warmup ramp is visited), --max-updates;
# speed flags = the CL-0 validated set (bit-exact here per preflight_rt50); no block checkpointing; 15 NUMA-local CPUs/rank.
set -euo pipefail
RUN=$1; GPUS=$2; MB=$3; UPDATES=$4; TAG=$5; WORKERS=${6:-3}
source "$(dirname "$0")/env.sh"
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
WORLD=$(echo "$GPUS" | tr ',' '\n' | wc -l)
OUT=$ERT_WORK/timing/$TAG/$RUN
[ -e "$OUT" ] && { echo "refusing: $OUT exists"; exit 3; }
cd "$ELASTIC_ROOT"
[ -z "$(git status --porcelain -- scripts src)" ] || { echo "refusing: scripts/src not clean"; exit 2; }
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=$GPUS BIND_LOG=$OUT/binding.log BIND_TAG=$TAG/$RUN
printf '{"run":"%s","gpus":"%s","world":%s,"microbatch":%s,"updates":%s,"workers":%s,"revision":"%s","host":"%s","job":"%s","start_utc":"%s"}\n' \
  "$RUN" "$GPUS" "$WORLD" "$MB" "$UPDATES" "$WORKERS" "$(git rev-parse HEAD)" "$(hostname)" "${SLURM_JOB_ID:-}" "$(date -u +%FT%TZ)" > "$OUT/launch.json"
echo "[$(date -u +%FT%TZ)] start $TAG/$RUN gpus $GPUS mb $MB updates $UPDATES"
rc=0
torchrun --standalone --nproc_per_node="$WORLD" --no-python "$(dirname "$0")/bind_gpu.sh" scripts/train_transport.py \
  --run-name "$RUN" --output-dir "$OUT" --teacher "$ELASTIC_TEACHER" --vae-path "$ELASTIC_VAE" --manifest "$ELASTIC_MANIFEST" \
  --latent-dir "$ELASTIC_LATENTS" --init-artifact "$ELASTIC_INIT" --expected-gpus "$WORLD" --microbatch "$MB" \
  --workers "$WORKERS" --save-every 100 --no-checkpoint-blocks --epochs 1 --max-updates "$UPDATES" \
  --bf16-frozen --teacher-graph --no-fill-uninitialized --gc-freeze > "$OUT/train.log" 2>&1 || rc=$?
echo "[$(date -u +%FT%TZ)] end $TAG/$RUN rc=$rc"
# Timing runs need no 1.7 GB recovery state; keep metrics/config/logs only.
rm -f "$OUT/latest.pt"
exit $rc
