#!/usr/bin/env bash
# One training attempt of an RT50-ep5 run on GPUs 0-3 (world 4, microbatch 32), 15 NUMA-0 CPUs per rank (bind_gpu.sh).
# Slurm step in job 5758: -c120 --cpu-bind=none --gres=gpu:8, started by queue_ep5.sh. Usage: train_run.sh RUN REVISION
# Holds RUNS/RUN/run.lock for its lifetime (inherited by torchrun and the ranks) so a second launch of the same run
# refuses (exit 75); resumes from latest.pt when present; refuses code other than REVISION with clean scripts/src.
set -euo pipefail
RUN=$1; REV=$2
source "$(dirname "$0")/env_ep5.sh"
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
OUT=$ERT_EP5/runs/$RUN; mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || { echo "another process owns $RUN"; exit 75; }
cd "$ELASTIC_ROOT"
[ "$(git rev-parse HEAD)" = "$REV" ] && [ -z "$(git status --porcelain -- scripts src)" ] \
  || { echo "code is not $REV with clean scripts/src"; exit 2; }
busy=1
for i in $(seq 90); do
  busy=$(nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
  [ "$busy" = 0 ] && break; sleep 10
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'; echo
[ "$busy" = 0 ] || { echo "GPUs 0-3 still in use after 15 min"; exit 3; }
echo "${SLURM_STEP_ID:-}" > "$OUT/step_id"
args=(--run-name "$RUN" --output-dir "$OUT" --teacher "$ELASTIC_TEACHER" --vae-path "$ELASTIC_VAE" --manifest "$ELASTIC_MANIFEST"
      --latent-dir "$ELASTIC_LATENTS" --init-artifact "$ELASTIC_INIT" --expected-gpus 4 --microbatch 32 --workers 3
      --epochs 1 --save-every 200 --no-checkpoint-blocks --bf16-frozen --teacher-graph --no-fill-uninitialized --gc-freeze
      --hf-folder not-published)
[ -f "$OUT/latest.pt" ] && args+=(--resume)
printf '{"event":"launch","run":"%s","time":"%s","job":"%s","step":"%s","host":"%s","revision":"%s","resume":%s}\n' "$RUN" \
  "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" "$(hostname)" "$REV" "$([ -f "$OUT/latest.pt" ] && echo true || echo false)" \
  >> "$OUT/launches.jsonl"
echo "[$(date -u +%FT%TZ)] $RUN: launch ${args[*]}"
export CUDA_VISIBLE_DEVICES=0,1,2,3 BIND_LOG=$OUT/binding.log BIND_TAG=$RUN
torchrun --standalone --nproc_per_node=4 --no-python "$(dirname "$0")/bind_gpu.sh" scripts/train_transport.py "${args[@]}" \
  >> "$OUT/train.log" 2>&1
