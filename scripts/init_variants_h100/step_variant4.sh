#!/usr/bin/env bash
# Runs inside an overlapping step (all 4 GPUs) of an interactive allocation, started by stepwise_watcher.sh on the login
# node. Trains VARIANT on all 4 H100s: DDP world 4, microbatch 8, accumulation 4 (the v0 release layout), 4 loader workers
# per rank. A fresh run first stages the latent cache and passes a 10-update smoke on these 4 GPUs (branch smoke check);
# then job_variant.sh trains to completion (run lock; resumes from latest.pt; retries after a crash).
set -uo pipefail
VARIANT=${VARIANT:?set VARIANT}
WORK=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work
RUN=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants/$VARIANT
SRC_CACHE=/groups/yshang/an221229/checkpoints/LoopWAM_NT/full_libero_v0_v4a2_20261026/latents
export GPUS=0,1,2,3 NPROC=4 WORKERS=4
TAG=loopwam_iv_${VARIANT}_${SLURM_JOB_ID}
CACHE=/tmp/$TAG/latents
log() { echo "[$(date -Is)] $*"; }
trainers() { ps -u "$USER" -o pid=,args= | awk -v d="--output-dir $RUN/" '$2 != "awk" && index($0, d) {print $1}'; }
log "step $SLURM_JOB_ID.$SLURM_STEP_ID on $(hostname); CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES; cpus $(nproc)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c 'H100 80GB HBM3')" = 4 ] || { log "need 4 H100 80GB HBM3 GPUs"; exit 2; }
[ -z "$(trainers)" ] || { log "a $VARIANT trainer is already running here: $(trainers)"; exit 3; }
for i in $(seq 1 60); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
  [ "$USED" -lt 2000 ] && break
  log "GPU memory in use ($USED MiB); waiting"; sleep 20
done
[ "$USED" -lt 2000 ] || { log "GPUs are not free"; exit 3; }
mkdir -p "$RUN"
if [ ! -f "$RUN/train/manifest.json" ] && [ ! -f "$RUN/smoke_4gpu.json" ]; then
  source "$WORK/env.sh"
  [ -d "$CACHE" ] && [ ! -f "$CACHE/metadata.json" ] && rm -rf "$CACHE"
  nice -n 10 python "$WORK/seed_latent_cache.py" "$SRC_CACHE" "$CACHE" | tee -a "$RUN/cache_log.jsonl" || { log "cache staging failed"; rm -rf "/tmp/$TAG"; exit 4; }
  if ! VARIANT=$VARIANT TAG="4gpu_$SLURM_JOB_ID" CACHE=$CACHE bash "$WORK/smoke.sh" > "$WORK/logs/smoke_${VARIANT}_4gpu_${SLURM_JOB_ID}.log" 2>&1; then
    log "smoke failed; see logs/smoke_${VARIANT}_4gpu_${SLURM_JOB_ID}.log"; rm -rf "/tmp/$TAG"; exit 5
  fi
  grep '"h100_passed"\|"init"' "$WORK/logs/smoke_${VARIANT}_4gpu_${SLURM_JOB_ID}.log" | tail -2 > "$RUN/smoke_4gpu.json"
  log "smoke passed: $(tail -c 400 "$RUN/smoke_4gpu.json")"
fi
VARIANT=$VARIANT bash "$WORK/job_variant.sh"
