#!/usr/bin/env bash
# One training attempt of GR1-E0 x5 on GPUs 4-7 (world 4, microbatch 32, 15 NUMA-1 CPUs per rank via the clone's
# plans/elastic_robotwin/bind_gpu.sh) up to the end of epoch TARGET: --epochs 5 --max-updates TARGET*47032, so every
# epoch end stops the run, its epoch_EE.pt is verified and evaluated, and the next epoch resumes from latest.pt.
# Slurm step in job 5758 (-c120 --cpu-bind=none --gres=gpu:8 -J training_vln), started by orchestrate_gr1e.sh.
# Same layout and speed flags as the RT50 runs (train_e10x5.sh); never trains when latest.pt is already at TARGET (a
# resume at update == --max-updates would train one update past it): if the epoch-end export or status.json is then
# missing (a stop between the save and the export), it runs --resume --finalize-only, which writes them from latest.pt
# and trains nothing. Usage: train_gr1e.sh REVISION TARGET_EPOCH
set -euo pipefail
REV=$1; TARGET=$2
source "$(dirname "$0")/env_gr1e.sh"
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
RUN=$GR1E_RUN
case "$TARGET" in 1|2|3|4|5) ;; *) echo "bad target epoch $TARGET"; exit 2 ;; esac
MAX=$((TARGET * GR1E_PER_EPOCH))
OUT=$EGR_WORK/runs/$RUN; mkdir -p "$OUT"
exec 9>"$OUT/run.lock"
flock -n 9 || { echo "another process owns $RUN"; exit 75; }
cd "$ELASTIC_ROOT"
[ "$(git rev-parse HEAD)" = "$REV" ] && [ -z "$(git status --porcelain -- scripts src experiments plans)" ] \
  || { echo "code is not $REV with clean scripts/src/experiments/plans"; exit 2; }
[ -f "$ELASTIC_SELECTION" ] && [ -n "${ELASTIC_TEACHER:-}" ] && [ -f "$ELASTIC_INIT" ] || { echo "no teacher selection / init"; exit 2; }
if [ -f "$OUT/latest.pt" ]; then
  have=$(python - "$OUT/latest.pt" <<'PY'
import sys, torch
print(torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True)["update"])
PY
)
  if [ "$have" -gt "$MAX" ]; then echo "latest.pt is at update $have > $MAX: past the end of epoch $TARGET"; exit 4; fi
  if [ "$have" = "$MAX" ]; then
    if python3 -I - "$OUT" "$TARGET" "$MAX" <<'PY'
import json, sys
from pathlib import Path
out, e, m = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
try: s = json.loads((out / "status.json").read_text())
except Exception: sys.exit(1)
sys.exit(0 if s.get("update") == m and s.get("epoch") == e and s.get("next_micro") == 0 and (out / f"epoch_{e:02d}.pt").is_file() else 1)
PY
    then echo "latest.pt is at update $have = end of epoch $TARGET with its export and status: nothing to train"; exit 0; fi
    echo "latest.pt is at update $have = end of epoch $TARGET but epoch_$(printf %02d "$TARGET").pt or status.json is missing: --finalize-only"
    FINALIZE=1
  else
    echo "latest.pt at update $have; training to $MAX"
  fi
fi
busy=1
for i in $(seq 90); do
  busy=$(nvidia-smi -i "$GR1E_GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
  [ "$busy" = 0 ] && break; sleep 10
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'; echo
[ "$busy" = 0 ] || { echo "GPUs $GR1E_GPUS still in use after 15 min"; exit 3; }
echo "${SLURM_STEP_ID:-}" > "$OUT/step_id"
args=(--run-name "$RUN" --output-dir "$OUT" --teacher "$ELASTIC_TEACHER" --teacher-selection "$ELASTIC_SELECTION"
      --vae-path "$ELASTIC_VAE" --manifest "$ELASTIC_MANIFEST" --latent-dir "$ELASTIC_LATENTS" --init-artifact "$ELASTIC_INIT"
      --expected-gpus 4 --microbatch 32 --workers 3 --epochs "$GR1E_EPOCHS" --max-updates "$MAX" --save-every 200
      --no-checkpoint-blocks --bf16-frozen --teacher-graph --no-fill-uninitialized --gc-freeze --hf-folder not-published)
[ -f "$OUT/latest.pt" ] && args+=(--resume)
[ "${FINALIZE:-0}" = 1 ] && args+=(--finalize-only)
printf '{"event":"launch","run":"%s","time":"%s","job":"%s","step":"%s","host":"%s","revision":"%s","resume":%s,"finalize_only":%s,"target_epoch":%s,"max_updates":%s}\n' \
  "$RUN" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-}" "${SLURM_STEP_ID:-}" "$(hostname)" "$REV" \
  "$([ -f "$OUT/latest.pt" ] && echo true || echo false)" "$([ "${FINALIZE:-0}" = 1 ] && echo true || echo false)" "$TARGET" "$MAX" >> "$OUT/launches.jsonl"
echo "[$(date -u +%FT%TZ)] $RUN: launch ${args[*]}"
export CUDA_VISIBLE_DEVICES=$GR1E_GPUS BIND_LOG=$OUT/binding.log BIND_TAG=$RUN
torchrun --standalone --nproc_per_node=4 --no-python "$ELASTIC_ROOT/plans/elastic_robotwin/bind_gpu.sh" scripts/train_transport.py \
  "${args[@]}" >> "$OUT/train.log" 2>&1
