#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/transport_env.sh"
: "${RT_RUN:?experiment name}"
: "${RT_OUTPUT:?independent output path}"
mkdir -p "$RT_OUTPUT"
exec 9>"$RT_OUTPUT/run.lock"
flock -n 9 || { echo "Another process owns this experiment; refusing duplicate."; exit 75; }
cd "$RT_ROOT"
python scripts/upload_transport.py --run-name "$RT_RUN" --output-dir "$RT_OUTPUT" --watch > "$RT_OUTPUT/upload.log" 2>&1 &
uploader=$!
trainer_pid=""
preempted=0
on_signal() {
    preempted=1
    if [[ -n "$trainer_pid" ]]; then pkill -USR1 -P "$trainer_pid" || true; fi
}
trap on_signal USR1 TERM
cleanup() { kill "$uploader" 2>/dev/null || true; }
trap cleanup EXIT
for attempt in 1 2 3; do
    args=(--expected-gpus "${RT_WORLD_SIZE:-2}" --run-name "$RT_RUN" --output-dir "$RT_OUTPUT" --microbatch "${RT_MICROBATCH:-64}" --workers 4)
    if [[ "${RT_CHECKPOINT_BLOCKS:-0}" == 0 ]]; then args+=(--no-checkpoint-blocks); fi
    if [[ -f "$RT_OUTPUT/latest.pt" ]]; then args+=(--resume); fi
    if [[ -n "${RT_INIT_CHECKPOINT:-}" ]]; then args+=(--init-checkpoint "$RT_INIT_CHECKPOINT"); fi
    torchrun --standalone --nproc_per_node="${RT_WORLD_SIZE:-2}" scripts/train_transport.py "${args[@]}" >> "$RT_OUTPUT/train.log" 2>&1 &
    trainer_pid=$!
    code=0
    wait "$trainer_pid" || code=$?
    if kill -0 "$trainer_pid" 2>/dev/null; then wait "$trainer_pid" || code=$?; fi
    trainer_pid=""
    if [[ "$preempted" == 1 ]]; then
        if [[ "${RT_CAN_REQUEUE:-0}" == 1 ]]; then scontrol requeue "$SLURM_JOB_ID"; fi
        exit 0
    fi
    if [[ "$code" == 0 ]]; then break; fi
    if [[ "$attempt" == 3 ]]; then exit "$code"; fi
    sleep 15
done
# One writer uploads at a time. Retry all completed epochs even if watch exited.
kill "$uploader" 2>/dev/null || true
wait "$uploader" 2>/dev/null || true
python scripts/upload_transport.py --run-name "$RT_RUN" --output-dir "$RT_OUTPUT"
python - "$RT_OUTPUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
assert json.loads((out/"status.json").read_text())["status"]=="complete"
assert all((out/f"epoch_{e:02d}.pt.uploaded.json").is_file() for e in (8,9,10))
print("Training and epoch uploads verified")
PY
