#!/usr/bin/env bash
# One elastic run on the GPUs in CUDA_VISIBLE_DEVICES: resume-aware, single-writer, retried,
# with a background verified uploader to anhdao69/LoopWAM_NT/libero-all-suites/Elastic/<run>.
# Required: ELASTIC_RUN, ELASTIC_OUTPUT, ELASTIC_INIT, ELASTIC_WORLD; environment from env.sh.
set -euo pipefail
: "${ELASTIC_RUN:?run id}" "${ELASTIC_OUTPUT:?output directory}" "${ELASTIC_INIT:?init artifact}" "${ELASTIC_WORLD:?GPUs}"
: "${ELASTIC_ROOT:?source checkout}" "${ELASTIC_TEACHER:?}" "${ELASTIC_VAE:?}" "${ELASTIC_MANIFEST:?}" "${ELASTIC_LATENTS:?}"
mkdir -p "$ELASTIC_OUTPUT"
exec 9>"$ELASTIC_OUTPUT/run.lock"
flock -n 9 || { echo "Another process owns $ELASTIC_RUN; refusing duplicate."; exit 75; }
cd "$ELASTIC_ROOT"
uploader=""
if [[ -n "${ELASTIC_HF_TOKEN_FILE:-}" ]]; then
    python scripts/upload_elastic.py --run-name "$ELASTIC_RUN" --output-dir "$ELASTIC_OUTPUT" \
        --token-file "$ELASTIC_HF_TOKEN_FILE" --watch >> "$ELASTIC_OUTPUT/upload.log" 2>&1 &
    uploader=$!
fi
trainer_pid=""; stopping=0
on_signal() { stopping=1; if [[ -n "$trainer_pid" ]]; then pkill -USR1 -P "$trainer_pid" || true; fi; }
trap on_signal USR1 TERM
cleanup() { if [[ -n "$uploader" ]]; then kill "$uploader" 2>/dev/null || true; fi; }
trap cleanup EXIT
for attempt in 1 2 3; do
    args=(--run-name "$ELASTIC_RUN" --output-dir "$ELASTIC_OUTPUT" --teacher "$ELASTIC_TEACHER"
          --vae-path "$ELASTIC_VAE" --manifest "$ELASTIC_MANIFEST" --latent-dir "$ELASTIC_LATENTS"
          --init-artifact "$ELASTIC_INIT" --expected-gpus "$ELASTIC_WORLD" --microbatch "${ELASTIC_MICROBATCH:-64}"
          --workers "${ELASTIC_WORKERS:-2}" --save-every "${ELASTIC_SAVE_EVERY:-100}" --no-checkpoint-blocks
          --hf-folder "https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/libero-all-suites/Elastic/$ELASTIC_RUN")
    if [[ -n "${ELASTIC_CKPT_MIN_BLOCKS:-}" ]]; then args+=(--checkpoint-min-blocks "$ELASTIC_CKPT_MIN_BLOCKS"); fi
    if [[ -n "${ELASTIC_MAX_UPDATES:-}" ]]; then args+=(--max-updates "$ELASTIC_MAX_UPDATES"); fi
    if [[ -f "$ELASTIC_OUTPUT/latest.pt" ]]; then args+=(--resume); fi
    echo "{\"event\":\"launch\",\"run\":\"$ELASTIC_RUN\",\"attempt\":$attempt,\"time\":\"$(date -Is)\",\"gpus\":\"${CUDA_VISIBLE_DEVICES:-}\",\"job\":\"${SLURM_JOB_ID:-}\"}" >> "$ELASTIC_OUTPUT/launches.jsonl"
    torchrun --standalone --nproc_per_node="$ELASTIC_WORLD" scripts/train_transport.py "${args[@]}" >> "$ELASTIC_OUTPUT/train.log" 2>&1 &
    trainer_pid=$!
    code=0
    wait "$trainer_pid" || code=$?
    if kill -0 "$trainer_pid" 2>/dev/null; then wait "$trainer_pid" || code=$?; fi
    trainer_pid=""
    if [[ "$stopping" == 1 ]]; then exit 0; fi
    if [[ "$code" == 0 ]]; then break; fi
    if [[ "$attempt" == 3 ]]; then exit "$code"; fi
    sleep 30
done
if [[ -n "${ELASTIC_MAX_UPDATES:-}" ]]; then exit 0; fi
python - "$ELASTIC_OUTPUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
status=json.loads((out/"status.json").read_text())
assert status["status"]=="complete" and status["update"]==21700, status
assert all((out/f"epoch_{e:02d}.pt").is_file() for e in (8,9,10)) and (out/"step_00002000.pt").is_file()
print("training complete:",out)
PY
if [[ -n "$uploader" ]]; then wait "$uploader" || true; uploader=""; fi
