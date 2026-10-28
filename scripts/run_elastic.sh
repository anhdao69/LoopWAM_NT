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
# Lustre is mounted with localflock, so flock only excludes writers on this node. Across nodes a run is owned
# through an atomic mkdir claim (owner job + heartbeat); recent training output also counts as ownership.
claim="$ELASTIC_OUTPUT/claim.d"
token="${SLURM_JOB_ID:-nojob}:$(hostname):$$"
recently_written() {
    local f now; now=$(date +%s)
    for f in train.log metrics.jsonl; do
        if [[ -f "$ELASTIC_OUTPUT/$f" ]] && (( now - $(stat -c %Y "$ELASTIC_OUTPUT/$f") < 900 )); then return 0; fi
    done
    return 1
}
claim_live() {
    local job beat
    if [[ ! -f "$claim/heartbeat" ]]; then (( $(date +%s) - $(stat -c %Y "$claim" 2>/dev/null || echo 0) < 120 )); return; fi
    job=$(cat "$claim/job" 2>/dev/null || true); beat=$(stat -c %Y "$claim/heartbeat")
    [[ -n "$job" && "$(squeue -h -j "$job" -o %T 2>/dev/null)" == RUNNING ]] && (( $(date +%s) - beat < 600 ))
}
if [[ -d "$claim" ]]; then
    if claim_live; then echo "$ELASTIC_RUN is claimed by $(cat "$claim/owner" 2>/dev/null)"; exit 75; fi
    mv "$claim" "$ELASTIC_OUTPUT/claim.stale.$(date +%s).$$" 2>/dev/null || exit 75
fi
if recently_written; then echo "$ELASTIC_RUN was written in the last 15 min by another process"; exit 75; fi
mkdir "$claim" 2>/dev/null || exit 75
echo "${SLURM_JOB_ID:-}" > "$claim/job"; echo "$token" > "$claim/owner"; echo "${ELASTIC_LANE:-}" > "$claim/lane"
touch "$claim/heartbeat"
( exec 9>&-; while sleep 60; do touch "$claim/heartbeat" 2>/dev/null || exit 0; done ) &
heartbeat=$!
cd "$ELASTIC_ROOT"
uploader=""
if [[ -n "${ELASTIC_HF_TOKEN_FILE:-}" ]]; then
    python scripts/upload_elastic.py --run-name "$ELASTIC_RUN" --output-dir "$ELASTIC_OUTPUT" \
        --token-file "$ELASTIC_HF_TOKEN_FILE" --watch >> "$ELASTIC_OUTPUT/upload.log" 2>&1 9>&- &
    uploader=$!
fi
trainer_pid=""; stopping=0
on_signal() { stopping=1; if [[ -n "$trainer_pid" ]]; then pkill -USR1 -P "$trainer_pid" || true; fi; }
trap on_signal USR1 TERM
cleanup() {
    if [[ -n "$uploader" ]]; then kill "$uploader" 2>/dev/null || true; fi
    kill "$heartbeat" 2>/dev/null || true
    if [[ "$(cat "$claim/owner" 2>/dev/null)" == "$token" ]]; then rm -rf "$claim"; fi
}
trap cleanup EXIT
revision=$(git rev-parse HEAD 2>/dev/null || echo unknown)
for attempt in 1 2 3; do
    args=(--run-name "$ELASTIC_RUN" --output-dir "$ELASTIC_OUTPUT" --teacher "$ELASTIC_TEACHER"
          --vae-path "$ELASTIC_VAE" --manifest "$ELASTIC_MANIFEST" --latent-dir "$ELASTIC_LATENTS"
          --init-artifact "$ELASTIC_INIT" --expected-gpus "$ELASTIC_WORLD" --microbatch "${ELASTIC_MICROBATCH:-64}"
          --workers "${ELASTIC_WORKERS:-2}" --save-every "${ELASTIC_SAVE_EVERY:-100}" --no-checkpoint-blocks
          --hf-folder "https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/libero-all-suites/Elastic/$ELASTIC_RUN")
    if [[ -n "${ELASTIC_CKPT_MIN_BLOCKS:-}" ]]; then args+=(--checkpoint-min-blocks "$ELASTIC_CKPT_MIN_BLOCKS"); fi
    if [[ -n "${ELASTIC_MAX_UPDATES:-}" ]]; then args+=(--max-updates "$ELASTIC_MAX_UPDATES"); fi
    if [[ -f "$ELASTIC_OUTPUT/latest.pt" ]]; then args+=(--resume); fi
    echo "{\"event\":\"launch\",\"run\":\"$ELASTIC_RUN\",\"attempt\":$attempt,\"time\":\"$(date -Is)\",\"gpus\":\"${CUDA_VISIBLE_DEVICES:-}\",\"job\":\"${SLURM_JOB_ID:-}\",\"host\":\"$(hostname)\",\"lane\":\"${ELASTIC_LANE:-}\",\"revision\":\"$revision\"}" >> "$ELASTIC_OUTPUT/launches.jsonl"
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
