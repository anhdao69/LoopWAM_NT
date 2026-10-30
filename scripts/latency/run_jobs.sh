#!/usr/bin/env bash
# Runs latency jobs (jobs.py JSON lines) one at a time on one GPU; skips jobs whose result file exists.
# graph/compile are timed on every k-th observation, k read from $LAT/graph_stride (default 5) when a job starts.
# Usage: run_jobs.sh <jobs.jsonl> <physical GPU> <cpu list for taskset, e.g. 0-3> <job id> [<job id> ...]
# Needs latency_env.sh sourced (LAT, LAT_CODE, LAT_OBS, LAT_OBS_SHA, LAT_TEXT). Results: ${LAT_RESULTS:-$LAT/results}/<id>.json.
set -uo pipefail
JOBS=$1; GPU=$2; CPUS=$3; shift 3
RES=${LAT_RESULTS:-$LAT/results}
mkdir -p "$RES" "$LAT/logs"
for id in "$@"; do
  out=$RES/$id.json
  [ -f "$out" ] && { echo "[$(date -Is)] $id: done already"; continue; }
  spec=$(python -c "import json,sys; [print(l.strip()) for l in open(sys.argv[1]) if json.loads(l)['id']==sys.argv[2]]" "$JOBS" "$id")
  [ -n "$spec" ] || { echo "[$(date -Is)] $id: no such job"; continue; }
  args=$(python - "$spec" <<'EOF'
import json, shlex, sys
j = json.loads(sys.argv[1])
a = ["--code-root", j["code_root"], "--family", j["family"], "--model", j["name"], "--checkpoint", j["checkpoint"],
     "--configs", ",".join(j["configs"]), "--vae-path", j["vae"], "--backends", j["backends"],
     "--extra", json.dumps(j["extra"])]
if j["stats"]:
    a += ["--stats", j["stats"]]
print(" ".join(shlex.quote(x) for x in a))
EOF
)
  echo "[$(date -Is)] $id: start on $(hostname) GPU $GPU cpus $CPUS: $args"
  eval CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU taskset -c $CPUS python "$LAT_CODE/measure_latency.py" $args --eager-stride 10 --graph-stride "$(cat "$LAT/graph_stride" 2>/dev/null || echo 5)" --eager-warmup 5 \
    --text-cache-dir "$LAT_TEXT" --observations "$LAT_OBS" --observations-sha256 "$LAT_OBS_SHA" \
    --output "$out.tmp" > "$LAT/logs/job_$id${LAT_LOG_SUFFIX:-}.log" 2>&1
  rc=$?
  if [ $rc = 0 ]; then mv "$out.tmp" "$out"; echo "[$(date -Is)] $id: done"; else echo "[$(date -Is)] $id: FAILED rc=$rc (logs/job_$id.log)"; fi
done
