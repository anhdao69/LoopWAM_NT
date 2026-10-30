#!/usr/bin/env bash
# Latency queue worker: walks the job list (jobs.py JSON lines, priority order) and runs every job it can claim
# (atomic mkdir of $LAT/claims/<id>) on one GPU, one job at a time; any number of workers on any GPUs share the queue.
# Usage: run_queue.sh <jobs.jsonl> <physical GPU> <cpu list for taskset, e.g. 0-3>
# Needs latency_env.sh sourced. Results: $LAT/results/<id>.json; logs: $LAT/logs/job_<id>.log.
set -uo pipefail
JOBS=$1; GPU=$2; CPUS=$3
mkdir -p "$LAT/claims"
for id in $(python -c "import json,sys; [print(json.loads(l)['id']) for l in open(sys.argv[1])]" "$JOBS"); do
  [ -f "$LAT/results/$id.json" ] && continue
  if [ -n "${LAT_CLAIM_DEADLINE:-}" ] && [ "$(date +%s)" -ge "$LAT_CLAIM_DEADLINE" ]; then
    echo "[$(date -Is)] queue worker GPU $GPU: claim deadline passed"; break
  fi
  mkdir "$LAT/claims/$id" 2>/dev/null || continue
  echo "$(hostname) GPU $GPU $(date -Is)" > "$LAT/claims/$id/owner"
  bash "$LAT_CODE/run_jobs.sh" "$JOBS" "$GPU" "$CPUS" "$id"
done
echo "[$(date -Is)] queue worker GPU $GPU: nothing left"
