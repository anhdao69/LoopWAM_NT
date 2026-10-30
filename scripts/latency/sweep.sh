#!/usr/bin/env bash
# All-model latency sweep inside one overlapping step of an interactive job: one queue worker per listed GPU, each
# pinned to its own share of the step's physical cores (both hyperthreads of a core go to the same worker), sharing
# the priority-ordered job queue.
# Usage (inside the step): sweep.sh <jobs.jsonl> <comma-separated physical GPUs>
set -uo pipefail
JOBS=$1; GPUS=$2
source /groups/yshang/an221229/checkpoints/LoopWAM_NT/latency_20261030/latency_env.sh
cpus=$(python - <<'EOF'
import os
c, order = sorted(os.sched_getaffinity(0)), []
for x in c:
    sib = open(f"/sys/devices/system/cpu/cpu{x}/topology/thread_siblings_list").read().strip()
    for y in sorted(int(s) for s in sib.replace("-", ",").split(",")):
        if y in c and y not in order:
            order.append(y)
print(",".join(map(str, order)))
EOF
)
IFS=, read -ra C <<< "$cpus"; IFS=, read -ra G <<< "$GPUS"
per=$(( ${#C[@]} / ${#G[@]} ))
echo "[$(date -Is)] sweep on $(hostname): GPUs $GPUS, CPUs $cpus, $per CPUs per worker"
for i in "${!G[@]}"; do
  mine=$(IFS=,; echo "${C[*]:$((i * per)):$per}")
  bash "$LAT_CODE/run_queue.sh" "$JOBS" "${G[$i]}" "$mine" > "$LAT/logs/worker_gpu${G[$i]}.log" 2>&1 &
done
wait
echo "[$(date -Is)] sweep finished"
# Interference control: re-measure one model on the first GPU with the other three idle (if time allows).
if [ -n "${LAT_CONTROL_NAME:-}" ] && [ "$(date +%s)" -lt "${LAT_CONTROL_DEADLINE:-0}" ]; then
  id=$(python -c "import json,sys; [print(json.loads(l)['id']) for l in open(sys.argv[1]) if json.loads(l)['name']==sys.argv[2]]" "$JOBS" "$LAT_CONTROL_NAME")
  mine=$(IFS=,; echo "${C[*]:0:$per}")
  echo "[$(date -Is)] control: $LAT_CONTROL_NAME (job $id) alone on GPU ${G[0]} CPUs $mine"
  LAT_RESULTS=$LAT/results_control LAT_LOG_SUFFIX=_control bash "$LAT_CODE/run_jobs.sh" "$JOBS" "${G[0]}" "$mine" "$id"
fi
