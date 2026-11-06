#!/usr/bin/env bash
# Throughput diagnosis inside job 5623 (read-only).
E=/mnt/data/vmo-ai-task/anhdh35/elastic_cl0/eval
date -u
python3 - <<'PY'
import json, glob
for p in sorted(glob.glob("/mnt/data/vmo-ai-task/anhdh35/elastic_cl0/eval/final_rollout/*/groups/*.json")):
    g = json.load(open(p)); e = g["episodes"]
    st = sum(x["policy_steps"] for x in e); q = sum(x["queries"] for x in e)
    qs = sum(x["query_seconds"] for x in e); rs = sum(x["rollout_seconds"] for x in e)
    print("%-70s %s/%s  %.0fs  query %.1f ms/q  sim %.1f ms/step" % (p.split("final_rollout/")[1][:70], g["successes"], len(e),
          g["elapsed_seconds"], 1e3 * qs / q, 1e3 * (rs - qs) / (st + 30 * len(e))))
PY
echo "--- job cpus"; cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || grep Cpus_allowed_list /proc/self/status
lscpu | grep -E "^(CPU\(s\)|Thread|Core|Socket|Model name|NUMA node[0-9] CPU)"
echo "--- cgroup cpu.stat"; P=$(cat /proc/self/cgroup | cut -d: -f3); for d in /sys/fs/cgroup$P /sys/fs/cgroup$(dirname $P) /sys/fs/cgroup$(dirname $(dirname $P)); do [ -f $d/cpu.stat ] && { echo $d; head -1 $d/cpu.stat; }; done
a=$(awk '/usage_usec/{print $2}' /sys/fs/cgroup$(dirname $(dirname $P))/cpu.stat); sleep 10; b=$(awk '/usage_usec/{print $2}' /sys/fs/cgroup$(dirname $(dirname $P))/cpu.stat); echo "job cpu use over 10s: $(( (b-a)/100000 ))% (x100 = cores)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader
nvidia-smi dmon -c 3 -s u 2>/dev/null | tail -5
ps -eo pcpu,psr,comm --sort=-pcpu | awk '$3=="python"' | awk '{n++; s+=$1} END{print n" python procs, total %cpu "s}'
PY2=1
