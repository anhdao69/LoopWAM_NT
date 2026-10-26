#!/bin/bash
set -euo pipefail
R=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_centered_20261026
while squeue --steps -h -j 4925 -o %i | grep -qx 4925.199; do sleep 5; done
python3 - <<'PY'
from pathlib import Path
import json
r=Path("/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_centered_20261026")
assert not list((r/"clean_benchmark").glob("*_failed.json"))
trials=json.loads((r/"bench_A.json").read_text())+json.loads((r/"clean_benchmark/bench_B.json").read_text())
assert len(trials)==10
selected=json.loads((r/"selected.json").read_text())
assert selected["config"]==min((t for t in trials if t["status"]=="passed"),key=lambda t:t["seconds"])["config"]
selected["reason"]="Fastest of DDP and uncontended ZeRO-1/2 sweep; global128 and FP32 state verified"
(r/"selected.json").write_text(json.dumps(selected,indent=2))
PY
exec srun --jobid=4925 --overlap --ntasks=1 --cpus-per-task=48 --gres=gpu:4 --job-name=test_training bash "$R/lane.sh" B smoke
