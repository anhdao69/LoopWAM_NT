#!/usr/bin/env bash
# Pre-launch validation of the RT50-ep5 queue on GPUs 0-3 (one -c120 --cpu-bind=none --gres=gpu:8 step of job 5758).
#  1. preflight_rt50 --family RT50-ep5: identity/dataset/teacher(=deployed infer_action)/init on GPU 0; the speed options
#     bit-exact for E0/E1/E2 on GPUs 1-3
#  2. resume equivalence with the production layout and flags (world 4, mb 32), RT50-ep5-E2: A = 30 updates straight,
#     B = 20 updates then --resume to 30; metrics, weights, optimizer, EMA and RNG must be bit-identical
#  3. exports from A's latest.pt through the trainer's export_epoch (update-2000 and epoch layouts) verified by
#     verify_rt50_export.py (static + loader rollout == training model, FLAT9) plus the all-padding microbatch check
set -uo pipefail
source "$(dirname "$0")/env_ep5.sh"
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
V=$ERT_EP5/prelaunch; mkdir -p "$V"; cd "$ELASTIC_ROOT"
cpus() { case $1 in 0) echo 0-7,64-70;; 1) echo 8-14,71-78;; 2) echo 15-22,79-85;; 3) echo 23-29,86-93;; esac; }
rc() { echo "$1 rc=$2" >> "$V/validate.rc"; }
one() { local gpu=$1 name=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu taskset -c "$(cpus "$gpu")" python -u "$@" > "$V/$name.log" 2>&1; rc "$name" $?; }
( while sleep 20; do echo "$(date -u +%T) $(free -g | awk '/Mem:/{print "used",$3,"avail",$7}') job_mem_GiB $(( $(cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | sed -E 's#(/job_[0-9]+).*#\1#')/memory.current) >> 30 ))"; done ) > "$V/memory.log" 2>&1 &
MEM=$!
: > "$V/validate.rc"
# Wait on these PIDs only: a bare `wait` would also wait for the endless memory sampler.
P=()
one 0 preflight_core scripts/preflight_rt50.py --family RT50-ep5 --common-init "$ELASTIC_INIT" --checks identity,dataset,teacher,init --out "$V/preflight_core.json" &
P+=($!)
for i in 0 1 2; do
  one $((i + 1)) preflight_speed_E$i scripts/preflight_rt50.py --family RT50-ep5 --common-init "$ELASTIC_INIT" --checks "" \
    --speed-runs RT50-ep5-E$i --out "$V/preflight_speed_E$i.json" &
  P+=($!)
done
wait "${P[@]}"
train() {  # OUT MAX_UPDATES [--resume]
  CUDA_VISIBLE_DEVICES=0,1,2,3 BIND_LOG=$1.binding.log BIND_TAG=validate torchrun --standalone --nproc_per_node=4 --no-python \
    "$(dirname "$0")/bind_gpu.sh" scripts/train_transport.py --run-name RT50-ep5-E2 --output-dir "$1" --teacher "$ELASTIC_TEACHER" \
    --vae-path "$ELASTIC_VAE" --manifest "$ELASTIC_MANIFEST" --latent-dir "$ELASTIC_LATENTS" --init-artifact "$ELASTIC_INIT" \
    --expected-gpus 4 --microbatch 32 --workers 3 --epochs 1 --save-every 10 --max-updates "$2" --no-checkpoint-blocks \
    --bf16-frozen --teacher-graph --no-fill-uninitialized --gc-freeze --hf-folder not-published "${@:3}" >> "$1.log" 2>&1
}
R=$V/resume; mkdir -p "$R"
train "$R/A" 30; rc resume_A $?
train "$R/B" 20; rc resume_B1 $?
cp "$R/B/latest.pt" "$R/B_update20_latest.pt"
train "$R/B" 30 --resume; rc resume_B2 $?
python - "$R" > "$V/resume_compare.log" 2>&1 <<'PY'; rc resume_compare $?
import json, sys, torch
from pathlib import Path
R = Path(sys.argv[1])
def rows(d): return [json.loads(l) for l in (R / d / "metrics.jsonl").read_text().splitlines()]
A, B = rows("A"), rows("B")
skip = {"seconds", "time", "data_wait_seconds", "peak_allocated_gb"}   # wall clock and allocator high-water marks
metrics_equal = len(A) == len(B) == 30 and all({k: v for k, v in a.items() if k not in skip} == {k: v for k, v in b.items() if k not in skip}
                                               for a, b in zip(A, B))
import numpy as np
def eq(x, y):
    if torch.is_tensor(x): return torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, np.ndarray): return isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y)   # numpy RNG state
    if isinstance(x, dict): return isinstance(y, dict) and x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)): return type(x) == type(y) and len(x) == len(y) and all(eq(p, q) for p, q in zip(x, y))
    return x == y
a, b = (torch.load(R / d / "latest.pt", map_location="cpu", weights_only=False) for d in ("A", "B"))
parts = {k: eq(a[k], b[k]) for k in ("weights", "optimizer", "ema", "rng", "contract", "update", "epoch", "next_micro", "windows_seen", "status", "scheduler")}
b20 = torch.load(R / "B_update20_latest.pt", map_location="cpu", weights_only=False)
moved = sum(not torch.equal(b20["weights"][k], a["weights"][k]) for k in a["weights"])
result = dict(metrics_equal=metrics_equal, updates=(len(A), len(B)), parts=parts, tensors_changed_20_to_30=f"{moved}/{len(a['weights'])}",
              resumed_launch=[json.loads(l)["event"] for l in (R / "B.log").read_text().splitlines() if l.startswith('{"event": "stopped"')][:2],
              passed=metrics_equal and all(parts.values()) and moved > 0 and a["update"] == 30)
print(json.dumps(result, indent=1)); sys.exit(0 if result["passed"] else 1)
PY
one 0 verify_early scripts/verify_rt50_export.py --run-dir "$R/A" --pairs written:latest.pt --write-export early --padding-check \
  --out "$V/verify_early.json" &
P=($!)
one 1 verify_epoch scripts/verify_rt50_export.py --run-dir "$R/A" --pairs written:latest.pt --write-export epoch \
  --out "$V/verify_epoch.json" &
P+=($!)
wait "${P[@]}"
kill $MEM 2>/dev/null
echo VALIDATE_DONE >> "$V/validate.rc"
