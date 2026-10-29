#!/usr/bin/env bash
# Protocol parity check for the elastic evaluation stack: re-runs the first 10 episodes of two scored RT-A2 S1 groups
# (libero_spatial and libero_10, seed 42, task 0) with the elastic branch's runner and environment, then compares every
# episode (success, policy steps, queries) against the stored H100 RT records. Episodes of a group run sequentially in
# one seeded environment, so a matching stack must reproduce the first episodes exactly.
# Like the RT H100 evaluation, it runs without --verify-text-hash (RT-A2 was trained on the remote server's text
# cache, whose files hash differently from the H100 copy).
# Usage (inside an overlapping step of the eval job): parity_rt.sh <physical GPU>
set -uo pipefail
GPU=$1
source /groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work/eval_env.sh
cd "$ELASTIC_ROOT"
OUT=${PARITY_OUT:-/groups/yshang/an221229/checkpoints/LoopWAM_NT/elastic_work/parity_rt}
REF=/groups/yshang/an221229/checkpoints/LoopWAM_NT/rt_eval_h100/final_rollout
SHA=25ad8ca4705e41c805464f9d7fe99e4a3c4264f192a9f205612078b75a208c3d
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU nice -n 10 python scripts/evaluate_rt_matched.py --tag parity_rollout \
  --conditions RT-A2:S1:cuda-graph --expected-sha "RT-A2=$SHA" --output-root "$OUT" \
  --weights-root /groups/yshang/an221229/checkpoints/LoopWAM_NT --vae-path "$ELASTIC_VAE" --text-cache-dir "$EVAL_TEXT" \
  --gpus "$GPU" --workers-per-gpu 2 --suites libero_spatial,libero_10 --seeds 42 --tasks 0 \
  --episodes 10 --no-videos > "$OUT/runner.log" 2>&1
echo "runner rc=$?"
python - "$OUT" "$REF" "$SHA" <<'EOF'
import json, sys, glob
from pathlib import Path
out, ref, sha = sys.argv[1:]
key = f"residual_trans__RT-A2__S1__{sha}"
new_dirs = glob.glob(f"{out}/*/{key}") + glob.glob(f"{out}/{key}")
assert len(new_dirs) == 1, new_dirs
new_m = json.loads((Path(new_dirs[0]) / "manifest.json").read_text())
ref_m = json.loads((Path(ref) / key / "manifest.json").read_text())
print("depths equal:", new_m["depths"] == ref_m["depths"], new_m["depths"])
skip = {"videos", "episodes", "seeds", "suites", "tasks"}
print("protocol equal except subset/videos:", {k: v for k, v in new_m["protocol"].items() if k not in skip} ==
      {k: v for k, v in ref_m["protocol"].items() if k not in skip})
fields = ("success", "policy_steps", "queries", "initial_state_index", "seed")
total = same = 0
for g in ("libero_spatial_seed42_task0", "libero_10_seed42_task0"):
    a = json.loads((Path(new_dirs[0]) / "groups" / f"{g}.json").read_text())["episodes"]
    b = json.loads((Path(ref) / key / "groups" / f"{g}.json").read_text())["episodes"][:len(a)]
    for x, y in zip(a, b):
        total += 1; ok = all(x[f] == y[f] for f in fields); same += ok
        if not ok: print("MISMATCH", g, {f: (x[f], y[f]) for f in fields})
    print(g, "episodes", len(a), "successes new/ref", sum(e["success"] for e in a), sum(e["success"] for e in b),
          "steps new", [e["policy_steps"] for e in a], "ref", [e["policy_steps"] for e in b])
print(f"PARITY {same}/{total} episodes identical (success, policy_steps, queries)")
EOF
