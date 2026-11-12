#!/usr/bin/env bash
# Pre-launch validation of GR1-E0 x5 on GPUs 4-7 (one -c120 --cpu-bind=none --gres=gpu:8 step of job 5758), the RT50
# validation (validate_e10x5.sh) for the GR1 port plus the eval-server checks:
#  1. preflight_gr1: identity/dataset(= GR1 v0 trainer fields)/teacher(= deployed infer_action)/server(elastic server
#     adapter = deployed LoopWAMGR1Policy on full observations)/init on GPU 4; speed options bit-exact for GR1-E0 on GPU 5
#  2. resume equivalence with the production layout and flags (world 4, mb 32, --epochs 5): A = 30 updates straight,
#     B = 20 updates then --resume to 30; metrics, weights, optimizer, EMA and RNG must be bit-identical
#  3. schedule contract of A: planned 235,160 updates (5 x 47,032), warmup 7,055, a 235,160-entry schedule sequence,
#     the logged LRs = base LR x lr_factor(update) of the trainer, teacher = selection, recipe GR1-E0
#  4. exports from A's latest.pt through the trainer's export_epoch (update-2000 and epoch layouts) verified by
#     verify_gr1_export.py (static + loader rollout == training model, FLAT9; eval server == loader rollout for the
#     three eval schedules) plus the all-padding microbatch check
#  5. eval smoke: run_eval.py --elastic-schedule flat_k1_n2 on the epoch-layout export, 2 tasks x 2 episodes, EGL on
#     GPUs 4-7 (server metadata, query count, success list)
#  6. (reported, not required for training) the --finalize-only recovery path of train_gr1e.sh: a copy of A's latest.pt
#     forged to the end of epoch 1 (epoch 1, next_micro 0, update 47,032) with a stale status.json; --resume
#     --finalize-only must write epoch_01.pt and status.json from it, train nothing and leave latest.pt unchanged; the
#     export is then verified with verify_gr1_export.py
set -uo pipefail
source "$(dirname "$0")/env_gr1e.sh"
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK
V=$EGR_WORK/prelaunch; mkdir -p "$V"; cd "$ELASTIC_ROOT"
busy=1
for i in $(seq 90); do
  busy=$(nvidia-smi -i "$GR1E_GPUS" --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
  [ "$busy" = 0 ] && break; sleep 10
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > "$V/gpus_at_start.txt"
[ "$busy" = 0 ] || { echo "GPUs $GR1E_GPUS still in use after 15 min" > "$V/validate.rc"; exit 3; }
BIND=$ELASTIC_ROOT/plans/elastic_robotwin/bind_gpu.sh
rc() { echo "$1 rc=$2" >> "$V/validate.rc"; }
one() { local gpu=$1 name=$2; shift 2   # one process on one physical GPU, pinned to that GPU's 15 NUMA-1 CPUs
  CUDA_VISIBLE_DEVICES=$gpu LOCAL_RANK=0 RANK=0 BIND_LOG=$V/binding.log BIND_TAG=$name bash "$BIND" "$@" > "$V/$name.log" 2>&1
  rc "$name" $?; }
( while sleep 20; do echo "$(date -u +%T) $(free -g | awk '/Mem:/{print "used",$3,"avail",$7}') job_mem_GiB $(( $(cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | sed -E 's#(/job_[0-9]+).*#\1#')/memory.current) >> 30 ))"; done ) > "$V/memory.log" 2>&1 &
MEM=$!
: > "$V/validate.rc"
echo "start $(date -u +%FT%TZ) revision $(git rev-parse HEAD) teacher $ELASTIC_TEACHER init $ELASTIC_INIT selection $(sha256sum "$ELASTIC_SELECTION")" > "$V/validate.info"
P=()
one 4 preflight_core scripts/preflight_gr1.py --common-init "$ELASTIC_INIT" --reference-init-json "$ELASTIC_REFERENCE_INIT_JSON" \
  --checks identity,dataset,teacher,server,init --out "$V/preflight_core.json" &
P+=($!)
one 5 preflight_speed_E0 scripts/preflight_gr1.py --common-init "$ELASTIC_INIT" --checks "" --speed-runs GR1-E0 \
  --out "$V/preflight_speed_E0.json" &
P+=($!)
wait "${P[@]}"
train() {  # OUT MAX_UPDATES [--resume]
  CUDA_VISIBLE_DEVICES=$GR1E_GPUS BIND_LOG=$1.binding.log BIND_TAG=validate ${TRAIN_TIMEOUT:+timeout --kill-after=60 $TRAIN_TIMEOUT} \
    torchrun --standalone --nproc_per_node=4 --no-python \
    "$BIND" scripts/train_transport.py --run-name "$GR1E_RUN" --output-dir "$1" --teacher "$ELASTIC_TEACHER" \
    --teacher-selection "$ELASTIC_SELECTION" --vae-path "$ELASTIC_VAE" --manifest "$ELASTIC_MANIFEST" --latent-dir "$ELASTIC_LATENTS" \
    --init-artifact "$ELASTIC_INIT" --expected-gpus 4 --microbatch 32 --workers 3 --epochs "$GR1E_EPOCHS" --save-every 10 \
    --max-updates "$2" --no-checkpoint-blocks --bf16-frozen --teacher-graph --no-fill-uninitialized --gc-freeze \
    --hf-folder not-published "${@:3}" >> "$1.log" 2>&1
}
R=$V/resume; mkdir -p "$R"
train "$R/A" 30; rc resume_A $?
train "$R/B" 20; rc resume_B1 $?
cp "$R/B/latest.pt" "$R/B_update20_latest.pt"
train "$R/B" 30 --resume; rc resume_B2 $?
python - "$R" > "$V/resume_compare.log" 2>&1 <<'PY'; rc resume_compare $?
import json, sys, torch
import numpy as np
from pathlib import Path
R = Path(sys.argv[1])
def rows(d): return [json.loads(l) for l in (R / d / "metrics.jsonl").read_text().splitlines()]
A, B = rows("A"), rows("B")
skip = {"seconds", "time", "data_wait_seconds", "peak_allocated_gb"}   # wall clock and allocator high-water marks
metrics_equal = len(A) == len(B) == 30 and all({k: v for k, v in a.items() if k not in skip} == {k: v for k, v in b.items() if k not in skip}
                                               for a, b in zip(A, B))
def eq(x, y):
    if torch.is_tensor(x): return torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    if isinstance(x, np.ndarray): return isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x, y)   # numpy RNG state
    if isinstance(x, dict): return isinstance(y, dict) and x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)): return type(x) == type(y) and len(x) == len(y) and all(eq(p, q) for p, q in zip(x, y))
    return x == y
a, b = (torch.load(R / d / "latest.pt", map_location="cpu", weights_only=False) for d in ("A", "B"))
parts = {k: eq(a[k], b[k]) for k in ("weights", "ema", "rng", "contract", "update", "epoch", "next_micro", "windows_seen", "status", "scheduler")}
# Optimizer: param_groups, state keys and step counts bit-identical; Adam moments bit-identical or, per tensor, within
# TOL x max|moment| (about 8 float32 ulps at the tensor's scale). After a resume the caching allocator lays tensors out
# differently, so a few gradient reductions (norm / modulation parameters, one matmul weight) sum in another order;
# diagnosed 2026-11-08 (elastic_robotwin/e10x5/diag_resume_20261108): straight runs are bit-identical to each other,
# resumed runs to each other, straight vs resumed differ in 77/684 moment tensors by <= 8.6e-8 of their scale, with
# weights, EMA, RNG and every metric bit-identical. A missing or wrong state would differ by orders of magnitude more.
TOL = 1e-6
oa, ob = a["optimizer"], b["optimizer"]
opt = dict(param_groups=eq(oa["param_groups"], ob["param_groups"]),
           state_keys=oa["state"].keys() == ob["state"].keys() and all(oa["state"][i].keys() == ob["state"][i].keys() for i in oa["state"]))
exact = close = 0; worst = 0.; bad = []
if opt["state_keys"]:
    for i in oa["state"]:
        for k, x in oa["state"][i].items():
            y = ob["state"][i][k]
            if eq(x, y): exact += 1; continue
            if k in ("exp_avg", "exp_avg_sq") and torch.is_tensor(x) and torch.is_tensor(y) and x.dtype == y.dtype and x.shape == y.shape \
                    and bool(torch.isfinite(y).all()):
                rel = float((x.double() - y.double()).abs().max() / x.double().abs().max().clamp_min(1e-300))
                if rel <= TOL: close += 1; worst = max(worst, rel); continue
                bad.append([i, k, rel])
            else: bad.append([i, k, None])
opt.update(state_tensors_exact=exact, moments_within_tolerance=close, worst_relative=worst, tolerance=TOL, out_of_tolerance=bad[:5],
           ok=bool(opt["param_groups"] and opt["state_keys"] and not bad))
parts["optimizer"] = opt["ok"]
b20 = torch.load(R / "B_update20_latest.pt", map_location="cpu", weights_only=False)
moved = sum(not torch.equal(b20["weights"][k], a["weights"][k]) for k in a["weights"])
seconds = [r["seconds"] for r in A[5:]]; waits = [r["data_wait_seconds"] for r in A[5:]]
result = dict(metrics_equal=metrics_equal, updates=(len(A), len(B)), parts=parts, optimizer=opt, tensors_changed_20_to_30=f"{moved}/{len(a['weights'])}",
              mean_seconds_per_update_6_30=sum(seconds) / len(seconds), max_data_wait_6_30=max(waits),
              peak_allocated_gb=max(r["peak_allocated_gb"] for r in A),
              passed=metrics_equal and all(parts.values()) and moved > 0 and a["update"] == 30)
print(json.dumps(result, indent=1)); sys.exit(0 if result["passed"] else 1)
PY
python - "$R/A" "$GR1E_EPOCHS" "$GR1E_PER_EPOCH" "$(sha256sum "$ELASTIC_TEACHER" | cut -d' ' -f1)" "$(sha256sum "$ELASTIC_INIT" | cut -d' ' -f1)" \
  "$ELASTIC_SELECTION" > "$V/schedule_contract.log" 2>&1 <<'PY'; rc schedule_contract $?
import json, math, sys
from pathlib import Path
import torch
from train_loopwam import lr_factor   # the trainer's own schedule (train_transport imports it)
A, epochs, per, tsha, isha, sel = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
total = epochs * per; warm = math.ceil(.03 * total)
cfg = json.loads((A / "config.json").read_text()); st = json.loads((A / "status.json").read_text())
seq = json.loads((A / "schedule_sequence.json").read_text())
rows = [json.loads(l) for l in (A / "metrics.jsonl").read_text().splitlines()]
record = json.loads(Path(sel).read_text())
base = [1e-3, 3e-5]   # optimizer_groups order: step condition 1e-3, then the P-c student expert 3e-5
lrs_ok = all(len(r["lr"]) == 2 and all(abs(l - b * lr_factor(r["update"] - 1, total, warm)) <= 1e-12 * b for l, b in zip(r["lr"], base)) for r in rows)
lat = torch.load(A / "latest.pt", map_location="cpu", weights_only=False, mmap=True)
parent = cfg["config"]["parent"]
checks = dict(
    planned_updates=cfg["planned_updates"] == total == 235160, epochs=cfg["epochs"] == epochs and cfg["arguments"]["epochs"] == epochs,
    windows=cfg["windows"] == 6020058 and math.ceil(cfg["windows"] / 128) == per,
    scheduler=st["scheduler"] == dict(step=30, total=total, warmup=warm) and warm == 7055,
    sequence=seq["updates"] == total and len(seq["sequence"]) == total and seq["sha256"] == cfg["schedule_sequence_sha256"],
    sequence_matches_metrics=all(r["schedule"] == seq["sequence"][r["update"] - 1] for r in rows),
    lr_schedule=lrs_ok,
    teacher=cfg["asset_sha256"]["teacher"] == tsha == parent["sha256"] == record["teacher_sha256"] and parent["epoch"] == record["chosen_epoch"],
    init=cfg["asset_sha256"]["elastic_init"] == isha,
    recipe=cfg["run"] == "GR1-E0" and cfg["config"]["kind"] == "T0" and cfg["config"]["dataset"] == "robocasa_gr1"
           and cfg["config"]["end_prefix"] == 12 and cfg["config"]["gripper_loss"] is False and cfg["action_dim"] == 29,
    grip_zero=all(r["grip"] == 0 for r in rows),
    contract_in_recovery=lat["contract"]["planned_updates"] == total and lat["contract"]["epochs"] == epochs
                         and lat["contract"]["config"]["parent"] == parent)
print(json.dumps(dict(checks=checks, first_lr=rows[0]["lr"], last_lr=rows[-1]["lr"], parent=parent, passed=all(checks.values())), indent=1))
sys.exit(0 if all(checks.values()) else 1)
PY
P=()
one 4 verify_early scripts/verify_gr1_export.py --run-dir "$R/A" --pairs written:latest.pt --write-export early --padding-check \
  --server-schedules flat_k1_n2,flat_k4_n2,flat_k4_n5 --out "$V/verify_early.json" &
P+=($!)
one 5 verify_epoch scripts/verify_gr1_export.py --run-dir "$R/A" --pairs written:latest.pt --write-export epoch \
  --out "$V/verify_epoch.json" &
P+=($!)
wait "${P[@]}"
# eval smoke through the production eval script's code path (run_eval.py --elastic-schedule), 2 tasks x 2 episodes
SM=$V/eval_smoke; CK=$R/A/epoch_00.pt
if [ -f "$CK" ]; then
  EVAL_CK=$CK EVAL_STEP=30 EVAL_OUT=$SM EVAL_EPISODES=2 EVAL_PORT=7950 \
    EVAL_TASKS=gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env,gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env \
    bash "$(dirname "$0")/eval_gr1e.sh" 0 flat_k1_n2 1 > "$V/eval_smoke.log" 2>&1
  python - "$SM" "$CK" > "$V/eval_smoke_check.log" 2>&1 <<'PY'; rc eval_smoke $?
import glob, hashlib, json, sys
from pathlib import Path
out, ck = Path(sys.argv[1]), sys.argv[2]
sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
cells = sorted(glob.glob(str(out / "n12/*.json"))); rows = []
for f in cells:
    d = json.loads(Path(f).read_text()); m = d["server_metadata"]
    rows.append(dict(task=Path(f).stem, ok=d["n_episodes"] == 2 and d["n_action_steps"] == 12 and len(d["successes"]) == 2
        and m["schedule"] == "flat_k1_n2" and m["loops"] == [1, 1] and m["num_inference_steps"] == 2 and m["export_sha256"] == sha
        and m["step"] == 30 and m["noise_seed"] == 42 and m["transport_kind"] == "T0" and 2 * 60 <= d["client"]["queries"] <= 2 * 61
        and d["env"].get("MUJOCO_GL") == "egl", successes=d["successes"], queries=d["client"]["queries"]))
result = dict(cells=len(cells), rows=rows, passed=len(cells) == 2 and all(r["ok"] for r in rows))
json.dump(result, open(out.parent / "eval_smoke.json", "w"), indent=1); print(json.dumps(result, indent=1))
sys.exit(0 if result["passed"] else 1)
PY
else
  rc eval_smoke 9
fi
F=$R/F; rm -rf "$F" "$R/F.log"; mkdir -p "$F"
python - "$R/A" "$F" "$GR1E_PER_EPOCH" > "$V/finalize_setup.log" 2>&1 <<'PY'
import json, shutil, sys, torch
from pathlib import Path
A, F, per = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
x = torch.load(A / "latest.pt", map_location="cpu", weights_only=False)
x.update(epoch=1, next_micro=0, update=per, status="running", scheduler=dict(x["scheduler"], step=per))
torch.save(x, F / "latest.pt")
shutil.copyfile(A / "status.json", F / "status.json")   # stale, as after a stop between the latest.pt and status.json writes
print(json.dumps(dict(forged={k: x[k] for k in ("epoch", "next_micro", "update", "status", "scheduler")})))
PY
srv=$?
LSHA=$(sha256sum "$F/latest.pt" | cut -d' ' -f1)
frc=99; [ $srv = 0 ] && { TRAIN_TIMEOUT=1800 train "$F" "$GR1E_PER_EPOCH" --resume --finalize-only; frc=$?; }
python - "$F" "$LSHA" "$GR1E_PER_EPOCH" > "$V/finalize_check.log" 2>&1 <<'PY'
import hashlib, json, sys, torch
from pathlib import Path
F, lsha, per = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
text = Path(str(F) + ".log").read_text()
st = json.loads((F / "status.json").read_text())
x = torch.load(F / "latest.pt", map_location="cpu", weights_only=False, mmap=True)
want = {k: v for k, v in x.items() if k not in ("format", "weights", "optimizer", "ema", "rng", "contract")}
h = hashlib.sha256()
with open(F / "latest.pt", "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
checks = dict(finalized_event='"event": "finalized"' in text and '"status_rewritten": true' in text,
              status_from_latest=st == want and st["update"] == per and st["epoch"] == 1 and st["next_micro"] == 0,
              latest_unchanged=h.hexdigest() == lsha, export_written=(F / "epoch_01.pt").is_file(),
              nothing_trained=not (F / "metrics.jsonl").exists() and not (F / "config.json").exists(),
              only_expected_checkpoints=sorted(p.name for p in F.glob("*.pt")) == ["epoch_01.pt", "latest.pt"])
print(json.dumps(dict(checks=checks, status=st, passed=all(checks.values())), indent=1)); sys.exit(0 if all(checks.values()) else 1)
PY
crc=$?; rc finalize $(( frc != 0 ? 100 + frc : crc ))
if [ -f "$F/epoch_01.pt" ]; then
  one 4 verify_finalize scripts/verify_gr1_export.py --run-dir "$F" --pairs epoch_01.pt:latest.pt --out "$V/verify_finalize.json"
else
  rc verify_finalize 9
fi
kill $MEM 2>/dev/null
echo VALIDATE_DONE >> "$V/validate.rc"
