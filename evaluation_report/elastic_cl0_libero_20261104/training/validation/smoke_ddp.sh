#!/usr/bin/env bash
# 4-GPU DDP smoke of the real trainer with all speed flags: CL0-E2 straight 30 updates (A) vs 15 + resume to 30 (B),
# then CL0-E0 for 20 updates (C). Compares A and B metrics (updates 1-30) and final recovery weights bit for bit.
source /mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_cl0/plans/elastic_cl0/env.sh
S=$CL0_WORK/validation/smoke; rm -rf $S; mkdir -p $S; cd $ELASTIC_ROOT
export CUDA_VISIBLE_DEVICES=0,1,2,3
SPEED="--bf16-frozen --teacher-graph --gc-freeze --no-fill-uninitialized"
run() { name=$1; out=$2; shift 2
  torchrun --standalone --nproc_per_node=4 scripts/train_transport.py --run-name $name --output-dir $out --teacher $ELASTIC_TEACHER \
    --vae-path $ELASTIC_VAE --manifest $ELASTIC_MANIFEST --latent-dir $ELASTIC_LATENTS --init-artifact $ELASTIC_INIT \
    --expected-gpus 4 --microbatch 32 --workers 3 --save-every 100 --no-checkpoint-blocks $SPEED "$@" >> $out.log 2>&1; }
run CL0-E2 $S/A --max-updates 30; echo "A rc=$?"
run CL0-E2 $S/B --max-updates 15; echo "B1 rc=$?"
run CL0-E2 $S/B --max-updates 30 --resume; echo "B2 rc=$?"
python - $S <<"PY"
import json, sys, torch
from pathlib import Path
S = Path(sys.argv[1])
def rows(p):
    out = {}
    for line in open(p):
        r = json.loads(line); out[r["update"]] = r
    return out
a, b = rows(S/"A/metrics.jsonl"), rows(S/"B/metrics.jsonl")
keys = [k for k in a[1] if k not in ("seconds", "peak_allocated_gb", "measured_action_blocks")]
same = all(a[u][k] == b[u][k] for u in range(1, 31) for k in keys)
wa = torch.load(S/"A/latest.pt", map_location="cpu", weights_only=False); wb = torch.load(S/"B/latest.pt", map_location="cpu", weights_only=False)
wsame = all(torch.equal(wa["weights"][k], wb["weights"][k]) for k in wa["weights"]) and all(torch.equal(wa["ema"][k].cpu(), wb["ema"][k].cpu()) for k in wa["ema"])
sec = sorted(a[u]["seconds"] for u in range(6, 31))
print(json.dumps(dict(metrics_bit_identical=same, weights_and_ema_bit_identical=wsame, updates=len(a), keys=keys,
    median_seconds_A=sec[len(sec)//2], schedules=[a[u]["schedule"] for u in range(1, 31)],
    loss_A=[round(a[u]["local"], 6) for u in (1, 10, 20, 30)], peak_gb=max(a[u]["peak_allocated_gb"] for u in a))))
PY
run CL0-E0 $S/C --max-updates 20; echo "C rc=$?"
tail -n 3 $S/C/metrics.jsonl | cut -c1-300
echo SMOKE_DONE
