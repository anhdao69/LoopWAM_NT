#!/usr/bin/env bash
set -euo pipefail
source scripts/transport_env.sh
for phase in uninterrupted split resume; do
    output=runs/residual_transport/resume_reference_det
    stop=12
    extra=()
    if [[ "$phase" != uninterrupted ]]; then output=runs/residual_transport/resume_split_det; fi
    if [[ "$phase" == split ]]; then stop=2; fi
    if [[ "$phase" == resume ]]; then extra+=(--resume); fi
    torchrun --standalone --nproc_per_node=2 scripts/train_transport.py --run-name RT-A --output-dir "$output" --microbatch 2 --global-batch 4 --workers 2 --max-updates "$stop" --smoke "${extra[@]}" > "plans/residual_transport_evidence/resume_$phase.log" 2>&1
done
python - <<'PY'
import json,torch
from pathlib import Path
root=Path("runs/residual_transport")
a=torch.load(root/"resume_reference_det/latest.pt",map_location="cpu",weights_only=False)
b=torch.load(root/"resume_split_det/latest.pt",map_location="cpu",weights_only=False)
def equal(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
for key in ("weights","optimizer","ema","update","epoch","next_micro","windows_seen","scheduler"):
    assert equal(a[key],b[key]),key
logs=[]
for name in ("resume_reference_det","resume_split_det"):
    logs.append([json.loads(x) for x in (root/name/"metrics.jsonl").read_text().splitlines()])
keys=("update","windows_seen","local","end","grip","anchor","grad_norm","schedule")
assert len(logs[0])==len(logs[1])==12
for x,y in zip(*logs):
    for k in keys: assert x[k]==y[k],(k,x[k],y[k])
result=dict(resumed_updates=10,bit_exact=True,weights=True,optimizer=True,ema=True,losses=True,gradient_norms=True)
Path("plans/residual_transport_evidence/resume_verification.json").write_text(json.dumps(result,indent=2))
print(result)
PY
