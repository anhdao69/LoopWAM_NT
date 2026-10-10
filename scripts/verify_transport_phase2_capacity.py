#!/usr/bin/env python3
"""Per-device phase-2 capacity and gradient coverage, without simulator evaluation."""
import argparse,gc,json,time
from pathlib import Path
import torch
from torch.utils.data._utils.collate import default_collate
from train_transport import ASSETS,PARENT_RUN,sample_noise
from transport_experiments import EXPERIMENTS
from fastwam.datasets.transport_cached import TransportCachedDataset
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.models.wan22.transport_policy import TransportPolicy
from fastwam.models.wan22.residual_transport import SCHEDULES
p=argparse.ArgumentParser();p.add_argument("--run",required=True);p.add_argument("--microbatch",type=int,required=True);a=p.parse_args()
torch.set_num_threads(4);torch.cuda.set_device(0);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
cfg=EXPERIMENTS[a.run]
data=TransportCachedDataset(PARENT_RUN/"train/data/data_manifest.json",PARENT_RUN/"latents")
ids=list(range(a.microbatch))
sample={k:v.cuda() if torch.is_tensor(v) else v for k,v in default_collate([data[i] for i in ids]).items()}
sample.update(action_scale=data.transform.normalizer["action"].scale.cuda(),action_offset=data.transform.normalizer["action"].offset.cuda())
parent=create_loopwam(checkpoint_path=PARENT_RUN/"train/latest.pt",vae_path=ASSETS/"checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",device="cuda:0")
model=TransportPolicy(parent,**{k:cfg[k] for k in ("kind","scope","regime","step_cond","anchor")},checkpoint_blocks=False).cuda()
opt=torch.optim.AdamW(model.optimizer_groups(),betas=(.9,.95),eps=1e-8,fused=True)
ema={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
schedule="S1" if a.run=="RT-A2" else "parent2"
records=[]
for step in range(2):
    noise,anchor,t=(x.cuda() for x in sample_noise(ids,step,42))
    start=time.perf_counter()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        loss,_=model(sample,SCHEDULES[schedule],noise,anchor,t)
    (loss/a.microbatch).backward()
    missing=[n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing,missing
    grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    opt.step();opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    record=dict(run=a.run,microbatch=a.microbatch,schedule=schedule,step=step,
        loss=float(loss)/a.microbatch,gradient_norm=float(grad),
        peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
        seconds=time.perf_counter()-start,all_trainable_parameters_have_gradients=True)
    records.append(record);print(json.dumps(record),flush=True)
Path(f"plans/residual_transport_evidence/capacity_{a.run}_{a.microbatch}.json").write_text(json.dumps(records,indent=2))
