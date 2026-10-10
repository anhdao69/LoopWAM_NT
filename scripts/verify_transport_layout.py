#!/usr/bin/env python3
"""Compare 2/4 logical rank layouts on exactly two physical H100s.

Gloo sums CPU gradient vectors to permit two logical ranks per device in the
four-rank smoke. This validates partition/scaling, not four-device NCCL.
"""
import json,os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.utils.data._utils.collate import default_collate
from train_transport import ASSETS,PARENT_RUN,sample_noise
from fastwam.datasets.transport_cached import TransportCachedDataset
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.models.wan22.transport_policy import TransportPolicy
from fastwam.models.wan22.residual_transport import SCHEDULES
rank=int(os.environ["RANK"]);world=int(os.environ["WORLD_SIZE"]);device=rank%2
torch.cuda.set_device(device);torch.set_num_threads(2);torch.manual_seed(42)
dist.init_process_group("gloo")
data=TransportCachedDataset(PARENT_RUN/"train/data/data_manifest.json",PARENT_RUN/"latents")
ids=list(range(rank,8,world))
sample={k:v.cuda() if torch.is_tensor(v) else v for k,v in default_collate([data[i] for i in ids]).items()}
sample.update(action_scale=data.transform.normalizer["action"].scale.cuda(),action_offset=data.transform.normalizer["action"].offset.cuda())
parent=create_loopwam(checkpoint_path=PARENT_RUN/"train/latest.pt",vae_path=ASSETS/"checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",device=f"cuda:{device}")
model=TransportPolicy(parent,kind="T4",scope="pb",checkpoint_blocks=False).cuda()
noise,anchor,t=(x.cuda() for x in sample_noise(ids,0,42))
with torch.autocast("cuda",dtype=torch.bfloat16):loss,_=model(sample,SCHEDULES["S2"],noise,anchor,t)
(loss/8).backward()
grads=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().cpu() for p in model.parameters() if p.requires_grad])
value=loss.detach().float().cpu()/8
dist.all_reduce(grads);dist.all_reduce(value)
if rank==0:
    out=Path("plans/residual_transport_evidence")
    result=dict(logical_ranks=world,physical_gpus=2,windows=8,loss=float(value),gradient_norm=float(grads.norm()))
    (out/f"layout_{world}.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    if world==4:
        previous=json.loads((out/"layout_2.json").read_text())
        result["relative_errors"]={k:abs(result[k]-previous[k])/max(abs(previous[k]),1e-8) for k in ("loss","gradient_norm")}
        assert all(v<.02 for v in result["relative_errors"].values()),result
        result["tolerance"]=.02;result["four_physical_gpu_nccl_tested"]=False
        (out/"layout_verification.json").write_text(json.dumps(result,indent=2))
dist.destroy_process_group()
