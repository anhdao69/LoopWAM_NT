#!/usr/bin/env python3
"""Synthetic local export/stack roundtrip. Never publish these test checkpoints."""
import copy,gc,json
from pathlib import Path
import torch
from torch.utils.data._utils.collate import default_collate
from train_transport import ASSETS,PARENT_RUN,export_epoch,initialize_stacked,sample_noise
from transport_experiments import EXPERIMENTS
from fastwam.datasets.transport_cached import TransportCachedDataset
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.models.wan22.transport_policy import TransportPolicy
from fastwam.models.wan22.transport_inference import TransportInference
from fastwam.models.wan22.residual_transport import SCHEDULES,merge_lora

torch.set_num_threads(4); torch.manual_seed(42)
out=Path("runs/residual_transport/export_verification");out.mkdir(parents=True,exist_ok=True)
vae=ASSETS/"checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
parent=create_loopwam(checkpoint_path=PARENT_RUN/"train/latest.pt",vae_path=vae,device="cuda:0")
data=TransportCachedDataset(PARENT_RUN/"train/data/data_manifest.json",PARENT_RUN/"latents")
sample={k:v.cuda() for k,v in default_collate([data[0]]).items() if torch.is_tensor(v)}
noise=sample_noise([0],0,42)[0].cuda()
model=TransportPolicy(parent,kind="T4",scope="pb",checkpoint_blocks=False).cuda().eval()
recovery=torch.load("runs/residual_transport/verified_bench_A64/latest.pt",map_location="cpu",weights_only=False)
with torch.no_grad():
    for name,p in model.named_parameters():
        if p.requires_grad:p.copy_(recovery["weights"][name])
    with torch.autocast("cuda",dtype=torch.bfloat16):
        expected={name:model.rollout(sample,SCHEDULES[name],noise).cpu() for name in ("S1","S2","parent10")}
export_epoch(model,out,8,EXPERIMENTS["RT-A"],17360,recovery["contract"])
del model,parent;gc.collect();torch.cuda.empty_cache()
infer=TransportInference(out/"epoch_08.pt",vae)
errors={}
for name in expected:
    actual=infer.actions(sample,noise,name).cpu()
    errors[name]=float((actual-expected[name]).abs().max())
    torch.testing.assert_close(actual,expected[name],atol=0,rtol=0)
    graphed=infer.actions(sample,noise,name,True).cpu()
    torch.testing.assert_close(graphed,expected[name],atol=0,rtol=0)
print(json.dumps(dict(export_errors=errors)),flush=True)
# B2a uses full-action tuning, so export/init must copy weights and conditioning exactly.
parent=infer.parent
b2a=TransportPolicy(parent,kind="T0",scope="pc",step_cond=True,checkpoint_blocks=False).cuda()
with torch.no_grad(): b2a.step_condition.net[-1].weight.normal_(std=1e-4)
export_epoch(b2a,out,10,EXPERIMENTS["RT-B2a"],21700,recovery["contract"])
payload=torch.load(out/"epoch_10.pt",map_location="cpu",weights_only=False,mmap=True)
stack=TransportPolicy(parent,kind="T4",scope="pb",step_cond=True,checkpoint_blocks=False).cuda()
teacher_before={k:v.detach().clone() for k,v in parent.action_expert.state_dict().items()}
initialize_stacked(stack,payload,"cuda:0")
merged=merge_lora(copy.deepcopy(stack.student.mixtures["action"]).cpu()).state_dict()
for k,v in b2a.student.mixtures["action"].state_dict().items():
    torch.testing.assert_close(merged[k],v.cpu(),atol=0,rtol=0)
for k,v in stack.step_condition.state_dict().items():
    torch.testing.assert_close(v,b2a.step_condition.state_dict()[k],atol=0,rtol=0)
for k,v in parent.action_expert.state_dict().items():
    torch.testing.assert_close(v,teacher_before[k],atol=0,rtol=0)
result=dict(export_max_absolute_error=errors,stack_exact=True,teacher_unchanged=True,synthetic_checkpoint_not_for_publication=True)
Path("plans/residual_transport_evidence/export_verification.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
