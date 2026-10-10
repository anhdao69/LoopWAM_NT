#!/usr/bin/env python3
"""Native H100 numerical contracts, no simulator evaluation."""
import gc
import json
from pathlib import Path
import torch
from torch.utils.data._utils.collate import default_collate
from train_transport import ASSETS,PARENT_RUN,sample_noise
from transport_experiments import EXPERIMENTS
from fastwam.datasets.transport_cached import TransportCachedDataset
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.models.wan22.transport_policy import TransportPolicy
from fastwam.models.wan22.residual_transport import SCHEDULES

def main():
    torch.cuda.set_device(0); torch.set_num_threads(4); torch.manual_seed(42)
    data=TransportCachedDataset(PARENT_RUN/"train/data/data_manifest.json",PARENT_RUN/"latents")
    sample={k:v.cuda() if torch.is_tensor(v) else v for k,v in default_collate([data[0]]).items()}
    sample.update(action_scale=data.transform.normalizer["action"].scale.cuda(),
                  action_offset=data.transform.normalizer["action"].offset.cuda())
    parent=create_loopwam(checkpoint_path=PARENT_RUN/"train/latest.pt",
        vae_path=ASSETS/"checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",device="cuda:0")
    noise,anchor,t=tuple(x.cuda() for x in sample_noise([0],0,42))
    result={}
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        policy=TransportPolicy(parent,kind="T0",checkpoint_blocks=False).eval()
        context,mask,keys,values,attention=policy.prefill(sample)
        expected=noise.clone()
        times,deltas=parent.infer_action_scheduler.build_inference_schedule(10,noise.device,noise.dtype)
        for time,delta in zip(times,deltas):
            v=parent._denoise_action_with_video_cache(expected,time[None],context,mask,keys,values,attention)
            expected=parent.infer_action_scheduler.step(v,delta,expected)
        actual=policy.rollout(sample,SCHEDULES["parent10"],noise)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        result["cold_parent_bit_exact"]=True
        del policy
    for name,cfg in EXPERIMENTS.items():
        torch.manual_seed(42)
        policy=TransportPolicy(parent,**{k:cfg[k] for k in ("kind","scope","regime","step_cond","anchor")},
                               checkpoint_blocks=True,loss_weights=cfg.get("loss_weights")).cuda().train()
        for sched in cfg["schedules"]:
            policy.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss,logs=policy(sample,SCHEDULES[sched],noise,anchor,t)
            assert torch.isfinite(loss),name
            loss.backward()
            grads=[p.grad for p in policy.parameters() if p.requires_grad and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads),name
            # Teacher and video remain frozen, even for full-action fine-tuning.
            assert all(p.grad is None for p in parent.parameters()),name
            result[name+"/"+sched]=dict(loss=float(loss),gradient_norm=float(torch.stack([g.float().norm() for g in grads]).norm()),
                trainable=sum(p.numel() for p in policy.parameters() if p.requires_grad))
            print(json.dumps({name+"/"+sched:result[name+"/"+sched]}),flush=True)
        del policy; gc.collect(); torch.cuda.empty_cache()
    Path("plans/residual_transport_evidence/native_variants.json").write_text(json.dumps(result,indent=2))
    print("Native contracts passed",flush=True)
if __name__=="__main__": main()
