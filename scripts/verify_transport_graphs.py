#!/usr/bin/env python3
"""Fifty changing queries per schedule: CUDA graph against eager, no simulator."""
import gc
import json
from pathlib import Path
import torch
from torch.utils.data._utils.collate import default_collate
from train_transport import ASSETS,PARENT_RUN,sample_noise
from fastwam.datasets.transport_cached import TransportCachedDataset
from fastwam.models.wan22.transport_inference import TransportInference
from fastwam.models.wan22.residual_transport import SCHEDULES,ResidualTransport,StepCondition

def main():
    torch.cuda.set_device(0); torch.set_num_threads(4); torch.manual_seed(42)
    data=TransportCachedDataset(PARENT_RUN/"train/data/data_manifest.json",PARENT_RUN/"latents")
    infer=TransportInference(str(PARENT_RUN/"train/latest.pt"),
        str(ASSETS/"checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),zero_shot_kind="T4")
    result={}
    cases=[("T4",s) for s in SCHEDULES]
    cases += [(kind,s) for kind in ("T0","T1","T2","T3","T5","T6") for s in ("S1","S2")]
    for kind,name in cases:
        infer.graphs.clear(); gc.collect(); torch.cuda.empty_cache()
        infer.policy.transport=ResidualTransport(kind,512).cuda().eval().requires_grad_(False)
        # Verify phase-2 conditioning too; perturb away from zero for a meaningful replay test.
        infer.policy.step_condition=StepCondition(512).cuda().eval().requires_grad_(False)
        with torch.no_grad(): infer.policy.step_condition.net[-1].weight.normal_(std=1e-4)
        maximum=0.
        for query in range(50):
            row=data[(query*5551)%len(data)]
            batch=default_collate([row])
            sample={k:batch[k].cuda() for k in ("first_frame_latents","context","context_mask","proprio")}
            noise=sample_noise([query],1,42)[0].cuda()
            eager=infer.actions(sample,noise,name,False)
            graphed=infer.actions(sample,noise,name,True)
            torch.testing.assert_close(graphed,eager,atol=2e-5,rtol=2e-5)
            maximum=max(maximum,float((graphed-eager).abs().max()))
        result[kind+"/"+name]=dict(queries=50,max_absolute_error=maximum)
        print(json.dumps({kind+"/"+name:result[kind+"/"+name]}),flush=True)
    Path("plans/residual_transport_evidence/graph_verification.json").write_text(json.dumps(result,indent=2))
if __name__=="__main__": main()
