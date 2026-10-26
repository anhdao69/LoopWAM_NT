#!/usr/bin/env python3
"""Strict native checkpoint restore and deterministic finite cached inference."""
import argparse
import json
from pathlib import Path
import torch
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.datasets.loopwam_long import build_full_libero_datasets


def inference_inputs(sample):
    return dict(input_image=sample['video'][:,0],proprio=sample['proprio'][0:1],
                context=sample['context'].unsqueeze(0),context_mask=sample['context_mask'].unsqueeze(0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(4);torch.cuda.set_device(0)
    model=create_loopwam(checkpoint_path=a.checkpoint,
        vae_path='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',device='cuda:0').eval()
    train,_,_=build_full_libero_datasets('data/lerobot_v30','data/text_embeds_cache/libero',str(out.parent/'inference_data'))
    sample=train[0]
    def query():
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            return model.infer_action(**inference_inputs(sample),
                action_horizon=32,num_inference_steps=10,text_cfg_scale=1.,seed=42)['action'].float().cpu()
    a1=query();a2=query()
    if a1.shape!=(32,7) or not torch.isfinite(a1).all() or not torch.equal(a1,a2):
        raise ValueError('Invalid/nonrepeatable native action inference')
    result=dict(passed=True,partition=list(model.mot.partition),loops=model.mot.loops,
                effective_depth=model.mot.effective_depth(),parameters=sum(p.numel() for p in model.policy_parameters()),
                action_shape=list(a1.shape),finite=True,repeat_bit_identical=True,
                max_abs=float(a1.abs().max()),gpu=torch.cuda.get_device_name())
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
