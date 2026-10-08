#!/usr/bin/env python3
"""Compare native BF16 eager and fullgraph-compiled action inference."""
import argparse,json
from pathlib import Path
import torch
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.datasets.loopwam_long import build_long_datasets


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 if Path(a.output).exists():raise ValueError('Fresh output required')
 torch.set_num_threads(4);torch.cuda.set_device(0)
 model=create_loopwam(checkpoint_path=a.checkpoint,vae_path='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',device='cuda:0').eval()
 _,val,_=build_long_datasets('data/lerobot_v30/libero_10_no_noops_lerobot','data/text_embeds_cache/libero',str(Path(a.output).with_suffix('')))
 sample=val[0];kwargs=dict(input_image=sample['video'][:,0],proprio=sample['proprio'][0],context=sample['context'],context_mask=sample['context_mask'],action_horizon=32,num_inference_steps=10,seed=42)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  eager=model.infer_action(**kwargs)['action'].float().clone()
  compiled=model.infer_action(**kwargs,compile_action_infer=True)['action'].float().clone()
  again=model.infer_action(**kwargs,compile_action_infer=True)['action'].float().clone()
 torch.testing.assert_close(compiled,eager,atol=2e-3,rtol=2e-3)
 torch.testing.assert_close(again,compiled,atol=0,rtol=0)
 Path(a.output).write_text(json.dumps(dict(mode=model.mot.action_kv_mode,passed=True,max_abs_difference=float((compiled-eager).abs().max()),atol=.002,rtol=.002,repeat_bit_identical=True),indent=2))
if __name__=='__main__':main()
