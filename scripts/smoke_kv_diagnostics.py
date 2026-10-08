#!/usr/bin/env python3
"""Exercise native epoch diagnostics and prove RNG/parameters are unchanged."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.datasets.loopwam_long import build_long_datasets
from loopwam_kv_diagnostics import collect_kv_diagnostics


def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);args=p.parse_args()
 if Path(args.output).exists():raise ValueError('Fresh diagnostic output required')
 torch.set_num_threads(4);torch.cuda.set_device(0)
 model=create_loopwam(checkpoint_path=args.checkpoint,vae_path='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',device='cuda:0').train()
 _,validation,_=build_long_datasets('data/lerobot_v30/libero_10_no_noops_lerobot','data/text_embeds_cache/libero',str(Path(args.output).with_suffix('')))
 cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state();rng=random.getstate();np_rng=np.random.get_state()
 versions=[p._version for p in model.parameters()]
 result=collect_kv_diagnostics(model,validation,0)
 assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
 assert rng==random.getstate() and np.array_equal(np_rng[1],np.random.get_state()[1])
 assert versions==[p._version for p in model.parameters()] and model.training
 result['rng_and_parameters_unchanged']=True
 Path(args.output).write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
