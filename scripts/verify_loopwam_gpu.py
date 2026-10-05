"""Full-size checkpoint reconstruction and held-out action inference acceptance."""
import argparse
import json
from pathlib import Path
import torch
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.datasets.loopwam_long import build_long_datasets
from train_loopwam import evaluate_open_loop

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',required=True)
p.add_argument('--output-dir',required=True)
a=p.parse_args()
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
model=create_loopwam(checkpoint_path=a.checkpoint,
    vae_path='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',device='cuda',model_dtype=torch.float32)
_,val,_=build_long_datasets('data/lerobot_v30/libero_10_no_noops_lerobot','data/text_embeds_cache/libero',str(out/'data'))
metrics=evaluate_open_loop(model,val,2,42)
print(json.dumps({'checkpoint_reconstructed':True,'policy_parameters':sum(p.numel() for p in model.policy_parameters()),**metrics}),flush=True)
(out/'result.json').write_text(json.dumps(metrics,indent=2))
