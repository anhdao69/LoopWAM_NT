import argparse,json,torch
from pathlib import Path
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.datasets.loopwam_long import build_long_datasets
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--backend',default='inductor');a=p.parse_args()
out=Path(a.output);assert not out.exists()
torch.set_num_threads(4);torch.cuda.set_device(0)
model=create_loopwam(checkpoint_path=a.checkpoint,vae_path='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',device='cuda:0').eval()
_,val,_=build_long_datasets('data/lerobot_v30/libero_10_no_noops_lerobot','data/text_embeds_cache/libero',str(out.with_suffix('')))
s=val[0];kw=dict(input_image=s['video'][:,0],proprio=s['proprio'][0],context=s['context'],context_mask=s['context_mask'],action_horizon=32,num_inference_steps=10,seed=42)
records=[]
def instrument(fn):
 compiled=torch.compile(fn,backend=a.backend,fullgraph=True,**({'mode':'reduce-overhead'} if a.backend=='inductor' else {}))
 def call(**kwargs):
  result=compiled(**kwargs)
  cloned=torch.utils._pytree.tree_map(lambda t:t.clone() if isinstance(t,torch.Tensor) else t,result)
  reference=fn(**kwargs)
  xx=torch.utils._pytree.tree_leaves(cloned);yy=torch.utils._pytree.tree_leaves(reference)
  diffs=[float((x.float()-y.float()).abs().max()) for x,y in zip(xx,yy) if isinstance(x,torch.Tensor)]
  records.append(dict(function=fn.__name__,max_abs=max(diffs),tensor_max_abs=diffs))
  return cloned
 return call
model._prefill_video_cache_compiled=instrument(model.mot.prefill_video_cache_tensor)
model._denoise_action_with_video_cache_compiled=instrument(model._denoise_action_with_video_cache)
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 eager=model.infer_action(**kw)['action'].clone()
 compiled=model.infer_action(**kw,compile_action_infer=True)['action'].clone()
 again=model.infer_action(**kw,compile_action_infer=True)['action'].clone()
r=dict(checkpoint=a.checkpoint,backend=a.backend,mode=model.mot.action_kv_mode,final_max_abs=float((compiled-eager).abs().max()),repeat_bit_identical=torch.equal(compiled,again),within_gate=torch.allclose(compiled,eager,atol=.002,rtol=.002),boundaries=records)
out.write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
