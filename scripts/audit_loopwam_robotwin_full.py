"""Fail-closed full-release/cache/schedule and native donor initialization audit."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from fastwam.datasets.chronoloop_robot import RobotWindows,robot_spec,LATENT_SHAPE,latent_provenance
from fastwam.datasets.loopwam_latent_cache import LoopWAMLatentCache
from fastwam.models.wan22.loopwam_init import sha256_file,load_init_artifact,select_donor_layers
from fastwam.models.wan22.loopwam_robot import create_loopwam_robot,PARAMETERS
from fastwam.models.wan22.loop_mot import LoopMoT
from fastwam.datasets.loopwam_robot_batches import training_budget
p=argparse.ArgumentParser();p.add_argument('--shared',required=True);p.add_argument('--output',required=True);a=p.parse_args()
shared=Path(a.shared);out=Path(a.output);out.mkdir(exist_ok=True)
vae='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth';artifact='checkpoints/LoopWAM/wan21_compact_donors.pt'
w=RobotWindows(robot_spec('robotwin50'),cache_dir=shared/'data',text_threads=4)
assert len(w)==6075103 and len(w.episodes)==27500 and len(w.groups)==50
assert all(len(ids)==550 for ids in w.spec.groups.values())
m=w.manifest();assert sha256_file(w.spec.stats_path)=='7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095'
assert m['normalization_sha256']=='38ed9c8d466356fb8f4688da470b14a036dbc51e0fc5484cc5959b507adbcd48'
cache=LoopWAMLatentCache(shared/'latents',len(w),latent_provenance(w,sha256_file(vae)),latent_shape=LATENT_SHAPE)
assert np.all(cache._valid==1)
idx=np.linspace(0,len(w)-1,1024).astype(int)
assert torch.isfinite(torch.from_numpy(np.array(cache._latents[idx]).view(np.int16)).view(torch.bfloat16)).all()
w.warm_text([w.prompt(int(i)) for i in idx])
for i in idx[:16]:
 fields=w.fields(int(i));assert fields['action'].shape==(32,14) and fields['proprio'].shape[-1]==14
per_epoch,total,tail=training_budget(len(w),128,5)
s={'summary':dict(updates=total,valid_windows=len(w)*5,updates_per_epoch=per_epoch,last_batch_windows=tail,epoch_complete_update={str(e):e*per_epoch for e in range(1,6)})}
assert (per_epoch,total,tail)==(47462,237310,95)
print(json.dumps(dict(event='data_passed',summary=s['summary'])),flush=True)
model,_=create_loopwam_robot(vae,init_artifact=artifact,device='cpu',seed=42)
assert type(model.mot) is LoopMoT and not any('memory' in n for n,_ in model.named_parameters())
assert sum(p.numel() for p in model.policy_parameters())==PARAMETERS
bank=load_init_artifact(artifact);checked=0
for name,expert in [('video',model.video_expert),('action',model.action_expert)]:
 expected=select_donor_layers(bank[name+'_state_dict'])
 for key,value in expert.state_dict().items():
  if name=='action' and key.split('.')[0] in ('action_encoder','head'):continue
  assert torch.equal(value,expected[key]),(name,key)
  checked+=1
result=dict(passed=True,model='original LoopWAM v0 / LoopMoT',episodes=27500,tasks=50,windows=6075103,
 cache_valid=6075103,finite_latent_samples=1024,schedule=s['summary'],parameters=PARAMETERS,donor_tensors_equal=checked,
 init_artifact_sha256=sha256_file(artifact),normalization_sha256=m['normalization_sha256'],memory_parameters=0,history_frames=0)
(out/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
