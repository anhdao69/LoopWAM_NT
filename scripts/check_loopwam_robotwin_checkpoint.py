"""Native original-v0 strict reload, loss equivalence and cached-observation inference gate."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from fastwam.models.wan22.loopwam_robot import create_loopwam_robot,RobotTrainForward,PARAMETERS
from fastwam.models.wan22.loop_mot import LoopMoT
from fastwam.datasets.chronoloop_robot import RobotWindows,robot_spec,LATENT_SHAPE,latent_provenance
from fastwam.datasets.loopwam_latent_cache import LoopWAMLatentCache
from fastwam.models.wan22.loopwam_init import sha256_file
from train_loopwam_robotwin_full import slot_noise
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--shared',required=True);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(2);torch.cuda.set_device(0);vae='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
model,payload=create_loopwam_robot(vae,checkpoint_path=a.checkpoint,device='cuda:0',checkpoint_blocks=False)
assert type(model.mot) is LoopMoT and sum(p.numel() for p in model.policy_parameters())==PARAMETERS
assert all(p.dtype==torch.float32 for p in model.policy_parameters())
assert payload['optimizer']['state']
assert all(v.dtype==torch.float32 for s in payload['optimizer']['state'].values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq'))
assert not any('memory' in key for key in payload['mot'])
for family,state in [('mot',model.mot.state_dict()),('proprio_encoder',model.proprio_encoder.state_dict())]:
    assert state.keys()==payload[family].keys()
    assert all(torch.equal(v.detach().cpu(),payload[family][k]) for k,v in state.items())
model.mot.structured_attention=True;model.mot.structured_attention_observation_tokens=480
shared=Path(a.shared);w=RobotWindows(robot_spec('robotwin50'),cache_dir=shared/'data',text_threads=4)
cache=LoopWAMLatentCache(shared/'latents',len(w),latent_provenance(w,sha256_file(vae)),latent_shape=LATENT_SHAPE)
f=w.fields(0);sample={k:v.unsqueeze(0).cuda() for k,v in f.items() if isinstance(v,torch.Tensor)}
sample['latents']=torch.from_numpy(np.array(cache._latents[0:1]).view(np.int16)).view(torch.bfloat16).float().cuda()
sample['video']=torch.zeros((),device='cuda').expand(1,3,9,384,320)
noise=slot_noise(model,[42],torch.device('cuda:0'));model.eval()
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
 noisy=model.prepare_training_batch(sample,noise_video=noise['video'],noise_action=noise['action'],timestep_video=noise['t_video'],timestep_action=noise['t_action'])
 original,_=model.reduce_flow_losses(model.forward_exits(noisy),noisy)
 wrapped,_,_=RobotTrainForward(model)([dict(sample=sample,noise=noise)])
 assert torch.equal(original,wrapped) and torch.isfinite(wrapped)
 # Decode the real cached observation to a packed three-camera image; this is a
 # native inference execution check, not a simulator success-rate evaluation.
 image=model.vae.model.decode(sample['latents'][:,:,:1].to(torch.bfloat16),model.vae.scale)[:,:,0].clamp(-1,1)
 model.vae.model.clear_cache()
 kw=dict(input_image=image,proprio=sample['proprio'][0,0],context=sample['context'],context_mask=sample['context_mask'],num_inference_steps=10,seed=42)
 first=model.infer_action(**kw)['action'];second=model.infer_action(**kw)['action']
 assert first.shape==(32,14) and torch.isfinite(first).all() and torch.equal(first,second)
result=dict(passed=True,step=payload['step'],parameters=PARAMETERS,model='original LoopWAM v0 / LoopMoT',policy_dtype='float32',optimizer_moments='float32',model_weights_restored_exact=True,loss_wrapper_exact=True,inference_shape=list(first.shape),inference_finite=True,repeated_seed_exact=True)
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
