"""Original LoopWAM v0 with fresh 14-D robot interfaces and cached RoboTwin inputs.

No ChronoLoop modules, memory, recurrent state, or history tokens are instantiated.
"""
import copy
import torch
from torch import nn
from .loopwam import LoopWAM
from .loop_mot import LoopMoT
from .loopwam_init import (build_target_experts, load_init_artifact, load_wan21_vae,
                          architecture_metadata_for_version, target_configs, DONOR_INDICES, sha256_file)

FORMAT = 'loopwam-s-robot-v1'
PARAMETERS = 584567886

class RobotLoopWAM(LoopWAM):
    def _encode_training_video(self, sample, input_video, tiled=False):
        latents=sample.get('latents')
        if tiled or latents is None or latents.ndim!=5 or tuple(latents.shape[1:])!=(16,3,48,40):
            raise ValueError('RoboTwin training requires verified cached [B,16,3,48,40] latents')
        return latents

    def save_robot(self,path,optimizer=None,step=None,training_state=None):
        from pathlib import Path
        path=Path(path)
        payload=dict(format_version=FORMAT,version='v0',video_loops=4,action_loops=4,
            inference_loops=4,trained_max_loops=4,exit_weight_scale=1.,memory_tokens=0,history_frame=0,
            architecture=self.architecture_metadata,mot=self.mot.state_dict(),
            proprio_encoder=self.proprio_encoder.state_dict(),step=step,training_state=training_state)
        if optimizer is not None:payload['optimizer']=optimizer.state_dict()
        tmp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,tmp);tmp.replace(path)

def fresh_robot_interfaces(model,seed):
    ref=model.action_expert.action_encoder.weight
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        enc=nn.Linear(14,model.action_expert.hidden_dim)
        head=nn.Linear(model.action_expert.hidden_dim,14)
        prop=nn.Linear(14,model.proprio_encoder.out_features)
    model.action_expert.action_encoder=enc.to(device=ref.device,dtype=ref.dtype)
    model.action_expert.head=head.to(device=ref.device,dtype=ref.dtype)
    model.proprio_encoder=prop.to(device=ref.device,dtype=ref.dtype)
    model.action_expert.action_dim=14;model.proprio_dim=14

def validate_robot_payload(p):
    expected=dict(format_version=FORMAT,version='v0',video_loops=4,action_loops=4,memory_tokens=0,history_frame=0,inference_loops=4,trained_max_loops=4,exit_weight_scale=1.)
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('Not an original v0 4/4 robot checkpoint')
    v,a=target_configs(12);a['action_dim']=14
    m=p['architecture']
    if m['target_video_config']!=v or m['target_action_config']!=a or m.get('robot_proprio_dim')!=14 or m.get('donor_indices')!=list(DONOR_INDICES):
        raise ValueError('Robot checkpoint architecture/donor mismatch')
    if m.get('initialization_mode')!='fresh_wan_donors_no_embodied_pretraining':raise ValueError('Robot initialization provenance mismatch')

def create_loopwam_robot(vae_path,*,init_artifact=None,checkpoint_path=None,device='cpu',checkpoint_blocks=True,seed=42,artifact_sha256=None):
    if checkpoint_path:
        p=torch.load(checkpoint_path,map_location='cpu',mmap=True,weights_only=False);validate_robot_payload(p)
        from .wan_video_dit import WanVideoDiT
        from .action_dit import ActionDiT
        m=copy.deepcopy(p['architecture']);video=WanVideoDiT(**m['target_video_config']);action=ActionDiT(**m['target_action_config'])
    else:
        if init_artifact is None:raise ValueError('Canonical Wan donor artifact required; trained parents are not accepted')
        artifact=load_init_artifact(init_artifact)
        video,action,_=build_target_experts(artifact,device=device,dtype=torch.float32)
        m=architecture_metadata_for_version(artifact['metadata'],'v0')
        m.update(initialization_mode='fresh_wan_donors_no_embodied_pretraining',init_artifact_sha256=artifact_sha256 or sha256_file(init_artifact),robot_interface_seed=seed,robot_proprio_dim=14)
        p=None
    video=video.to(device=device,dtype=torch.float32);action=action.to(device=device,dtype=torch.float32)
    vae=load_wan21_vae(vae_path,device=device,dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32)
    mot=LoopMoT({'video':video,'action':action},loops=4,version='v0',checkpoint_blocks=checkpoint_blocks)
    model=RobotLoopWAM(video_expert=video,action_expert=action,mot=mot,vae=vae,text_dim=4096,
        proprio_dim=14 if p else 8,device=device,torch_dtype=torch.float32,version='v0',exit_weight_scale=1.,architecture_metadata=m,
        video_train_shift=5.,video_infer_shift=5.,action_train_shift=1.,action_infer_shift=1.)
    if p:
        model.mot.load_state_dict(p['mot'],strict=True);model.proprio_encoder.load_state_dict(p['proprio_encoder'],strict=True)
    else:
        fresh_robot_interfaces(model,seed);m['target_action_config']={**m['target_action_config'],'action_dim':14}
    return model,p

class RobotTrainForward(nn.Module):
    """DDP entry point: preserve original v0 losses, convert per-window mean to sum."""
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,segment,state=None,sequential=False):
        if sequential or state is not None or len(segment)!=1:raise ValueError('Original v0 is stateless')
        sample=segment[0]['sample'];n=segment[0]['noise']
        noisy=self.model.prepare_training_batch(sample,noise_video=n['video'],noise_action=n['action'],timestep_video=n['t_video'],timestep_action=n['t_action'])
        loss,logs=self.model.reduce_flow_losses(self.model.forward_exits(noisy),noisy)
        b=sample['action'].shape[0]
        return loss*b,{name:logs[key]*b for name,key in [('video','loss_video'),('action','loss_action'),('video_raw','exit4/video_raw'),('action_raw','exit4/action_raw')]},None

def verify_optimizer_restore(optimizer,expected):
    actual=optimizer.state_dict()
    if actual['param_groups']!=expected['param_groups'] or actual['state'].keys()!=expected['state'].keys():
        raise ValueError('Optimizer groups or parameter IDs changed on restore')
    for pid,state in expected['state'].items():
        if actual['state'][pid].keys()!=state.keys():raise ValueError('Optimizer state keys changed')
        for key,value in state.items():
            restored=actual['state'][pid][key]
            equal=(restored.dtype==value.dtype and torch.equal(restored.detach().cpu(),value.detach().cpu())) if torch.is_tensor(value) else restored==value
            if not equal:raise ValueError(f'Optimizer restore mismatch: {pid}/{key}')
