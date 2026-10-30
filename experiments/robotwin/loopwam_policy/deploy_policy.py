"""Stateless original LoopWAM v0 policy for the full RoboTwin benchmark."""
import hashlib
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

CODE_ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(CODE_ROOT));sys.path.insert(0,str(CODE_ROOT/'src'))
os.environ.setdefault('DIFFSYNTH_MODEL_BASE_PATH',str(CODE_ROOT/'checkpoints'))
from fastwam.datasets.chronoloop_robot import preprocess_frames,robot_spec
from fastwam.datasets.loopwam_long import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
# Only the validated T5 transport/cache helper is shared; no ChronoLoop model is constructed.
from experiments.robotwin.chronoloop_policy.deploy_policy import TextEncoder,CAMERAS

class LoopWAMRoboTwinPolicy:
    def __init__(self,checkpoint,data_dir,vae_path,seed=42,t5_server=None,expected_step=237310,compile_action_infer=False):
        from fastwam.models.wan22.loopwam_robot import create_loopwam_robot
        from fastwam.models.wan22.loop_mot import LoopMoT
        torch.set_num_threads(2)
        self.model,payload=create_loopwam_robot(vae_path,checkpoint_path=checkpoint,device='cuda',checkpoint_blocks=False)
        if type(self.model.mot) is not LoopMoT or payload['step']!=int(expected_step):raise ValueError('Wrong model or checkpoint step')
        stats=json.loads((Path(data_dir)/'dataset_stats.json').read_text())
        if hashlib.sha256(json.dumps(stats,sort_keys=True).encode()).hexdigest()!=payload['training_state']['contract']['normalization_sha256']:
            raise ValueError('Evaluation normalization differs from training')
        self.normalizer={kind:SingleFieldLinearNormalizer({k.removeprefix('global_'):torch.tensor(v,dtype=torch.float32) for k,v in stats[kind]['default'].items() if k.startswith('global_')},mode='z-score') for kind in ('action','state')}
        self.model.eval();del payload
        spec=robot_spec('robotwin50');self.text=TextEncoder(spec.text_cache_dir,spec.text_suffix,'cuda',server=t5_server)
        self.seed=int(seed);self.compile_action_infer=bool(compile_action_infer)
        self.pending=deque();self.episode=-1;self.queries=0;self.latency=[]

    def reset(self):
        self.pending.clear();self.queries=0;self.episode+=1

    def should_request_observation(self):return not self.pending

    def _query(self,observation,instruction):
        obs=observation['observation']
        views=[torch.from_numpy(np.ascontiguousarray(obs[c]['rgb'])).permute(2,0,1).unsqueeze(0) for c in CAMERAS]
        image=preprocess_frames(views)[0]
        joint=torch.as_tensor(np.asarray(observation['joint_action']['vector'],dtype=np.float32))
        if joint.shape!=(14,):raise ValueError('Expected14D joint state')
        proprio=self.normalizer['state'].forward(joint[None])[0]
        context,mask=self.text(DEFAULT_PROMPT.format(task=instruction))
        if self.queries==0:print(json.dumps(dict(loopwam_episode=self.episode,instruction=instruction,noise_seed=self.seed)),flush=True)
        start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            result=self.model.infer_action(input_image=image,proprio=proprio,context=context,context_mask=mask,
                num_inference_steps=10,seed=self.seed,rand_device='cpu',compile_action_infer=self.compile_action_infer)
        torch.cuda.synchronize();self.latency.append(time.perf_counter()-start);self.queries+=1
        action=result['action'].detach().float().cpu()
        if action.shape!=(32,14) or not torch.isfinite(action).all():raise ValueError('Invalid action output')
        return self.normalizer['action'].backward(action[None])[0].numpy()

    def step(self,task_env,observation):
        if not self.pending:
            if observation is None:raise ValueError('Missing replanning observation')
            chunk=self._query(observation,task_env.get_instruction())
            self.pending.extend(np.asarray(a,dtype=np.float32) for a in chunk[:24])
        task_env.take_action(self.pending.popleft(),action_type='qpos')

def get_model(args):
    if int(args.get('replan_steps',24))!=24 or int(args.get('num_inference_steps',10))!=10:raise ValueError('Protocol requires24replan/10denoise')
    return LoopWAMRoboTwinPolicy(args['ckpt_setting'],args['data_dir'],
        args.get('vae_path') or str(CODE_ROOT/'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'),
        seed=args.get('seed',42),t5_server=args.get('t5_server'),expected_step=args.get('expected_step',237310),
        compile_action_infer=args.get('compile_action_infer',False))

def eval(TASK_ENV,model,observation):model.step(TASK_ENV,observation)
def reset_model(model):model.reset()
