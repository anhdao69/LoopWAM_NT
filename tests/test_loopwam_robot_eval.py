import contextlib
from collections import deque
from types import SimpleNamespace
import numpy as np
import torch
import pytest

def test_original_policy_input_action_and_fixed_noise(monkeypatch):
    from experiments.robotwin.loopwam_policy import deploy_policy as d
    calls=[]
    class Norm:
        def forward(self,x):return x+1
        def backward(self,x):return x*2
    def infer(**kwargs):calls.append(kwargs);return {'action':torch.ones(32,14)}
    obj=object.__new__(d.LoopWAMRoboTwinPolicy)
    obj.model=SimpleNamespace(infer_action=infer);obj.normalizer={'state':Norm(),'action':Norm()}
    obj.text=lambda p:(torch.zeros(128,4096),torch.ones(128,dtype=torch.bool))
    obj.seed=42;obj.compile_action_infer=False;obj.pending=deque();obj.queries=0;obj.episode=0;obj.latency=[]
    monkeypatch.setattr(torch,'autocast',lambda *a,**k:contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    observation={'observation':{c:{'rgb':np.zeros((240,320,3),dtype=np.uint8)} for c in d.CAMERAS},'joint_action':{'vector':np.zeros(14)}}
    out=obj._query(observation,'lift pot')
    assert out.shape==(32,14) and (out==2).all()
    kw=calls[0];assert kw['input_image'].shape==(3,384,320) and torch.all(kw['proprio']==1)
    assert kw['seed']==42 and kw['num_inference_steps']==10 and kw['rand_device']=='cpu'
    assert not {'state','history_image'} & kw.keys()
    obj.reset();obj._query(observation,'lift pot');assert calls[-1]['seed']==42

def test_policy_replans_after_exactly24_actions():
    from experiments.robotwin.loopwam_policy.deploy_policy import LoopWAMRoboTwinPolicy
    obj=object.__new__(LoopWAMRoboTwinPolicy);obj.pending=deque();queries=[];actions=[]
    def query(*args):queries.append(args);return np.ones((32,14))
    obj._query=query
    env=SimpleNamespace(get_instruction=lambda:'task',take_action=lambda a,action_type:actions.append((a,action_type)))
    for i in range(24):obj.step(env,{} if i==0 else None)
    assert len(queries)==1 and len(actions)==24 and obj.should_request_observation()
    obj.step(env,{});assert len(queries)==2 and all(x[1]=='qpos' for x in actions)

def test_protocol_rejects_wrong_replan_depth():
    from experiments.robotwin.loopwam_policy.deploy_policy import get_model
    with pytest.raises(ValueError):get_model({'replan_steps':10})
