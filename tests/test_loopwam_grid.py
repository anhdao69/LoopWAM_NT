import copy
import pytest
import torch
from test_loop_mot import make_model,inputs,cached_action,assert_pair_close
from fastwam.models.wan22.loop_mot import LoopMoT


def model(v,a):
    base=make_model()
    return LoopMoT(dict(base.mixtures.items()),loops=v,action_loops=a)


@pytest.mark.parametrize('v,a',[(4,2),(2,2),(1,4)])
@pytest.mark.parametrize('structured',[False,True])
def test_grid_joint_cache_and_gradients(v,a,structured):
    joint=model(v,a);other=copy.deepcopy(joint);data=inputs()
    joint.structured_attention=structured;joint.structured_attention_observation_tokens=2
    # cached_action's helper asserts a second core exists, so build the cache here.
    keys,values=other.prefill_video_cache_tensor(data['video_tokens'][:,:2],data['video_freqs'][:2],
        data['video_t_mod'][:,:2],data['video_context'],data['video_context_mask'][:,:2],data['attention_mask'][:2,:2])
    mask=torch.cat((data['attention_mask'][4:,:2],data['attention_mask'][4:,4:]),dim=1)
    b=other.forward_action_with_video_cache_tensor(data['action_tokens'],data['action_freqs'],
        data['action_t_mod'],data['action_context'],data['action_context_mask'],keys,values,mask)
    y=joint.forward_joint_core(**data)[1]
    torch.testing.assert_close(y,b,atol=3e-5,rtol=3e-5)
    y.square().mean().backward();b.square().mean().backward()
    for (name,p),(_,q) in zip(joint.named_parameters(),other.named_parameters()):
        torch.testing.assert_close(p.grad if p.grad is not None else torch.zeros_like(p),
            q.grad if q.grad is not None else torch.zeros_like(q),atol=5e-5,rtol=5e-4,msg=name)


def test_one_video_four_action_reuses_one_core_cache():
    m=model(1,4)
    assert m.action_cache_slots()==tuple(list(range(3))+list(range(3,9))*4+list(range(9,12)))


@pytest.mark.parametrize('v,a',[(4,2),(2,2),(1,4)])
def test_grid_no_future_leak_and_checkpointed_gradients(v,a):
    m=model(v,a);data=inputs();baseline=m.forward_joint_core(**data)
    changed=dict(data,video_tokens=data['video_tokens'].clone());changed['video_tokens'][:,2:]+=40
    torch.testing.assert_close(m.forward_joint_core(**changed)[1],baseline[1],atol=1e-6,rtol=1e-6)
    checked=copy.deepcopy(m);checked.checkpoint_blocks=True
    y=checked.forward_joint_core(**data);assert_pair_close(y,baseline)
    sum(x.square().mean() for x in baseline).backward();sum(x.square().mean() for x in y).backward()
    for (name,p),(_,q) in zip(m.named_parameters(),checked.named_parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=5e-5,rtol=5e-4,msg=name)


@pytest.mark.parametrize('v,a',[(4,2),(2,2),(1,4)])
def test_grid_checkpoint_evaluation_spec(v,a):
    from scripts.evaluate_loopwam_libero import checkpoint_policy_spec
    p=dict(format_version='loopwam-s-v1',version='v0',trained_max_loops=4,inference_loops=v,
           video_loops=v,action_loops=a,loop_alignment='late')
    assert checkpoint_policy_spec(p)==('v0',v)


@pytest.mark.parametrize('v,a',[(4,2),(2,2),(1,4)])
def test_grid_factory_preserves_pair(v,a,tmp_path,monkeypatch):
    from test_loopwam_dense_s30 import tiny_configs
    from fastwam.models.wan22 import loopwam_init
    from fastwam.models.wan22.loopwam import LoopWAM,create_loopwam
    m=model(v,a);m.mixtures['video'].seperated_timestep=True
    m.mixtures['video'].video_attention_mask_mode='first_frame_causal'
    vae=torch.nn.Linear(1,1);vae.temporal_downsample_factor=4
    vc,ac=tiny_configs(12)
    policy=LoopWAM(video_expert=m.mixtures['video'],action_expert=m.mixtures['action'],mot=m,
        vae=vae,text_dim=4096,proprio_dim=8,device='cpu',torch_dtype=torch.float32,
        architecture_metadata=dict(target_video_config=vc,target_action_config=ac))
    path=tmp_path/'model.pt';policy.save_checkpoint(path)
    monkeypatch.setattr(loopwam_init,'target_configs',tiny_configs)
    monkeypatch.setattr(loopwam_init,'load_wan21_vae',lambda *args,**kw:copy.deepcopy(vae))
    loaded=create_loopwam(checkpoint_path=path,vae_path='test')
    assert (loaded.mot.loops,loaded.mot.action_loops)==(v,a)
    assert_pair_close(m.forward_joint_core(**inputs()),loaded.mot.forward_joint_core(**inputs()))
