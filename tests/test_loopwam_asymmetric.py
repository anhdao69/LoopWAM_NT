import copy
import pytest
import torch
from test_loop_mot import make_model, inputs, cached_action, assert_pair_close
from fastwam.models.wan22.loop_mot import LoopMoT


def asymmetric(action_loops=1):
    base=make_model()
    return LoopMoT(dict(base.mixtures.items()), loops=4, action_loops=action_loops)


def test_depth_and_late_alignment():
    model=asymmetric()
    assert model.action_loops==1
    assert model.action_cache_slots()==(0,1,2,21,22,23,24,25,26,27,28,29)
    assert len(model.virtual_schedule())==30


@pytest.mark.parametrize('structured',[False,True])
def test_asymmetric_cache_matches_joint_and_gradients(structured):
    joint=asymmetric(); data=inputs(); cached=copy.deepcopy(joint)
    joint.structured_attention=structured
    joint.structured_attention_observation_tokens=2
    a=joint.forward_joint_core(**data)[1]; b=cached_action(cached,data)
    torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)
    a.square().mean().backward(); b.square().mean().backward()
    for (name,p),(_,q) in zip(joint.named_parameters(),cached.named_parameters()):
        torch.testing.assert_close(p.grad if p.grad is not None else torch.zeros_like(p),
            q.grad if q.grad is not None else torch.zeros_like(q),atol=4e-5,rtol=5e-4,msg=name)


def test_video_unchanged_and_future_cannot_leak():
    model=asymmetric(); data=inputs(); coupled=make_model()
    original=model.forward_joint_core(**data)
    torch.testing.assert_close(original[0],coupled.forward_joint_core(**data)[0],atol=2e-5,rtol=2e-5)
    changed=dict(data,video_tokens=data['video_tokens'].clone())
    changed['video_tokens'][:,2:]+=50
    torch.testing.assert_close(original[1],model.forward_joint_core(**changed)[1],atol=1e-6,rtol=1e-6)


def test_checkpointed_asymmetric_gradients():
    model=asymmetric(); checked=copy.deepcopy(model); checked.checkpoint_blocks=True
    a=model.forward_joint_core(**inputs()); b=checked.forward_joint_core(**inputs())
    assert_pair_close(a,b)
    sum(x.square().mean() for x in a).backward();sum(x.square().mean() for x in b).backward()
    for (name,p),(_,q) in zip(model.named_parameters(),checked.named_parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,msg=name)


@pytest.mark.parametrize('value',[0,5,True,1.5])
def test_invalid_depth_rejected(value):
    with pytest.raises(ValueError):asymmetric(value)


def test_explicit_equal_depth_is_coupled():
    assert_pair_close(asymmetric(4).forward_joint_core(**inputs()),make_model().forward_joint_core(**inputs()))


def test_checkpoint_roundtrip_preserves_depth_and_rejects_wrong_model(tmp_path):
    from test_loopwam_policy import make_policy, tiny_noisy
    policy=make_policy();policy.mot.action_loops=1
    noisy=tiny_noisy(policy)
    expected=policy.forward_exits(noisy)[4]
    path=tmp_path/'asymmetric.pt';policy.save_checkpoint(path)
    payload=torch.load(path,weights_only=False)
    assert (payload['video_loops'],payload['action_loops'],payload['loop_alignment'])==(4,1,'late')
    loaded=make_policy()
    with pytest.raises(ValueError,match='depth differs'):loaded.load_checkpoint(path)
    loaded.mot.action_loops=1;loaded.load_checkpoint(path)
    assert_pair_close(expected,loaded.forward_exits(noisy)[4])


def test_asymmetric_requires_final_exit():
    with pytest.raises(ValueError,match='final video exit'):
        asymmetric().forward_joint_exits(**inputs(),exits=(1,4))


def test_factory_uses_saved_asymmetric_depth(tmp_path,monkeypatch):
    from test_loopwam_dense_s30 import tiny_configs
    from fastwam.models.wan22 import loopwam_init
    from fastwam.models.wan22.loopwam import LoopWAM,create_loopwam
    model=asymmetric();model.mixtures['video'].seperated_timestep=True
    model.mixtures['video'].video_attention_mask_mode='first_frame_causal'
    vae=torch.nn.Linear(1,1);vae.temporal_downsample_factor=4
    vc,ac=tiny_configs(12)
    policy=LoopWAM(video_expert=model.mixtures['video'],action_expert=model.mixtures['action'],
        mot=model,vae=vae,text_dim=4096,proprio_dim=8,device='cpu',torch_dtype=torch.float32,
        architecture_metadata=dict(target_video_config=vc,target_action_config=ac))
    path=tmp_path/'saved.pt';policy.save_checkpoint(path)
    monkeypatch.setattr(loopwam_init,'target_configs',tiny_configs)
    monkeypatch.setattr(loopwam_init,'load_wan21_vae',lambda *a,**kw:copy.deepcopy(vae))
    restored=create_loopwam(checkpoint_path=path,vae_path='test')
    assert restored.mot.loops==4 and restored.mot.action_loops==1
    assert_pair_close(model.forward_joint_core(**inputs()),restored.mot.forward_joint_core(**inputs()))
    with pytest.raises(ValueError,match='override'):
        create_loopwam(checkpoint_path=path,vae_path='test',action_loops=4)
