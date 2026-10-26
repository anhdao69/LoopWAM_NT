"""Partition ablations must change sharing, never the stored weights or masks."""
import copy
import pytest
import torch
from test_loop_mot import make_model, inputs, cached_action, assert_pair_close
from fastwam.models.wan22.loop_mot import LoopMoT
from fastwam.models.wan22.mot import MoT

GRID = [((0,6,6),4), ((1,6,5),4), ((3,6,3),4), ((5,6,1),4),
        ((6,6,0),4), ((1,9,2),3), ((4,3,5),7)]

def model_for(partition, loops):
    base = make_model()
    return LoopMoT(dict(base.mixtures.items()), loops=loops, partition=partition)

@pytest.mark.parametrize('partition,loops', GRID)
@pytest.mark.parametrize('structured', [False, True])
def test_partition_unroll_and_all_block_gradients(partition, loops, structured):
    model = model_for(partition, loops)
    model.structured_attention = structured
    model.structured_attention_observation_tokens = 2
    pre, core, post = partition
    indices = list(range(pre)) + list(range(pre, pre+core))*loops + list(range(pre+core,12))
    assert len(indices) == model.effective_depth() == 30
    assert [i for _, i in model.virtual_schedule()] == indices
    assert model.action_cache_slots() == tuple(range(30))
    dense = MoT(copy.deepcopy(dict(model.mixtures.items())))
    for name in ('video','action'):
        dense.mixtures[name].blocks = torch.nn.ModuleList([copy.deepcopy(model.mixtures[name].blocks[i]) for i in indices])
    dense.num_layers = 30
    actual = model.forward_joint_core(**inputs())
    reference = dense.forward_joint_core(**inputs())
    assert_pair_close(actual, reference)
    sum(x.square().mean() for x in actual).backward()
    sum(x.square().mean() for x in reference).backward()
    for expert in ('video','action'):
        for i in range(12):
            for name, p in model.mixtures[expert].blocks[i].named_parameters():
                qs = [dict(dense.mixtures[expert].blocks[j].named_parameters())[name].grad for j,n in enumerate(indices) if n==i]
                if p.grad is None:
                    assert all(q is None for q in qs)
                else:
                    torch.testing.assert_close(p.grad,sum(qs),atol=4e-5,rtol=5e-4)

@pytest.mark.parametrize('partition,loops', GRID)
def test_partition_cache_checkpointing_and_no_future_leak(partition, loops):
    m=model_for(partition,loops); checked=copy.deepcopy(m); checked.checkpoint_blocks=True
    data=inputs(); a=m.forward_joint_core(**data); b=checked.forward_joint_core(**data)
    assert_pair_close(a,b)
    torch.testing.assert_close(a[1],cached_action(m,data),atol=3e-5,rtol=3e-5)
    changed=dict(data,video_tokens=data['video_tokens'].clone());changed['video_tokens'][:,2:]+=50
    torch.testing.assert_close(a[1],m.forward_joint_core(**changed)[1],atol=1e-6,rtol=1e-6)
    sum(x.square().mean() for x in a).backward();sum(x.square().mean() for x in b).backward()
    for (name,p),(_,q) in zip(m.named_parameters(),checked.named_parameters()):
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,msg=name)

def test_explicit_default_is_bit_identical():
    a=make_model();b=model_for((3,6,3),4)
    for x,y in zip(a.forward_joint_core(**inputs()),b.forward_joint_core(**inputs())):
        assert torch.equal(x,y)
    assert a.state_dict().keys()==b.state_dict().keys()

@pytest.mark.parametrize('partition', [(0,0,12),(-1,6,7),(1,6,6),(True,6,5),(1,6),(1.,6,5)])
def test_invalid_partition(partition):
    with pytest.raises(ValueError): model_for(partition,4)

@pytest.mark.parametrize('partition,loops', GRID)
def test_checkpoint_roundtrip_and_partition_guard(tmp_path, partition, loops):
    from test_loopwam_policy import make_policy
    from fastwam.models.wan22.loopwam import _validate_checkpoint_depth
    from evaluate_loopwam_libero import checkpoint_policy_spec
    policy=make_policy();policy.mot=model_for(partition,loops)
    path=tmp_path/'student.pt';policy.save_checkpoint(path)
    saved=torch.load(path,weights_only=False)
    assert saved['block_partition']==list(partition)
    _validate_checkpoint_depth(saved)
    assert checkpoint_policy_spec(saved)==('v0',loops)
    restored=make_policy();restored.mot=model_for(partition,loops);restored.load_checkpoint(path)
    assert_pair_close(policy.mot.forward_joint_core(**inputs()),restored.mot.forward_joint_core(**inputs()))
    wrong=make_policy();wrong.mot=model_for((0,6,6) if partition!=(0,6,6) else (3,6,3),4)
    with pytest.raises(ValueError,match='partition'):wrong.load_checkpoint(path)
    bad=copy.deepcopy(saved);bad['block_partition']=[1,0,11]
    with pytest.raises(ValueError):_validate_checkpoint_depth(bad)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('partition,loops', GRID)
def test_cuda_partition_cache(partition, loops):
    m=model_for(partition,loops).cuda(); data={k:v.cuda() for k,v in inputs().items()}
    m.structured_attention=True;m.structured_attention_observation_tokens=2
    with torch.no_grad():
        torch.testing.assert_close(m.forward_joint_core(**data)[1],cached_action(m,data),atol=3e-5,rtol=3e-5)

@pytest.mark.parametrize('partition,loops', GRID)
def test_factory_restore_and_loss_contract(tmp_path, monkeypatch, partition, loops):
    from test_loopwam_dense_s30 import tiny_configs
    from test_loopwam_policy import make_policy, tiny_noisy
    from fastwam.models.wan22 import loopwam_init
    from fastwam.models.wan22.loopwam import create_loopwam
    policy=make_policy();policy.mot=model_for(partition,loops)
    # Policy loss must accept the seven-loop final exit.
    outputs=policy.forward_exits(tiny_noisy(policy))
    loss,_=policy.reduce_flow_losses(outputs,tiny_noisy(policy))
    assert set(outputs)=={loops} and torch.isfinite(loss)
    # Factory uses the native metadata shape (tiny configs stand in for native sizes).
    vc,ac=tiny_configs(12)
    policy.architecture_metadata=dict(target_video_config=vc,target_action_config=ac)
    policy.proprio_encoder=torch.nn.Linear(8,4096)
    path=tmp_path/'factory.pt';policy.save_checkpoint(path)
    monkeypatch.setattr(loopwam_init,'target_configs',tiny_configs)
    monkeypatch.setattr(loopwam_init,'load_wan21_vae',lambda *a,**kw:copy.deepcopy(policy.vae))
    restored=create_loopwam(checkpoint_path=path,vae_path='test')
    assert restored.mot.partition==partition and restored.mot.loops==loops
    assert_pair_close(policy.mot.forward_joint_core(**inputs()),restored.mot.forward_joint_core(**inputs()))
    with pytest.raises(ValueError,match='partition'):
        create_loopwam(checkpoint_path=path,vae_path='test',partition=(0,6,6) if partition!=(0,6,6) else (3,6,3))

@pytest.mark.parametrize('partition,loops', GRID)
def test_evaluator_partition_contract(partition,loops):
    from test_loopwam_full_eval import full_contract
    from evaluate_loopwam_libero import validate_checkpoint
    from fastwam.models.wan22.loop_mot import loop_capacity
    p,d=full_contract();p.update(block_partition=list(partition),video_loops=loops,
        action_loops=loops,inference_loops=loops,trained_max_loops=loop_capacity('v0',partition),loop_alignment='late')
    p['training_state']['contract'].update(block_partition=list(partition),video_loops=loops,action_loops=loops,loop_alignment='late')
    validate_checkpoint(p,d,'stats')
    p['training_state']['contract']['block_partition']=[0,6,6] if partition!=(0,6,6) else [3,6,3]
    with pytest.raises(ValueError,match='partition'):validate_checkpoint(p,d,'stats')

def test_factory_rejects_partition_corrupted_against_training_contract(tmp_path):
    from test_loopwam_policy import make_policy
    from fastwam.models.wan22.loopwam import _validate_checkpoint_depth
    policy=make_policy();path=tmp_path/'student.pt'
    policy.save_checkpoint(path,training_state={'contract':{'block_partition':[3,6,3]}})
    payload=torch.load(path,weights_only=False);payload['block_partition']=[0,6,6]
    with pytest.raises(ValueError,match='partition'):_validate_checkpoint_depth(payload)

def test_nondefault_checkpoint_format_prevents_silent_legacy_load(tmp_path):
    from test_loopwam_policy import make_policy
    policy=make_policy();policy.mot=model_for((0,6,6),4)
    path=tmp_path/'partition.pt';policy.save_checkpoint(path)
    p=torch.load(path,weights_only=False)
    assert p['format_version']=='loopwam-s-partition-v1'
