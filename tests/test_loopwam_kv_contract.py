import copy
import pytest
import torch
from test_loopwam_policy import make_policy
from fastwam.models.wan22.loop_mot import LoopMoT
from fastwam.models.wan22.loopwam import _validate_checkpoint_depth


def policy(mode):
    p=make_policy()
    p.mot=LoopMoT(dict(p.mot.mixtures.items()),loops=4,action_loops=1,action_kv_mode=mode)
    return p


@pytest.mark.parametrize('mode',['aligned','concat','mix'])
def test_mode_checkpoint_roundtrip_and_mismatch(tmp_path,mode):
    p=policy(mode)
    if mode=='mix': p.mot.action_kv_logits.data.normal_()
    path=tmp_path/'p.pt';p.save_checkpoint(path)
    payload=torch.load(path,weights_only=False)
    assert payload['action_kv_mode']==mode
    q=policy(mode);q.load_checkpoint(path)
    for name,value in p.mot.state_dict().items(): torch.testing.assert_close(value,q.mot.state_dict()[name])
    for other in {'aligned','concat','mix'}-{mode}:
        with pytest.raises(ValueError,match='mode'):policy(other).load_checkpoint(path)
    if mode=='aligned':
        del payload['action_kv_mode'];torch.save(payload,path);q.load_checkpoint(path)
    else:
        del payload['action_kv_mode'];torch.save(payload,path)
        with pytest.raises((ValueError,RuntimeError)):q.load_checkpoint(path)


def test_mode_validator_and_logit_shape(tmp_path):
    p=policy('mix');path=tmp_path/'p.pt';p.save_checkpoint(path)
    payload=torch.load(path,weights_only=False)
    for mode in ('wrong',None,True):
        with pytest.raises(ValueError):_validate_checkpoint_depth(dict(payload,action_kv_mode=mode))
    for version in ('v1','v2','dense_s12','dense_s30'):
        with pytest.raises(ValueError):_validate_checkpoint_depth(dict(payload,version=version))
    bad=copy.deepcopy(payload);bad['mot']['action_kv_logits']=torch.zeros(6,2,3)
    with pytest.raises(ValueError,match='logit'): _validate_checkpoint_depth(bad)


def test_mix_parameter_group_and_verified_real_count():
    from fastwam.training_backends import policy_optimizer_parameters, expected_policy_parameters
    for mode in ('aligned','concat','mix'):
        p=policy(mode);groups=policy_optimizer_parameters(p)
        opt=torch.optim.AdamW(groups,lr=1e-4,weight_decay=.01)
        assert len(opt.param_groups)==(2 if mode=='mix' else 1)
        flat=[v for g in opt.param_groups for v in g['params']]
        assert {id(v) for v in flat}=={id(v) for v in p.policy_parameters()}
        assert len(flat)==len({id(v) for v in flat})
        if mode=='mix':
            assert opt.param_groups[-1]['params']==[p.mot.action_kv_logits]
            assert opt.param_groups[-1]['weight_decay']==0
            assert opt.param_groups[0]['weight_decay']==.01
    assert expected_policy_parameters('v0','concat',4)==584536135
    assert expected_policy_parameters('v0','mix',4)==584536423
    assert expected_policy_parameters('dense_s30','aligned',1)==1416114247


def test_eval_rejects_mode_contract(tmp_path):
    from evaluate_loopwam_libero import validate_checkpoint
    p=policy('concat');path=tmp_path/'p.pt';p.save_checkpoint(path)
    payload=torch.load(path,weights_only=False)
    data=dict(train_windows=92678,normalization_source='training episodes only',normalization_sha256='test')
    contract=dict(version='v0',planned_updates=7250,global_batch=128,world=2,microbatch=8,
                  seed=42,train_windows=92678,normalization_sha256='test',video_loops=4,
                  action_loops=1,loop_alignment='late',action_kv_mode='concat')
    payload.update(step=10,training_state=dict(update=10,contract=contract))
    assert validate_checkpoint(payload,data,'test',smoke=True)==contract
    for bad in ('aligned','mix',None):
        contract['action_kv_mode']=bad
        with pytest.raises(ValueError,match='mode'):validate_checkpoint(payload,data,'test',smoke=True)


@pytest.mark.parametrize('mode',['concat','mix'])
def test_factory_infers_mode_and_rejects_overrides(tmp_path,monkeypatch,mode):
    from test_loopwam_dense_s30 import tiny_configs
    from fastwam.models.wan22 import loopwam_init
    from fastwam.models.wan22.loopwam import create_loopwam
    p=policy(mode);vc,ac=tiny_configs(12)
    p.architecture_metadata=dict(target_video_config=vc,target_action_config=ac)
    # The native factory uses proprio_dim=8/text_dim=4096; replace this tiny head accordingly.
    p.proprio_encoder=torch.nn.Linear(8,4096) if isinstance(p.proprio_encoder,torch.nn.Linear) else p.proprio_encoder
    path=tmp_path/'p.pt';p.save_checkpoint(path)
    monkeypatch.setattr(loopwam_init,'target_configs',tiny_configs)
    monkeypatch.setattr(loopwam_init,'load_wan21_vae',lambda *a,**kw:copy.deepcopy(p.vae))
    restored=create_loopwam(checkpoint_path=path,vae_path='test')
    assert restored.mot.action_kv_mode==mode and restored.mot.action_loops==1
    for name,value in p.mot.state_dict().items():torch.testing.assert_close(value,restored.mot.state_dict()[name])
    with pytest.raises(ValueError,match='mode'):
        create_loopwam(checkpoint_path=path,vae_path='test',action_kv_mode='aligned')
    with pytest.raises(ValueError,match='depth'):
        create_loopwam(checkpoint_path=path,vae_path='test',loops=3)
