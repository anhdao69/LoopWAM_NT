import copy
import pytest
from run_kv_campaign import QUEUES,train_command,eval_command,verify_fairness,select_candidate


def test_two_four_run_queues_and_explicit_contract():
    assert [len(q) for q in QUEUES.values()]==[4,4]
    assert {q[0]['mode'] for q in QUEUES.values()}=={'concat','mix'}
    for queue in QUEUES.values():
        for spec in queue:
            c=train_command(spec,dict(backend='ddp',microbatch=8),'out','cache')
            assert '--resume' not in c and c[c.index('--global-batch')+1]=='128'
            for seed in (42,43,44):
                e=eval_command(spec,'train','eval',8,seed)
                assert e[e.index('--seeds')+1]==str(seed)
            assert c[c.index('--action-kv-mode')+1]==spec['mode']


def test_fairness_compares_all_data_fields_and_assets():
    base=dict(data=dict(normalization_path='old',train_episodes=[1,2],content_files={'a':'hash'}),
              asset_sha256={'initialization':'init','vae':'vae'},seed=42,epochs=10,global_batch=128,
              train_windows=92678,planned_updates=7250,planned_windows=926780,
              policy_dtype='float32',optimizer_state_dtype='float32',compute_dtype='bfloat16',
              initialization_mode='canonical_wan_artifact_fresh_optimizer')
    candidate=copy.deepcopy(base);candidate['data']['normalization_path']='new'
    assert verify_fairness(candidate,base)['passed']
    for key in ('seed','planned_updates','asset_sha256'):
        bad=copy.deepcopy(candidate);bad[key]=None
        with pytest.raises(ValueError):verify_fairness(bad,base)
    bad=copy.deepcopy(candidate);bad['data']['content_files']['a']='changed'
    with pytest.raises(ValueError):verify_fairness(bad,base)


def test_long_prefers_matching_ddp_layout_even_if_small_speed_difference():
    trials=[dict(backend='ddp',microbatch=8,steady_seconds=3.5),dict(backend='zero1',microbatch=16,steady_seconds=3.)]
    assert select_candidate(trials,True)['microbatch']==8
    assert select_candidate(trials,False)['backend']=='zero1'
