import copy
import pytest
import torch
from fastwam.models.wan22 import loopwam_init as init

CASES=[((0,6,6),list(range(12,18))+list(range(24,30))),
       ((1,6,5),[0]+list(range(12,18))+list(range(25,30))),
       ((3,6,3),list(range(3))+list(range(12,18))+list(range(27,30))),
       ((5,6,1),list(range(5))+list(range(12,18))+[29]),
       ((6,6,0),list(range(6))+list(range(12,18))),
       ((1,9,2),[0]+list(range(10,19))+[28,29]),
       ((4,3,5),list(range(4))+list(range(13,16))+list(range(25,30)))]

@pytest.mark.parametrize('partition,expected',CASES)
def test_stage_donor_mapping_and_actual_selection(partition,expected):
    indices=init.stage_donor_indices(partition)
    assert list(indices)==expected and len(set(indices))==12
    state={f'blocks.{i}.weight':torch.tensor([i]) for i in range(30)}
    state['head.weight']=torch.tensor([999])
    picked=init.select_donor_layers(state,indices)
    assert [picked[f'blocks.{i}.weight'].item() for i in range(12)]==expected
    assert picked['head.weight'].item()==999

@pytest.mark.parametrize('partition,expected',CASES)
def test_build_experts_loads_selected_donors_and_same_heads(monkeypatch,partition,expected):
    from test_loopwam_dense_s30 import tiny_configs
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.action_dit import ActionDiT
    vc,ac=tiny_configs(30)
    v,a=WanVideoDiT(**vc),ActionDiT(**ac)
    with torch.no_grad():
        for model in (v,a):
            for i,b in enumerate(model.blocks):
                for p in b.parameters():p.fill_(i+1)
    artifact=dict(video_state_dict=v.state_dict(),action_state_dict=a.state_dict(),proprio_state_dict=torch.nn.Linear(8,4096).state_dict())
    monkeypatch.setattr(init,'validate_artifact',lambda a:None)
    monkeypatch.setattr(init,'target_configs',tiny_configs)
    video,action,proprio=init.build_target_experts(artifact,donor_indices=init.stage_donor_indices(partition))
    for expert in (video,action):
        for j,donor in enumerate(expected):
            for p in expert.blocks[j].parameters():assert torch.all(p==donor+1)
    torch.testing.assert_close(action.head.weight,a.head.weight,rtol=0,atol=0)
    torch.testing.assert_close(proprio.weight,artifact['proprio_state_dict']['weight'],rtol=0,atol=0)

@pytest.mark.parametrize('partition,expected',CASES)
def test_stage_checkpoint_metadata_guard(partition,expected):
    from test_loopwam_policy import make_policy
    from fastwam.models.wan22.loopwam import _validate_checkpoint_depth
    p=dict(version='v0',architecture=dict(donor_strategy='stage_centered',donor_indices=expected),block_partition=list(partition),trained_max_loops=7 if partition[1]==3 else 4)
    _validate_checkpoint_depth(p)
    bad=copy.deepcopy(p);bad['architecture']['donor_indices']=list(reversed(expected))
    with pytest.raises(ValueError,match='donor'):_validate_checkpoint_depth(bad)
    bad=copy.deepcopy(p);bad['training_state']=dict(contract=dict(block_partition=list(partition),donor_strategy='canonical'))
    with pytest.raises(ValueError,match='donor'):_validate_checkpoint_depth(bad)
