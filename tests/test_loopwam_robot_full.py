import torch
import pytest
from torch import nn

def test_cached_latents_never_encode_dummy_video():
    from fastwam.models.wan22.loopwam_robot import RobotLoopWAM
    model=object.__new__(RobotLoopWAM)
    lat=torch.randn(2,16,3,48,40)
    assert model._encode_training_video({'latents':lat},None) is lat
    with pytest.raises(ValueError):model._encode_training_video({'latents':torch.zeros(2,16,1,48,40)},None)
    with pytest.raises(ValueError):model._encode_training_video({},None)

def test_fresh_interfaces_seed_and_dimensions():
    from fastwam.models.wan22.loopwam_robot import fresh_robot_interfaces
    def model():
        m=nn.Module();m.action_expert=nn.Module();m.action_expert.hidden_dim=8
        m.action_expert.action_encoder=nn.Linear(7,8);m.action_expert.head=nn.Linear(8,7)
        m.proprio_encoder=nn.Linear(8,16);return m
    a,b=model(),model();fresh_robot_interfaces(a,42);fresh_robot_interfaces(b,42)
    assert a.action_expert.action_encoder.in_features==14 and a.action_expert.head.out_features==14
    assert a.proprio_encoder.in_features==14
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())

def test_v0_ddp_wrapper_returns_sample_sum():
    from fastwam.models.wan22.loopwam_robot import RobotTrainForward
    class Fake(nn.Module):
        def prepare_training_batch(self,sample,**kw):return sample
        def forward_exits(self,sample):return sample
        def reduce_flow_losses(self,outputs,noisy):
            loss=outputs['action'].square().mean();return loss,{'loss_video':loss.detach(),'loss_action':loss.detach(),'exit4/video_raw':loss.detach(),'exit4/action_raw':loss.detach()}
    action=torch.ones(3,32,14,requires_grad=True)
    noise={k:None for k in ('video','action','t_video','t_action')}
    loss,logs,_=RobotTrainForward(Fake())([{'sample':{'action':action},'noise':noise}],None,False)
    assert loss.item()==3 and logs['video'].item()==3
    loss.backward();assert torch.allclose(action.grad,torch.full_like(action,2/(32*14)))

@pytest.mark.parametrize('field,bad',[('inference_loops',3),('trained_max_loops',3),('exit_weight_scale',.5)])
def test_robot_loader_rejects_changed_loss_or_depth(field,bad):
    from fastwam.models.wan22.loopwam_robot import validate_robot_payload,FORMAT
    from fastwam.models.wan22.loopwam_init import target_configs,DONOR_INDICES
    v,a=target_configs(12);a['action_dim']=14
    p=dict(format_version=FORMAT,version='v0',video_loops=4,action_loops=4,memory_tokens=0,history_frame=0,inference_loops=4,trained_max_loops=4,exit_weight_scale=1.,architecture=dict(target_video_config=v,target_action_config=a,robot_proprio_dim=14,donor_indices=list(DONOR_INDICES),initialization_mode='fresh_wan_donors_no_embodied_pretraining'))
    validate_robot_payload(p);p[field]=bad
    with pytest.raises(ValueError):validate_robot_payload(p)

def test_exact_epoch_batches_cover_once_and_resume():
    from fastwam.datasets.loopwam_robot_batches import epoch_rank_batches,training_budget
    for epoch in range(2):
        ranks=[list(epoch_rank_batches(351,128,8,r,42+epoch)) for r in range(8)]
        global_rows=[sum([ranks[r][i] for r in range(8)],[]) for i in range(3)]
        assert [len(x) for x in global_rows]==[128,128,95]
        assert sorted(sum(global_rows,[]))==list(range(351))
        expected=torch.randperm(351,generator=torch.Generator().manual_seed(42+epoch)).tolist()
        assert sum(global_rows,[])==expected
        assert list(epoch_rank_batches(351,128,8,0,42+epoch,start_batch=1))==ranks[0][1:]
    assert training_budget(6075103,128,5)==(47462,237310,95)
