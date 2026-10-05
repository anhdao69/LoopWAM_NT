import torch
import pytest
from fastwam.models.wan22.loopwam import exit_weights, masked_action_loss, LoopWAM
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler


def test_exit_weights():
    assert exit_weights('v0', 4) == {4: 1.0}
    assert exit_weights('v1', 4) == {1: 1/6, 2: 1/6, 3: 1/6, 4: 0.5}
    assert exit_weights('v2', 1) == {1: 1.0}
    assert sum(exit_weights('v2', 4, 2).values()) == 2
    with pytest.raises(ValueError):
        exit_weights('unknown', 4)


def test_action_loss_padding_denominator():
    pred = torch.tensor([[[1., 1.], [3., 3.], [100., 100.]], [[9., 9.]] * 3])
    target = torch.zeros_like(pred)
    pad = torch.tensor([[False, False, True], [True, True, True]])
    torch.testing.assert_close(masked_action_loss(pred, target, pad), torch.tensor([5., 0.]))


def test_scheduler_target_sign():
    scheduler = WanContinuousFlowMatchScheduler(shift=1)
    clean, noise = torch.randn(2, 3, 7), torch.randn(2, 3, 7)
    flow = scheduler.training_target(clean, noise, torch.tensor(1000.))
    torch.testing.assert_close(scheduler.step(flow, torch.tensor(-1.), noise), clean)


def test_video_loss_masks_anchor_and_fully_padded_groups():
    policy = object.__new__(LoopWAM)
    torch.nn.Module.__init__(policy)
    policy.vae = type('VAE', (), {'temporal_downsample_factor': 4})()
    pred = torch.tensor([1., 50.]).reshape(1, 1, 2, 1, 1)
    pad = torch.tensor([[False, False, False, False, False, True, True, True, True]])
    loss = policy._compute_video_loss_per_sample(pred, torch.zeros_like(pred), pad, False)
    torch.testing.assert_close(loss, torch.ones(1))


def make_policy(version='v0'):
    from test_loop_mot import make_model
    mot = make_model(version)
    mot.mixtures['video'].seperated_timestep = True
    mot.mixtures['video'].video_attention_mask_mode = 'first_frame_causal'
    vae = torch.nn.Linear(1,1)
    vae.temporal_downsample_factor = 4
    return LoopWAM(video_expert=mot.mixtures['video'], action_expert=mot.mixtures['action'],
        mot=mot,vae=vae,text_dim=8,proprio_dim=2,device='cpu',torch_dtype=torch.float32,
        version=version, action_train_shift=1,action_infer_shift=1)


def tiny_noisy(policy):
    torch.manual_seed(22)
    return dict(latents_video=torch.randn(1,2,3,1,2),latents_action=torch.randn(1,3,3),
        timestep_video=torch.tensor([500.]),timestep_action=torch.tensor([250.]),
        context=torch.randn(1,4,8),context_mask=torch.tensor([[True,True,False,True]]),
        attention_mask=policy._build_mot_attention_mask(6,3,2,torch.device('cpu')),
        target_video=torch.randn(1,2,3,1,2),target_action=torch.randn(1,3,3),
        image_is_pad=torch.zeros(1,9,dtype=torch.bool),action_is_pad=torch.tensor([[False,False,True]]))


@pytest.mark.parametrize('version',['v0','v1','v2'])
def test_policy_save_load_all_depths_and_weighted_loss(tmp_path,version):
    policy=make_policy(version)
    noisy=tiny_noisy(policy)
    outputs=policy.forward_exits(noisy)
    loss,logs=policy.reduce_flow_losses(outputs,noisy)
    expected=sum(w*(logs[f'exit{k}/video']+logs[f'exit{k}/action']) for k,w in exit_weights(version,4).items())
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert torch.isfinite(loss)
    path=tmp_path/'model.pt'
    policy.save_checkpoint(path)
    loaded=make_policy(version)
    loaded.load_checkpoint(path)
    for k in range(1,5):
        a=policy.forward_exits(noisy,loops=k,exits=(k,))[k]
        b=loaded.forward_exits(noisy,loops=k,exits=(k,))[k]
        for x,y in zip(a,b): torch.testing.assert_close(x,y)
    assert all(not p.requires_grad for p in policy.vae.parameters())
    assert len(policy.policy_parameters())==len({id(p) for p in policy.policy_parameters()})


def test_fixed_noise_overfit():
    policy=make_policy('v0'); policy.mot.loops=1
    noisy=tiny_noisy(policy)
    opt=torch.optim.AdamW(policy.policy_parameters(),lr=.003)
    initial=None
    for step in range(30):
        opt.zero_grad()
        loss,_=policy.reduce_flow_losses(policy.forward_exits(noisy),noisy)
        if initial is None: initial=loss.item()
        loss.backward(); opt.step()
    assert loss.item()<initial*.3


def test_inference_contract_rejects_missing_state_and_wrong_horizon():
    policy=make_policy()
    with pytest.raises(ValueError,match='proprioception'):
        policy.infer_action()
    with pytest.raises(ValueError,match='32-action'):
        policy.infer_action(proprio=torch.zeros(2),action_horizon=8)
    import inspect
    assert inspect.signature(policy.infer_action).parameters['num_inference_steps'].default==10
