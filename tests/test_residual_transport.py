"""RT mathematical and production-block contracts."""
import copy
import importlib.util
import pytest
import torch
from test_loop_mot import make_model, inputs

def rt():
    assert importlib.util.find_spec("fastwam.models.wan22.residual_transport") is not None, "RT is not implemented"
    from fastwam.models.wan22 import residual_transport
    return residual_transport

def test_transport_initialization_and_forecast():
    m = rt()
    p, r, old = [torch.randn(2, 3, 8) for _ in range(3)]
    t, nt = torch.ones(2)*.8, torch.ones(2)*.7
    h = m.TransportHistory(r, old, torch.ones(2)*.9, t, torch.zeros_like(p))
    assert torch.equal(m.ResidualTransport("T0",8)(p,h,nt),torch.zeros_like(p))
    assert torch.equal(m.ResidualTransport("T1",8)(p,h,nt),r)
    torch.testing.assert_close(m.ResidualTransport("T2",8)(p,h,nt),r+(r-old),rtol=1e-5,atol=1e-6)
    for kind in ("T3","T4","T5"):
        mod=m.ResidualTransport(kind,8)
        torch.testing.assert_close(mod(p,h,nt),.95*r)
        assert sum(x.numel() for x in m.ResidualTransport(kind,512).parameters()) < 1_000_000
    assert torch.equal(m.ResidualTransport("T6",8)(p,h,nt),r)

def test_history_detaches_every_tensor():
    m=rt()
    xs=[torch.randn(2,3,8,requires_grad=True) for _ in range(5)]
    h=m.TransportHistory(*xs)
    detached=m.detach_boundary(xs[0],h)
    assert not detached[0].requires_grad
    assert all(not x.requires_grad for x in vars(detached[1]).values() if x is not None)

def cached_inputs(model,data):
    keys,values=model.prefill_video_cache_tensor(data["video_tokens"][:,:2],data["video_freqs"][:2],
        data["video_t_mod"][:,:2],data["video_context"],data["video_context_mask"][:,:2],
        data["attention_mask"][:2,:2])
    mask=torch.cat((data["attention_mask"][4:,:2],data["attention_mask"][4:,4:]),1)
    return (data["action_tokens"],data["action_freqs"],data["action_t_mod"],
            data["action_context"],data["action_context_mask"],keys,values,mask)

@pytest.mark.parametrize("loops",[1,2,3,4])
def test_action_parity_counts_and_slots(loops):
    m=rt(); model=make_model(); data=inputs(); args=cached_inputs(model,data)
    model.action_loops=loops
    expected=model.forward_action_with_video_cache_tensor(*args)
    trace=[]
    actual,h=m.action_core(model,*args,loops=loops,transport=m.ResidualTransport("T0",8),
        history=None,next_tau=torch.zeros(1),trace=trace)
    assert torch.equal(actual,expected)
    assert [s for _,s in trace]==list(model.action_cache_slots())
    assert len(trace)==6+6*loops
    assert h.shape==args[0].shape

def test_four_loops_always_cold_and_text_cache_equal():
    m=rt(); model=make_model(); data=inputs(); args=cached_inputs(model,data)
    history=m.TransportHistory(torch.randn_like(args[0]),None,None,torch.ones(1),None)
    expected=model.forward_action_with_video_cache_tensor(*args)
    kv=m.cache_text_kv(model.mixtures["action"],args[3])
    actual,_=m.action_core(model,*args,loops=4,transport=m.ResidualTransport("T1",8),
        history=history,next_tau=torch.zeros(1),text_kv=kv)
    assert torch.equal(actual,expected)

def test_lora_zero_and_merge():
    m=rt(); layer=torch.nn.Linear(8,12); x=torch.randn(2,3,8)
    mod=m.LoRALinear(copy.deepcopy(layer),rank=4)
    assert torch.equal(layer(x),mod(x))
    torch.nn.init.normal_(mod.B.weight,std=.1)
    torch.testing.assert_close(mod(x),mod.merged()(x),atol=1e-6,rtol=1e-5)

def test_teacher_targets_stop_gradient_and_grid():
    m=rt(); z=torch.randn(2,3,7,requires_grad=True)
    def velocity(x,t): return x*2+t[:,None,None]
    target=m.teacher_target(velocity,z,.8,.7)
    assert target.grad_fn is None and not target.requires_grad
    torch.testing.assert_close(target,velocity(z.detach(),torch.full((2,),.8)))
    end=z.detach()
    for i in range(5): end=end-.1*velocity(end,torch.full((2,),1.-i*.1))
    torch.testing.assert_close(m.teacher_target(velocity,z,1.,.5),(end-z.detach())/-.5)
    with pytest.raises(ValueError): m.teacher_target(velocity,z,1.,.75)

def test_step_condition_zero():
    m=rt(); mod=m.StepCondition(8,freq_dim=8)
    assert torch.count_nonzero(mod(torch.tensor([.1,.5])))==0

def test_schedules_counts():
    m=rt()
    expected={"S1":192,"S2":138,"S3":168,"S4":156,"S5":102,"S6":78,"S7":48,"S8":42}
    for name,count in expected.items(): assert sum(6+6*k for k in m.SCHEDULES[name])==count

def test_gripper_command_convention():
    m=rt()
    x=torch.zeros(1000,7); x[:500,-1]=-1; x[500:,-1]=1
    scale=torch.tensor([1.]*6+[2.]); offset=torch.tensor([0.]*6+[-1.])
    commands=m.gripper_command(x,scale,offset)
    assert torch.equal(commands[:500],torch.ones(500))
    assert torch.equal(commands[500:],-torch.ones(500))

def test_endpoint_losses_mask_and_gradient():
    m=rt(); student=torch.randn(2,32,7,requires_grad=True); teacher=torch.randn_like(student)
    end,grip=m.endpoint_losses(student,teacher,torch.ones(7),torch.zeros(7))
    assert end.shape==grip.shape==(2,)
    (end+grip).sum().backward()
    assert torch.count_nonzero(student.grad[:,10:])==0
    assert student.grad[:,:10].abs().sum()>0

def test_policy_velocity_uses_parent_timestep_units():
    from fastwam.models.wan22.transport_policy import TransportPolicy
    from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
    mot=make_model(); data=inputs()
    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mot=mot; self.action_expert=mot.mixtures["action"]; self.video_expert=mot.mixtures["video"]
            self.device="cpu"
            self.train_action_scheduler=WanContinuousFlowMatchScheduler(shift=1)
    policy=TransportPolicy(Parent(),kind="T0")
    z=torch.randn(1,32,3); raw_context=torch.randn(1,3,8); mask=torch.ones(1,3,dtype=torch.bool)
    condition=policy.conditioning(policy.parent.action_expert,raw_context,mask)
    args=cached_inputs(mot,data)
    cache=(args[5],args[6],torch.ones(32,34,dtype=torch.bool))
    prepared=policy.parent.action_expert.prepare(z,torch.tensor([800.]),raw_context,mask)
    expected=policy.parent.action_expert.post(mot.forward_action_with_video_cache_tensor(
        prepared[0],prepared[5],prepared[2],prepared[3],prepared[4],*cache))
    actual,_=policy.velocity(z,torch.tensor([.8]),condition,cache,teacher=True)
    assert torch.equal(actual,expected)

def test_all_transport_history_gradients_cut_at_boundary():
    m=rt()
    older=[torch.randn(2,3,8,requires_grad=True) for _ in range(3)]
    z=torch.randn(2,3,8,requires_grad=True)
    h=m.TransportHistory(older[0],older[1],torch.ones(2),torch.ones(2)*.9,older[2])
    z_cut,h_cut=m.detach_boundary(z,h)
    p=torch.randn(2,3,8,requires_grad=True)
    out=p+m.ResidualTransport("T2",8)(p,h_cut,torch.ones(2)*.8)+z_cut
    out.sum().backward()
    assert p.grad is not None
    assert z.grad is None and all(x.grad is None for x in older)

def test_core_residual_reconstructs_core_state():
    m=rt(); model=make_model(); data=inputs(); args=cached_inputs(model,data)
    seen=[]
    original=model._post
    def record(*a,**kw):
        result=original(*a,**kw); seen.append(result); return result
    model._post=record
    _,residual=m.action_core(model,*args,loops=2,transport=m.ResidualTransport("T0",8),
        history=None,next_tau=torch.zeros(1))
    torch.testing.assert_close(seen[2]+residual,seen[14],atol=1e-7,rtol=1e-7)

def test_paired_controls_share_identical_initial_lora():
    from fastwam.models.wan22.transport_policy import TransportPolicy
    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mot=make_model(); self.action_expert=self.mot.mixtures["action"]
            self.video_expert=self.mot.mixtures["video"]; self.device="cpu"
    parent=Parent()
    adapters=[]
    for kind in ("T0","T1","T3","T4","T5","T6"):
        torch.manual_seed(42)
        p=TransportPolicy(parent,kind=kind)
        adapters.append({n:v.detach().clone() for n,v in p.named_parameters() if ".A.weight" in n and n.startswith("student.")})
    assert adapters[0]
    for other in adapters[1:]:
        assert all(torch.equal(adapters[0][n],other[n]) for n in adapters[0])

def test_measured_action_backward_blocks():
    m=rt(); model=make_model(); data=inputs(); args=cached_inputs(model,data)
    model.rt_measurement={"forward":0,"backward":0}
    output,_=m.action_core(model,*args,loops=2,transport=m.ResidualTransport("T0",8),
        history=None,next_tau=torch.zeros(1),checkpoint_blocks=True)
    output.square().mean().backward()
    assert model.rt_measurement["backward"]==18
    assert model.rt_measurement["forward"]>=18
