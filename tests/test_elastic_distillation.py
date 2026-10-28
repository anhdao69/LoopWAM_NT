"""Loss/temporal semantics of the elastic objective (Task C), on the real forward() control flow."""
import pytest
import torch
from test_loop_mot import make_model, inputs
from fastwam.models.wan22.transport_policy import TransportPolicy
from fastwam.models.wan22 import residual_transport as rt
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

class Parent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mot = make_model(); self.action_expert = self.mot.mixtures["action"]
        self.video_expert = self.mot.mixtures["video"]; self.device = "cpu"
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(shift=1)

class ToyField(TransportPolicy):
    """Replaces only the network: teacher is an affine field, student a trainable scaled field."""
    def __init__(self, **kw):
        super().__init__(Parent(), kind="T0", scope="pc", step_cond=True, **kw)
        self.theta = torch.nn.Parameter(torch.tensor(-.6))
        self.calls = []
    def prefill(self, sample): return None, None, None, None, None
    @staticmethod
    def conditioning(expert, context, mask): return None, None, None
    def velocity(self, z, tau, conditioning, cache, *, teacher=False, loops=4, d=0., history=None, native_timestep=None, stats=None):
        if teacher:
            self.teacher_passes += 1
            return -.7 * z + .3 * tau[:, None, None], None
        self.calls.append(dict(z_grad=z.requires_grad, loops=loops, d=d, tau=tau.clone(),
                               history_grad=None if history is None else history.residual.requires_grad))
        v = self.theta * z + d + .1 * loops
        return v, self.theta * z

def sample(batch=2):
    torch.manual_seed(3)
    return dict(action=torch.randn(batch, 32, 7), action_is_pad=torch.zeros(batch, 32, dtype=torch.bool),
                sample_valid=torch.ones(batch, dtype=torch.bool), action_scale=torch.ones(7) * 2, action_offset=torch.ones(7) * -1)

def run(policy, schedule):
    torch.manual_seed(4)
    noise, anchor_noise, anchor_tau = torch.randn(2, 32, 7), torch.randn(2, 32, 7), torch.rand(2)
    loss, logs = policy(sample(), schedule, noise, anchor_noise, anchor_tau)
    loss.backward()
    return loss.detach(), logs, policy.theta.grad.clone(), noise

def test_teacher_macrosteps_on_the_deployed_grid():
    def velocity(x, t): return -.7 * x + .3 * t[:, None, None]
    z = torch.randn(2, 32, 7, requires_grad=True)
    for n in (2, 5, 10):
        for j in range(n):
            tau, next_tau = 1 - j / n, 1 - (j + 1) / n
            x = z.detach()
            for i in range(10 // n):  # 10/N reference microsteps of length 0.1
                t = round((1 - tau) * 10) + i
                x = x - .1 * velocity(x, torch.full((2,), 1 - t / 10))
            target = rt.teacher_target(velocity, z, tau, next_tau)
            assert target.grad_fn is None and not target.requires_grad
            torch.testing.assert_close(target, (x - z.detach()) / (next_tau - tau), rtol=1e-5, atol=1e-6)

@pytest.mark.parametrize("schedule", [(1, 1), (2,) * 5, (4,) * 10, (4, 1, 1, 1, 1), (4,) + (2,) * 9])
def test_teacher_prefix_reuse_is_bit_identical(schedule):
    reference = ToyField(); reused = ToyField(reuse_teacher_prefix=True)
    reused.load_state_dict(reference.state_dict())
    loss_a, logs_a, grad_a, _ = run(reference, schedule)
    loss_b, logs_b, grad_b, _ = run(reused, schedule)
    assert torch.equal(loss_a, loss_b) and torch.equal(grad_a, grad_b)
    assert all(torch.equal(logs_a[k], logs_b[k]) for k in logs_a)
    assert reference.teacher_passes == 20
    assert reused.teacher_passes == 20 - 10 // len(schedule)

@pytest.mark.parametrize("schedule", [(1, 1), (1,) * 5, (2,) * 10, (4, 1, 1, 1, 1)])
def test_student_calls_time_units_step_length_truncation_and_anchor(schedule):
    policy = ToyField(reuse_teacher_prefix=True)
    run(policy, schedule)
    n = len(schedule)
    steps, anchor = policy.calls[:n], policy.calls[n]
    assert len(policy.calls) == n + 1
    for j, (call, k) in enumerate(zip(steps, schedule)):
        assert call["loops"] == k and call["d"] == pytest.approx(1 / n) and call["d"] > 0
        assert torch.allclose(call["tau"], torch.full((2,), 1 - j / n))  # normalized tau; x1000 inside velocity
        # Final-three-step truncation: z and every history tensor detached when j <= N-3.
        assert call["z_grad"] == (j > n - 3 and j > 0)
        if j > 0: assert call["history_grad"] == (j > n - 3)
    assert anchor["loops"] == 4 and anchor["d"] == 0.

@pytest.mark.parametrize("schedule", [(1, 1), (2,) * 5, (4,) * 10])
def test_signed_euler_increments_and_first_ten_action_endpoint(schedule):
    policy = ToyField()
    with torch.no_grad(): policy.theta.zero_()
    _, logs, _, noise = run(policy, schedule)
    n, k = len(schedule), schedule[0]
    # v = d + .1k is constant; the increments (tau_{j+1}-tau_j) v sum to -v over tau 1 -> 0.
    student = noise - (1 / n + .1 * k)
    grid = torch.linspace(1., 0., 11); teacher = noise.clone()
    for j in range(10): teacher = teacher + (grid[j + 1] - grid[j]) * (-.7 * teacher + .3 * grid[j])
    torch.testing.assert_close(logs["end"], (student[:, :10] - teacher[:, :10]).square().mean((1, 2)).sum())

def test_native_timestep_units_and_step_condition_d():
    model = make_model(); data = inputs()
    parent = Parent(); parent.mot = model; parent.action_expert = model.mixtures["action"]; parent.video_expert = model.mixtures["video"]
    policy = TransportPolicy(parent, kind="T0", scope="pc", step_cond=True)
    seen = {}
    original = policy.step_condition.forward
    def spy(d): seen["d"] = d.clone(); return original(d)
    policy.step_condition.forward = spy
    expert = policy.student.mixtures["action"]
    stamps = {}
    import fastwam.models.wan22.transport_policy as tp
    real = tp.sinusoidal_embedding_1d
    def spy_embed(dim, t): stamps.setdefault("t", t.clone()); return real(dim, t)
    tp.sinusoidal_embedding_1d = spy_embed
    try:
        keys, values = model.prefill_video_cache_tensor(data["video_tokens"][:, :2], data["video_freqs"][:2],
            data["video_t_mod"][:, :2], data["video_context"], data["video_context_mask"][:, :2], data["attention_mask"][:2, :2])
        cache = (keys, values, torch.ones(3, 5, dtype=torch.bool))
        condition = policy.conditioning(expert, torch.randn(1, 3, 8), torch.ones(1, 3, dtype=torch.bool))
        condition = (condition[0], condition[1][:, :3], condition[2])
        policy.velocity(torch.randn(1, 3, 3), torch.tensor([.8]), condition, cache, loops=1, d=.2)
    finally:
        tp.sinusoidal_embedding_1d = real
    assert torch.equal(stamps["t"], torch.tensor([800.]))
    assert torch.equal(seen["d"], torch.tensor([.2]))

def test_residual_is_core_output_minus_fresh_prelude_even_when_warm_started():
    model = make_model(); data = inputs()
    keys, values = model.prefill_video_cache_tensor(data["video_tokens"][:, :2], data["video_freqs"][:2],
        data["video_t_mod"][:, :2], data["video_context"], data["video_context_mask"][:, :2], data["attention_mask"][:2, :2])
    mask = torch.cat((data["attention_mask"][4:, :2], data["attention_mask"][4:, 4:]), 1)
    args = (data["action_tokens"], data["action_freqs"], data["action_t_mod"], data["action_context"],
            data["action_context_mask"], keys, values, mask)
    seen = []
    original = model._post
    def record(*a, **kw):
        result = original(*a, **kw); seen.append(result); return result
    model._post = record
    previous = torch.randn_like(args[0])
    history = rt.TransportHistory(previous, None, None, torch.ones(1) * .9, None)
    out, residual = rt.action_core(model, *args, loops=2, transport=rt.ResidualTransport("T1", 8),
                                   history=history, next_tau=torch.ones(1) * .8)
    p, h = seen[2], seen[2 + 12]  # prelude output; state after the second core loop
    torch.testing.assert_close(residual, h - p, atol=0, rtol=0)  # not h - (p + previous)
    assert not torch.allclose(residual, h - (p + previous))

@pytest.mark.parametrize("loops,expected", [(1, [4]), (2, [3, 4]), (4, [1, 2, 3, 4])])
def test_late_alignment_reads_deep_video_loops(loops, expected):
    model = make_model(); data = inputs()
    keys, values = model.prefill_video_cache_tensor(data["video_tokens"][:, :2], data["video_freqs"][:2],
        data["video_t_mod"][:, :2], data["video_context"], data["video_context_mask"][:, :2], data["attention_mask"][:2, :2])
    mask = torch.cat((data["attention_mask"][4:, :2], data["attention_mask"][4:, 4:]), 1)
    trace = []
    rt.action_core(model, data["action_tokens"], data["action_freqs"], data["action_t_mod"], data["action_context"],
                   data["action_context_mask"], keys, values, mask, loops=loops, transport=rt.ResidualTransport("T0", 8),
                   history=None, next_tau=torch.zeros(1), trace=trace)
    slots = [slot for _, slot in trace]
    assert slots[:3] == [0, 1, 2] and slots[-3:] == [27, 28, 29]
    video_loops = sorted({(slot - 3) // 6 + 1 for slot in slots[3:-3]})
    assert video_loops == expected
    # Action keys read only cached observation slots (two observed tokens), never future video.
    assert keys[0].shape[1] == 2 and mask.shape[-1] == 2 + 3

def test_diagnostics_are_detached_and_valid_weighted():
    policy = ToyField(collect_diagnostics=True)
    loss, logs, _, _ = run(policy, (1,) * 5)
    assert "total" in logs and not any(v.requires_grad for v in logs.values())
