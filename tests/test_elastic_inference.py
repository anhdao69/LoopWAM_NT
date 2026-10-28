"""Teacher-free deployment semantics for elastic schedules (Task E, CPU part; GPU parity is in preflight)."""
import pytest
import torch
from test_elastic_distillation import ToyField
from fastwam.models.wan22.residual_transport import SCHEDULES, FLAT9
from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

class Recorder(ToyField):
    def __init__(self):
        super().__init__()
        self.parent.infer_action_scheduler = WanContinuousFlowMatchScheduler(shift=1)
        self.history_seen = []
    def velocity(self, z, tau, conditioning, cache, *, history=None, native_timestep=None, **kw):
        self.history_seen.append(dict(history=history is not None, loops=kw["loops"], d=kw["d"], t=native_timestep.clone()))
        return super().velocity(z, tau, conditioning, cache, history=history, native_timestep=native_timestep, **kw)

@pytest.mark.parametrize("name", FLAT9)
def test_rollout_uses_schedule_depths_d_and_resets_history_per_query(name):
    policy = Recorder().eval()
    noise = torch.randn(1, 32, 7)
    for _ in range(2):  # two robot queries
        policy.history_seen.clear()
        with torch.no_grad(): policy.rollout(dict(), SCHEDULES[name], noise)
        n = len(SCHEDULES[name])
        assert [c["loops"] for c in policy.history_seen] == list(SCHEDULES[name])
        assert all(c["d"] == pytest.approx(1 / n) for c in policy.history_seen)
        assert [c["history"] for c in policy.history_seen] == [False] + [True] * (n - 1)  # reset at each query
        times = torch.linspace(1, 0, n + 1)[:-1] * 1000
        torch.testing.assert_close(torch.cat([c["t"] for c in policy.history_seen]), times)

def test_inference_rejects_unknown_or_mismatched_schedule():
    from fastwam.models.wan22.transport_inference import TransportInference
    adapter = TransportInference.__new__(TransportInference)
    adapter.schedule = "flat_k1_n2"
    with pytest.raises(ValueError, match="Unknown RT schedule"):
        adapter.actions({}, torch.zeros(1, 32, 7), "flat_k3_n2")
    with pytest.raises(ValueError, match="NFE must match"):
        adapter.infer_action(context=torch.zeros(1, 4), context_mask=torch.ones(1, dtype=torch.bool),
                             proprio=torch.zeros(8), input_image=torch.zeros(3, 4, 4), num_inference_steps=10)
