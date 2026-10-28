"""Parameter scope and common-initialization contracts for the elastic campaign (Task B)."""
import torch
from test_loop_mot import make_model
from fastwam.models.wan22.transport_policy import TransportPolicy
from fastwam.models.wan22.residual_transport import ResidualTransport, StepCondition
from elastic_init import forked, label_seed

class Parent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mot = make_model(); self.action_expert = self.mot.mixtures["action"]
        self.video_expert = self.mot.mixtures["video"]; self.device = "cpu"

def common_state(parent, seed=42):
    width = parent.action_expert.hidden_dim
    return dict(student_action={k: v.clone() for k, v in parent.action_expert.state_dict().items()},
                step_condition=forked(label_seed(f"elastic-common-v1:step_condition:{seed}"), lambda: StepCondition(width)).state_dict(),
                transport={"T4": forked(label_seed(f"elastic-common-v1:transport:T4:{seed}"), lambda: ResidualTransport("T4", width)).state_dict()})

def elastic_policy(parent, kind, global_seed, state):
    torch.manual_seed(global_seed)  # different constructor RNG state per arm must not matter
    policy = TransportPolicy(parent, kind=kind, scope="pc", step_cond=True, reuse_teacher_prefix=True, collect_diagnostics=True)
    parts = ["student_action", "step_condition"]
    loaded = dict(student_action=state["student_action"], step_condition=state["step_condition"])
    if kind == "T4":
        loaded["transport"] = state["transport"]["T4"]; parts.append("transport")
    policy.load_initial_state(loaded, parts)
    return policy

def test_pc_scope_trains_whole_student_action_and_freezes_parent():
    parent = Parent(); policy = elastic_policy(parent, "T0", 1, common_state(parent))
    action = policy.student.mixtures["action"]
    assert all(p.requires_grad for p in action.parameters())
    assert all(p.requires_grad for p in policy.step_condition.parameters())
    assert not any(p.requires_grad for p in parent.parameters())
    assert policy.student.mixtures["video"] is parent.video_expert  # shared, frozen video expert
    names = {n for n, p in policy.named_parameters() if p.requires_grad}
    for part in ("action_encoder", "text_embedding", "time_embedding", "time_projection", "head", "blocks.0.", "blocks.4.", "blocks.11."):
        assert any(n.startswith("student.mixtures.action." + part) for n in names), part
    assert not any(n.startswith(("parent.", "delta_encoder.", "student.mixtures.video.")) for n in names)

def test_teacher_and_student_storage_distinct_but_initially_equal():
    parent = Parent(); policy = elastic_policy(parent, "T0", 1, common_state(parent))
    teacher = dict(parent.action_expert.named_parameters())
    for name, p in policy.student.mixtures["action"].named_parameters():
        assert p.data_ptr() != teacher[name].data_ptr()
        assert torch.equal(p, teacher[name])

def test_common_initialization_independent_of_transport_kind_and_constructor_rng():
    parent = Parent(); state = common_state(parent)
    arms = {kind: elastic_policy(parent, kind, seed, state) for kind, seed in (("T0", 1), ("T1", 2), ("T4", 3))}
    reference = arms["T0"].initial_state()
    for kind, policy in arms.items():
        current = policy.initial_state()
        for part in ("student_action", "step_condition"):
            assert current[part].keys() == reference[part].keys()
            assert all(torch.equal(current[part][k], reference[part][k]) for k in reference[part]), (kind, part)
    # Step-condition output starts at zero; T4 starts near 0.95 identity.
    assert torch.count_nonzero(arms["T4"].step_condition(torch.tensor([.1, .5]))) == 0
    assert torch.equal(arms["T4"].transport.state_dict()["B.weight"], torch.zeros_like(arms["T4"].transport.B.weight))
    # Rebuilding the artifact under a different global RNG state gives identical new-module tensors.
    torch.manual_seed(999); again = common_state(parent)
    for part in ("step_condition",):
        assert all(torch.equal(again[part][k], state[part][k]) for k in state[part])
    assert all(torch.equal(again["transport"]["T4"][k], state["transport"]["T4"][k]) for k in state["transport"]["T4"])

def test_deduplicated_parameter_counts_and_optimizer_groups():
    parent = Parent(); state = common_state(parent)
    width = parent.action_expert.hidden_dim
    step = sum(p.numel() for p in StepCondition(width).parameters())
    t4 = sum(p.numel() for p in ResidualTransport("T4", width).parameters())
    action = sum(p.numel() for p in parent.action_expert.parameters())
    for kind, extra in (("T0", 0), ("T1", 0), ("T4", t4)):
        policy = elastic_policy(parent, kind, 0, state)
        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        assert trainable == action + step + extra
        groups = policy.optimizer_groups()
        by_lr = {g["lr"]: (g["weight_decay"], sum(p.numel() for p in g["params"])) for g in groups}
        assert by_lr == {1e-3: (0., step + extra), 3e-5: (.01, action)}
    # Real-model arithmetic from the plan (verified at runtime by the GPU preflight).
    assert sum(p.numel() for p in StepCondition(512).parameters()) == 400_512
    assert sum(p.numel() for p in ResidualTransport("T4", 512).parameters()) == 76_448

def test_legacy_reconstruction_copies_rt_a2_init_into_t0():
    parent = Parent()
    torch.manual_seed(42)
    historical = TransportPolicy(parent, kind="T4", scope="pb", step_cond=True)  # RT-A2 constructor order
    state = historical.initial_state()
    torch.manual_seed(42)
    naive = TransportPolicy(parent, kind="T0", scope="pb", step_cond=True)
    # Constructing T0 directly consumes less RNG, so its step condition differs from RT-A2's.
    assert not torch.equal(naive.step_condition.net[0].weight, historical.step_condition.net[0].weight)
    torch.manual_seed(5)
    f0 = TransportPolicy(parent, kind="T0", scope="pb", step_cond=True)
    f0.load_initial_state(state, ("student_action", "step_condition"))
    current = f0.initial_state()
    for part in ("student_action", "step_condition"):
        assert all(torch.equal(current[part][k], state[part][k]) for k in state[part])
    lora = [k for k in current["student_action"] if k.endswith(".A.weight")]
    assert lora and all(torch.count_nonzero(current["student_action"][k]) > 0 for k in lora)
