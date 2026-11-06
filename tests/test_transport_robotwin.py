"""CPU checks of the RoboTwin-50 elastic port: LIBERO defaults unchanged, RoboTwin loss shape, recipes, trainer guards."""
import copy
import subprocess
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def legacy_endpoint_losses(student, teacher, scale, offset):
    """The pre-port endpoint_losses, verbatim."""
    from fastwam.models.wan22.residual_transport import gripper_command
    end = (student[:, :10].float() - teacher[:, :10].float()).square().mean((1, 2))
    target = gripper_command(teacher[:, :10], scale, offset)
    sign = torch.where(target >= 0, 1., -1.)
    grip = F.softplus(-5 * sign * gripper_command(student[:, :10], scale, offset)).mean(1)
    return end, grip


def test_libero_endpoint_losses_unchanged():
    from fastwam.models.wan22.residual_transport import endpoint_losses
    g = torch.Generator().manual_seed(0)
    student = torch.randn(4, 32, 7, generator=g); teacher = torch.randn(4, 32, 7, generator=g)
    scale = torch.rand(7, generator=g) + .5; offset = torch.randn(7, generator=g)
    for got, want in zip(endpoint_losses(student, teacher, scale, offset), legacy_endpoint_losses(student, teacher, scale, offset)):
        assert torch.equal(got, want)


def test_robotwin_endpoint_losses_prefix24_no_gripper():
    from fastwam.models.wan22.residual_transport import endpoint_losses
    student = torch.randn(3, 32, 14, requires_grad=True); teacher = torch.randn(3, 32, 14)
    end, grip = endpoint_losses(student, teacher, torch.ones(14), torch.zeros(14), prefix=24, gripper=False)
    assert end.shape == grip.shape == (3,) and torch.count_nonzero(grip) == 0
    assert torch.allclose(end, (student[:, :24] - teacher[:, :24]).square().mean((1, 2)))
    end.sum().backward()
    assert torch.count_nonzero(student.grad[:, 24:]) == 0 and torch.count_nonzero(student.grad[:, :24]) > 0


def test_recipes():
    from transport_experiments import ELASTIC
    for base in ("E0", "E1", "E2"):
        rt = ELASTIC[f"RT50-{base}"]
        assert {k: v for k, v in rt.items() if k not in ("parent", "dataset", "end_prefix", "gripper_loss")} == ELASTIC[base]
        assert (rt["dataset"], rt["end_prefix"], rt["gripper_loss"]) == ("robotwin50", 24, False)
        assert rt["parent"]["sha256"] == "70cf5af73db5fe582c50c5f9fb68296d5271a9c0fea51509b3cdb5f9696be1ea"
        # Epoch-5 teacher: the identical recipe, only the parent differs.
        ep5 = ELASTIC[f"RT50-ep5-{base}"]
        assert {k: v for k, v in ep5.items() if k != "parent"} == {k: v for k, v in rt.items() if k != "parent"}
        assert ep5["parent"] == dict(name="LoopWAM-v0-full-RoboTwin epoch 5", repository="anhdao69/LoopWAM_NT",
            file="robotwin/v0-video4-action4/policy.pt", format="loopwam-s-robot-v1", step=237310,
            sha256="38975220e9441303878d2dfb9956d57e6416c6a906c99c43d9bca015b092fc51")
        # LIBERO / CL-0 recipes carry no RoboTwin keys, so their resume contracts are unchanged.
        for name in (base, f"CL0-{base}"):
            assert not {"dataset", "end_prefix", "gripper_loss"} & set(ELASTIC[name])


def test_sample_noise_default_stream_unchanged():
    from train_transport import sample_noise
    noise, anchor, tau = sample_noise([5, -1, 17], 2, 42)
    g = torch.Generator().manual_seed(42 + 1000003 * 2 + 97 * 6)
    assert torch.equal(noise[0], torch.randn(32, 7, generator=g)) and torch.equal(anchor[0], torch.randn(32, 7, generator=g))
    assert torch.equal(tau[0:1], torch.rand(1, generator=g))
    robot = sample_noise([5], 2, 42, dim=14)[0]
    g = torch.Generator().manual_seed(42 + 1000003 * 2 + 97 * 6)
    assert robot.shape == (1, 32, 14) and torch.equal(robot[0], torch.randn(32, 14, generator=g))


def test_policy_forwards_endpoint_settings():
    from test_loop_mot import make_model
    from fastwam.models.wan22.transport_policy import TransportPolicy
    from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
    mot = make_model()
    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mot = mot; self.action_expert = mot.mixtures["action"]; self.video_expert = mot.mixtures["video"]
            self.device = "cpu"; self.train_action_scheduler = WanContinuousFlowMatchScheduler(shift=1)
    default = TransportPolicy(Parent(), kind="T0")
    robot = TransportPolicy(Parent(), kind="T0", end_prefix=24, gripper_loss=False)
    assert (default.end_prefix, default.gripper_loss) == (10, True) and (robot.end_prefix, robot.gripper_loss) == (24, False)


def test_export_epochs():
    from train_transport import export_epochs
    assert export_epochs(False, 10) == export_epochs(False, 3) == {8, 9, 10}   # LIBERO protocol unchanged
    assert export_epochs(True, 1) == {1} and export_epochs(True, 3) == {1, 2, 3}


def _tiny_robot_policy(tmp_path):
    """A tiny P-c policy over test_loop_mot's model plus a fake loopwam-s-robot-v1 teacher file of that parent."""
    from test_loop_mot import make_model
    from fastwam.models.wan22.transport_policy import TransportPolicy
    from fastwam.models.wan22.schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
    from fastwam.models.wan22.loopwam_init import sha256_file
    from train_transport import ROBOT_METADATA
    torch.manual_seed(0)
    mot = make_model()
    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mot = mot; self.action_expert = mot.mixtures["action"]; self.video_expert = mot.mixtures["video"]
            self.proprio_encoder = torch.nn.Linear(14, 8)
            self.device = "cpu"; self.train_action_scheduler = WanContinuousFlowMatchScheduler(shift=1)
    parent = Parent()
    teacher = {k: f"meta-{k}" for k in ROBOT_METADATA}
    teacher.update(format_version="loopwam-s-robot-v1", mot={k: v.clone() for k, v in parent.mot.state_dict().items()},
                   proprio_encoder={k: v.clone() for k, v in parent.proprio_encoder.state_dict().items()}, step=237310,
                   training_state=dict(next_epoch=5, contract=dict(data_manifest_sha256="d")))
    path = tmp_path / "teacher.pt"; torch.save(teacher, path)
    model = TransportPolicy(parent, kind="T4", scope="pc", step_cond=True, end_prefix=24, gripper_loss=False)
    with torch.no_grad():   # "training": move every trainable tensor
        for q in model.parameters():
            if q.requires_grad: q.add_(torch.randn_like(q) * 1e-3)
    contract = dict(dataset="robotwin50", teacher=str(path), asset_sha256=dict(teacher=sha256_file(path)),
                    global_batch=128, windows=6075103, config=dict(kind="T4", scope="pc"))
    return model, teacher, contract


@pytest.mark.parametrize("early", [False, True])
def test_robot_export_layout(tmp_path, early):
    from train_transport import export_epoch, ROBOT_EXPORT_FORMAT, ROBOT_METADATA
    model, teacher, contract = _tiny_robot_policy(tmp_path)
    out = tmp_path / "run"; out.mkdir()
    path = export_epoch(model, out, 1, contract["config"], 2000 if early else 47462, contract, early=early)
    assert path == out / ("step_00002000.pt" if early else "epoch_01.pt") and sorted(p.name for p in out.iterdir()) == [path.name]
    x = torch.load(path, weights_only=False)
    assert x["format_version"] == ROBOT_EXPORT_FORMAT and x["base_format_version"] == "loopwam-s-robot-v1"
    assert all(x[k] == teacher[k] for k in ROBOT_METADATA) and x["step"] == (2000 if early else 47462)
    A = "mixtures.action."
    student = model.student.mixtures["action"].state_dict()
    assert set(x["mot"]) == set(teacher["mot"]) and set(x["student_action"]) == set(student)
    for k, v in x["mot"].items():
        want = student[k[len(A):]] if k.startswith(A) else teacher["mot"][k]
        assert v.dtype == want.dtype and torch.equal(v, want), k
    assert any(not torch.equal(student[k], teacher["mot"][A + k]) for k in student)   # the student really trained
    assert all(torch.equal(x["proprio_encoder"][k], v) for k, v in teacher["proprio_encoder"].items())
    assert all(torch.equal(x["transport"][k], v) for k, v in model.transport.state_dict().items())
    assert all(torch.equal(x["step_condition"][k], v) for k, v in model.step_condition.state_dict().items())
    st = x["training_state"]
    assert st["contract"] == contract and st["update"] == x["step"] and st["merged_lora"] is False
    assert st["windows_seen"] == (2000 * 128 if early else 6075103) and st["epoch"] == 1
    assert st["parent_step"] == 237310 and st["parent_training_state"] == teacher["training_state"]


def test_robot_export_refuses_changed_teacher(tmp_path):
    from train_transport import export_epoch
    model, teacher, contract = _tiny_robot_policy(tmp_path)
    contract["asset_sha256"]["teacher"] = "0" * 64
    with pytest.raises(ValueError, match="Teacher file changed"):
        export_epoch(model, tmp_path, 1, contract["config"], 1, contract)
    assert not (tmp_path / "epoch_01.pt").exists()


def test_trainer_has_no_robotwin_update_cap():
    source = (ROOT / "scripts/train_transport.py").read_text()
    assert "--max-updates <2000" not in source and "NotImplementedError" not in source


def test_trainer_accepts_eight_gpus_and_keeps_libero_rules(tmp_path):
    import os
    out = subprocess.run([sys.executable, str(ROOT / "scripts/train_transport.py"), "--run-name", "E0", "--output-dir",
                          str(tmp_path / "run"), "--expected-gpus", "8", "--epochs", "3"], capture_output=True, text=True,
                         env=dict(os.environ, WORLD_SIZE="8"))
    # 8 GPUs pass the layout checks; LIBERO's ten-epoch rule still refuses before any work.
    assert out.returncode == 2 and "Production training must cover ten epochs" in out.stderr
    assert not (tmp_path / "run").exists()
