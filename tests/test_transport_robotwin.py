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


@pytest.mark.parametrize("extra", [[], ["--max-updates", "2000"]])
def test_trainer_refuses_robotwin_production_runs(tmp_path, extra):
    out = subprocess.run([sys.executable, str(ROOT / "scripts/train_transport.py"), "--run-name", "RT50-E0", "--output-dir",
                          str(tmp_path / "run"), "--expected-gpus", "1", *extra], capture_output=True, text=True)
    assert out.returncode == 2 and "--max-updates <2000" in out.stderr
    assert not (tmp_path / "run").exists()


def test_trainer_accepts_eight_gpus_and_keeps_libero_rules(tmp_path):
    import os
    out = subprocess.run([sys.executable, str(ROOT / "scripts/train_transport.py"), "--run-name", "E0", "--output-dir",
                          str(tmp_path / "run"), "--expected-gpus", "8", "--epochs", "3"], capture_output=True, text=True,
                         env=dict(os.environ, WORLD_SIZE="8"))
    # 8 GPUs pass the layout checks; LIBERO's ten-epoch rule still refuses before any work.
    assert out.returncode == 2 and "Production training must cover ten epochs" in out.stderr
    assert not (tmp_path / "run").exists()
