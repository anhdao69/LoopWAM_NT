"""Load a RoboTwin-50 elastic export (format loopwam-s-robot-rt-v1) as an inference-ready TransportPolicy.

The export is the teacher's loopwam-s-robot-v1 payload (FP32 frozen video expert, proprio encoder, architecture,
loop metadata) with the trained student written into mot["mixtures.action.*"], plus the LIBERO elastic fields:
transport_config, student_action, step_condition, transport, training_state. Its own format name keeps the original
RoboTwin deploy code (validate_robot_payload) from loading it as a plain policy without step conditioning.
Rollout uses only the frozen video/proprio parent, the student action expert, step condition and transport.
"""
from __future__ import annotations
import torch
from .loopwam_robot import FORMAT, create_loopwam_robot
from .transport_policy import TransportPolicy

EXPORT_FORMAT = "loopwam-s-robot-rt-v1"


def load_robot_transport(path, vae_path, device="cuda:0"):
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("format_version") != EXPORT_FORMAT or payload.get("base_format_version") != FORMAT:
        raise ValueError("Not a RoboTwin elastic export")
    config = payload["transport_config"]
    if config.get("dataset") != "robotwin50" or config.get("scope") != "pc":
        raise ValueError("RoboTwin elastic exports are full action-expert (P-c) RoboTwin-50 recipes")
    parent, _ = create_loopwam_robot(vae_path, payload=dict(payload, format_version=FORMAT), device=str(device),
                                     checkpoint_blocks=False)
    policy = TransportPolicy(parent, kind=config["kind"], scope="pc", step_cond=config["step_cond"], checkpoint_blocks=False,
                             end_prefix=config.get("end_prefix", 10), gripper_loss=config.get("gripper_loss", True))
    policy = policy.to(device).eval().requires_grad_(False)
    policy.student.mixtures["action"].load_state_dict(payload["student_action"], strict=True)
    policy.transport.load_state_dict(payload["transport"], strict=True)
    if (policy.step_condition is None) != (payload["step_condition"] is None):
        raise ValueError("Export and recipe disagree on step conditioning")
    if policy.step_condition is not None:
        policy.step_condition.load_state_dict(payload["step_condition"], strict=True)
    return policy, payload
