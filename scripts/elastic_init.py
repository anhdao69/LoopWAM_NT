#!/usr/bin/env python3
"""Hashed initial-tensor artifacts for the elastic campaign (plan section 9, Task B).

common_v1 (E0/E1/E2/SP11): the BASE10 action expert, a StepCondition and a T4 transport, each
new module drawn from its own forked, label-seeded CPU generator, so constructor order or the
transport kind cannot change the shared tensors.

rt_a2_legacy_v1 (F0-PB): replays the historical RT-A2 constructor sequence of
train_transport.main (global seeds, dataset, parent on cuda:0, TransportPolicy T4/P-b with step
conditioning), so the LoRA (CUDA RNG) and step-condition (CPU RNG after T4 consumed it) tensors
are the ones RT-A2 started from. RT-A2 published only trained weights; the reconstruction is
checked separately by replaying RT-A2's first logged updates (RT-A2-replay).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch

def label_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:15], 16)

def forked(seed, build):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return build()

def tensor_digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.view(torch.uint8).numpy().tobytes() if value.numel() else b"")
    return h.hexdigest()

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    return h.hexdigest()

def save(payload, out):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(".tmp"); torch.save(payload, temp); temp.replace(out)
    digests = {part: tensor_digest(payload[part]) for part in ("student_action", "step_condition")}
    digests.update({f"transport_{k}": tensor_digest(v) for k, v in payload["transport"].items()})
    record = dict(path=str(out), sha256=sha256_file(out), recipe=payload["recipe"],
                  teacher_sha256=payload["teacher_sha256"], tensor_sha256=digests,
                  metadata={k: v for k, v in payload.items() if k not in ("student_action", "step_condition", "transport")})
    out.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))

def common(a):
    from fastwam.models.wan22.loopwam import create_loopwam
    from fastwam.models.wan22.residual_transport import ResidualTransport, StepCondition
    parent = create_loopwam(checkpoint_path=a.teacher, vae_path=a.vae_path, device="cpu")
    action = {k: v.detach().clone() for k, v in parent.action_expert.state_dict().items()}
    seeds = dict(step_condition=label_seed(f"elastic-common-v1:step_condition:{a.seed}"),
                 T4=label_seed(f"elastic-common-v1:transport:T4:{a.seed}"))
    width = parent.action_expert.hidden_dim
    step = forked(seeds["step_condition"], lambda: StepCondition(width)).state_dict()
    t4 = forked(seeds["T4"], lambda: ResidualTransport("T4", width)).state_dict()
    save(dict(format="elastic-init-v1", recipe="common_v1", teacher_sha256=sha256_file(a.teacher),
              seed=a.seed, generator_seeds=seeds, generator="torch CPU default generator, fork_rng + manual_seed(label_seed)",
              student_action=action, step_condition=step, transport={"T4": t4}), a.out)

def legacy(a):
    # Same statement order as train_transport.main at RT source a7403b5 (unchanged at 944a07f).
    from fastwam.datasets.transport_cached import TransportCachedDataset
    from fastwam.models.wan22.loopwam import create_loopwam
    from fastwam.models.wan22.transport_policy import TransportPolicy
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0); torch.set_num_threads(4)
    device = torch.device("cuda", 0)
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data = TransportCachedDataset(a.manifest, a.latent_dir)
    if len(data) != 277713: raise ValueError("Unexpected full LIBERO window count")
    parent = create_loopwam(checkpoint_path=a.teacher, vae_path=a.vae_path, device=str(device))
    parent.requires_grad_(False).eval()
    model = TransportPolicy(parent, kind="T4", scope="pb", step_cond=True, regime="fu", anchor=True,
                            checkpoint_blocks=False, loss_weights=None).to(device)
    state = model.initial_state()
    lora = sorted(k for k in state["student_action"] if k.endswith((".A.weight", ".B.weight")))
    # Six core blocks x (self q,k,v,o + cross q,k,v,o + two FFN linears) x (A, B) = 120 tensors.
    if len(lora) != 6 * 10 * 2: raise ValueError("Expected rank-16 LoRA on ten projections in six core blocks")
    save(dict(format="elastic-init-v1", recipe="rt_a2_legacy_v1", teacher_sha256=sha256_file(a.teacher),
              seed=seed, historical_source="a7403b5aab915c0d535a37acdaaffde83fd16b8a",
              historical_recipe="RT-A2: T4, P-b, step_cond, world 1", torch_version=torch.__version__,
              gpu=torch.cuda.get_device_name(), lora_tensors=len(lora),
              student_action=state["student_action"], step_condition=state["step_condition"],
              transport={"T4": state["transport"]}), a.out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("recipe", choices=("common_v1", "rt_a2_legacy_v1"))
    p.add_argument("--teacher", required=True); p.add_argument("--vae-path", required=True)
    p.add_argument("--manifest"); p.add_argument("--latent-dir")
    p.add_argument("--seed", type=int, default=42); p.add_argument("--out", required=True)
    a = p.parse_args()
    common(a) if a.recipe == "common_v1" else legacy(a)

if __name__ == "__main__":
    main()
