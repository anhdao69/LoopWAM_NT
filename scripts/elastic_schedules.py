"""Elastic schedule grid, explicit IDs and the balanced per-update configuration sequence.

Legacy RT names (parent*, S1..S8) and their random.Random(seed+update*65537).choice
selector are untouched; the elastic IDs are registered in residual_transport.SCHEDULES.
"""
from __future__ import annotations
import hashlib
import json
import random
from fastwam.models.wan22.residual_transport import FLAT9, SCHEDULES

LOOPS = (1, 2, 4)
NFES = (2, 5, 10)
FRONT6 = ("S1", "S2", "S5", "S6", "S7", "S8")
FULL3 = ("flat_k4_n2", "flat_k4_n5", "flat_k4_n10")


def build_schedule(kind: str, action_loops: int, nfe: int) -> tuple[int, ...]:
    """flat: [k]*N. front: [4]+[k]*(N-1) for k<4. Only the supported K_a/NFE grid is accepted."""
    for value, allowed, name in ((action_loops, LOOPS, "action_loops"), (nfe, NFES, "nfe")):
        if isinstance(value, bool) or not isinstance(value, int) or value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}")
    if kind == "flat":
        return (action_loops,) * nfe
    if kind == "front":
        if action_loops == 4:
            raise ValueError("A first-full schedule needs fewer than four loops after the first step")
        return (4,) + (action_loops,) * (nfe - 1)
    raise ValueError("Schedule kind must be 'flat' or 'front'")


def block_calls(schedule) -> int:
    """Action-block applications for one query: sum_j (3 + 6*k_j + 3)."""
    return sum(6 + 6 * k for k in schedule)


def make_balanced_sequence(seed: int, updates: int, names=FLAT9) -> list[str]:
    """One configuration per optimizer update; each block of len(names) updates is a permutation.

    Uses its own string-seeded generator, independent of data order, noise and module init.
    """
    if updates < 0 or len(set(names)) != len(names) or any(name not in SCHEDULES for name in names):
        raise ValueError("Invalid balanced schedule request")
    rng = random.Random(f"elastic-balanced-v1:{seed}:{','.join(names)}")
    sequence = []
    while len(sequence) < updates:
        block = list(names)
        rng.shuffle(block)
        sequence.extend(block)
    return sequence[:updates]


def sequence_sha256(sequence) -> str:
    return hashlib.sha256(json.dumps(list(sequence)).encode()).hexdigest()


def schedule_sequence(config: dict, updates: int):
    """Per-update schedule names for an elastic experiment, or None for the legacy selector."""
    sampler = config.get("sampler", "legacy_choice")
    if sampler == "legacy_choice":
        return None
    if sampler == "balanced":
        return make_balanced_sequence(config["seed"], updates, tuple(config["schedules"]))
    if sampler == "fixed":
        if len(config["schedules"]) != 1:
            raise ValueError("A fixed sampler trains exactly one schedule")
        return [config["schedules"][0]] * updates
    raise ValueError(f"Unknown schedule sampler {sampler}")


def legacy_choice(seed: int, update: int, names) -> str:
    """The historical RT selector, kept bit-for-bit (F0-PB reproduces RT-A2's stream)."""
    return random.Random(seed + update * 65537).choice(list(names))
