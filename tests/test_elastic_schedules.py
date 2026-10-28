"""Elastic schedule registration, call counts and the balanced per-update sequence (Task A)."""
from collections import Counter
import pytest
from elastic_schedules import (FLAT9, FRONT6, build_schedule, block_calls, make_balanced_sequence,
                               sequence_sha256, schedule_sequence, legacy_choice)
from fastwam.models.wan22.residual_transport import SCHEDULES

def test_flat_grid_ids_shapes_and_counts():
    expected = {(1, 2): 24, (1, 5): 60, (1, 10): 120, (2, 2): 36, (2, 5): 90, (2, 10): 180,
                (4, 2): 60, (4, 5): 150, (4, 10): 300}
    assert FLAT9 == tuple(f"flat_k{k}_n{n}" for k, n in expected)
    for (k, n), calls in expected.items():
        schedule = build_schedule("flat", k, n)
        assert schedule == (k,) * n == SCHEDULES[f"flat_k{k}_n{n}"]
        assert block_calls(schedule) == calls

def test_front_schedules_match_legacy_ids_and_counts():
    expected = {"S1": (2, 10, 192), "S2": (1, 10, 138), "S5": (2, 5, 102), "S6": (1, 5, 78),
                "S7": (2, 2, 48), "S8": (1, 2, 42)}
    assert FRONT6 == tuple(expected)
    for name, (k, n, calls) in expected.items():
        schedule = build_schedule("front", k, n)
        assert schedule == SCHEDULES[name] and block_calls(schedule) == calls
        assert block_calls(schedule) - block_calls(build_schedule("flat", k, n)) == 6 * (4 - k)

@pytest.mark.parametrize("args", [("flat", 3, 2), ("flat", 1, 4), ("flat", 0, 2), ("front", 4, 5),
                                  ("diag", 1, 2), ("flat", True, 2), ("flat", 1.0, 2)])
def test_invalid_schedule_requests(args):
    with pytest.raises(ValueError):
        build_schedule(*args)

def test_legacy_schedule_names_and_order_unchanged():
    legacy = ["parent10", "parent20", "parent5", "parent4", "parent2", "parent1",
              "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    assert list(SCHEDULES)[:len(legacy)] == legacy
    assert list(SCHEDULES)[len(legacy):] == list(FLAT9)

def test_balanced_blocks_and_production_frequencies():
    sequence = make_balanced_sequence(42, 21700)
    assert len(sequence) == 21700
    for start in range(0, 21699 - 8, 9):
        assert sorted(sequence[start:start + 9]) == sorted(FLAT9)
    counts = Counter(sequence)
    assert set(counts) == set(FLAT9)
    assert sorted(counts.values()) == [2411] * 8 + [2412]
    # Not a fixed cyclic order: blocks are reshuffled.
    assert len({tuple(sequence[i:i + 9]) for i in range(0, 900, 9)}) > 50

def test_balanced_sequence_is_deterministic_and_prefix_stable():
    a = make_balanced_sequence(42, 21700)
    assert a == make_balanced_sequence(42, 21700)
    assert sequence_sha256(a) == sequence_sha256(make_balanced_sequence(42, 21700))
    assert make_balanced_sequence(42, 1000) == a[:1000]  # an interrupted run regenerates the same stream
    assert make_balanced_sequence(43, 21700) != a

def test_balanced_sequence_does_not_touch_global_rng():
    import random
    random.seed(7); expected = random.random()
    random.seed(7); make_balanced_sequence(42, 500); assert random.random() == expected

def test_experiment_samplers():
    from transport_experiments import ELASTIC
    seq = {name: schedule_sequence(ELASTIC[name], 21700) for name in ("E0", "E1", "E2", "SP11")}
    assert seq["E0"] == seq["E1"] == seq["E2"]  # one shared schedule stream for E0/E1/E2
    assert set(seq["SP11"]) == {"flat_k1_n2"} and len(seq["SP11"]) == 21700
    assert schedule_sequence(ELASTIC["F0-PB"], 21700) is None
    import random
    for update in (0, 1, 57, 21699):
        assert legacy_choice(42, update, FRONT6) == random.Random(42 + update * 65537).choice(list(FRONT6))
