"""ChronoLoop sampler / data tests (plan 3.3 "Sampler"). Needs the local full-LIBERO export."""
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from fastwam.datasets.chronoloop_data import (FullLiberoWindows, StreamScheduler, build_traversals, load_schedule,
                                              traversal_length, write_schedule)
from fastwam.datasets.loopwam_long import FULL_LIBERO_SUITES, build_full_libero_datasets

SHARED = os.environ.get('CHRONO_SHARED')
pytestmark = pytest.mark.skipif(not SHARED or not Path('data/lerobot_v30').exists(), reason='needs data')


@pytest.fixture(scope='module')
def windows():
    train, _, _ = build_full_libero_datasets('data/lerobot_v30', 'data/text_embeds_cache/libero', f'{SHARED}/data')
    return FullLiberoWindows(train)


def test_fields_match_parent_dataset(windows):
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.integers(0, len(windows), 40),
                          [e[2] + e[3] - 1 for e in windows.episodes[:5]],      # episode tails (padding)
                          [e[2] for e in windows.episodes[-5:]]])
    assert windows.verify_against_reference(idx) == len(idx)
    assert len(windows) == 277713 and len(windows.episodes) == 1712


def test_every_window_once_per_epoch_and_suite_shares(windows, tmp_path):
    lengths = np.asarray([e[3] for e in windows.episodes]); starts = np.asarray([e[2] for e in windows.episodes])
    trav = build_traversals(lengths, 2, 42)
    sched = StreamScheduler(trav, starts, lengths, 32, 4)
    seen = {0: Counter(), 1: Counter()}
    suite_count = Counter()
    while not sched.exhausted:
        slots, resets = sched.next_update()
        for row in slots:
            for x in row:
                if x is not None:
                    epoch = int(trav[x[1]][0])
                    seen[epoch][x[0]] += 1
                    if epoch == 0:
                        suite_count[windows.episodes[windows.episode_of[x[0]]][0]] += 1
    for epoch in (0, 1):
        assert len(seen[epoch]) == len(windows) and set(seen[epoch].values()) == {1}
    parent_share = {s: sum(e[3] for e in windows.episodes if e[0] == s) / len(windows) for s in FULL_LIBERO_SUITES}
    for s in FULL_LIBERO_SUITES:
        assert abs(suite_count[s] / len(windows) - parent_share[s]) < 0.01


def test_traversal_order_and_resume(windows):
    lengths = np.asarray([e[3] for e in windows.episodes]); starts = np.asarray([e[2] for e in windows.episodes])
    trav = build_traversals(lengths, 1, 3)
    a = StreamScheduler(trav, starts, lengths, 32, 4)
    out_a = [a.next_update() for _ in range(50)]
    b = StreamScheduler(trav, starts, lengths, 32, 4)
    for _ in range(20): b.next_update()
    c = StreamScheduler(trav, starts, lengths, 32, 4); c.load_state_dict(b.state_dict())
    assert [c.next_update() for _ in range(30)] == out_a[20:]
    # consecutive windows of a stream are 10 frames apart within one demo
    for slots, resets in out_a:
        for row in slots:
            xs = [x for x in row if x is not None]
            assert all(y[0] - x[0] == 10 and y[1] == x[1] for x, y in zip(xs, xs[1:]))
    assert traversal_length(25, 7) == 2 and traversal_length(5, 7) == 0


def test_shared_schedule_file(windows, tmp_path):
    path = tmp_path / 's.json'
    digest, payload = write_schedule(path, windows, epochs=1, seed=42)
    again, _ = write_schedule(path, windows, epochs=1, seed=42)
    assert digest == again == load_schedule(path)['sha256']
    summary = load_schedule(path)['summary']
    assert summary['valid_windows'] == len(windows)
    assert summary['epoch_complete_update']['1'] == summary['updates']
