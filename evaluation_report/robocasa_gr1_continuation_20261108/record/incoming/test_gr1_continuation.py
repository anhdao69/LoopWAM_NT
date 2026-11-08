"""CPU tests for the RoboCasa-GR1 epochs 6-10 continuation mode of scripts/train_loopwam_gr1.py."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

import train_loopwam_gr1 as T

PROD = Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101/train/production')
SCRIPT = Path(T.__file__).resolve()
U, W, P = T.UPDATES_PER_EPOCH, T.WINDOWS, T.PARENT_UPDATES


def test_constants_match_the_data_budget():
    from fastwam.datasets.loopwam_robot_batches import training_budget
    assert training_budget(W, 256, 5) == (U, P, 218)
    assert training_budget(W, 256, 10) == (U, 2 * P, 218)


def test_continuation_lr_is_cosine_3e5_to_3e7():
    assert T.continuation_factor(P) == 1.0
    assert T.continuation_factor(2 * P - 1) == pytest.approx(.01, abs=1e-15)
    xs = [T.continuation_factor(u) for u in range(P, 2 * P, 997)]
    assert all(a > b for a, b in zip(xs, xs[1:]))
    for e in range(6, 11):   # first update of each continuation epoch
        step = (e - 6) * U
        assert T.continuation_factor(P + step) == pytest.approx(.01 + .99 * .5 * (1 + math.cos(math.pi * step / (P - 1))))
    for bad in (P - 1, 2 * P, 0):
        with pytest.raises(ValueError):
            T.continuation_factor(bad)


def _parent_contract():
    if (PROD / 'manifest.json').exists():
        return json.loads((PROD / 'manifest.json').read_text())['contract']
    return dict(epochs=5, planned_updates=P, planned_windows=5 * W, lr_backbone=1e-4, lr_robot_io=1e-4, warmup_fraction=.05,
                save_epochs='3,4,5', world=4, microbatch=16, kernels=dict(rope='real'))


def test_continuation_contract_changes_only_schedule_fields():
    parent = _parent_contract(); c = T.continuation_contract(parent, 'ab' * 32)
    changed = {k for k in parent if parent[k] != c[k]}
    assert changed == {'epochs', 'planned_updates', 'planned_windows', 'lr_backbone', 'lr_robot_io', 'warmup_fraction', 'save_epochs'}
    assert set(c) - set(parent) == {'phase_start_update', 'phase_start_epoch', 'parent_checkpoint_sha256'}
    assert (c['epochs'], c['planned_updates'], c['planned_windows'], c['lr_backbone'], c['warmup_fraction'], c['save_epochs']) == \
        (10, 10 * U, 10 * W, 3e-5, 0., '6,7,8,9,10')
    assert (c['phase_start_update'], c['phase_start_epoch']) == (P, 5)
    with pytest.raises(ValueError):
        T.continuation_contract(dict(parent, epochs=4), 'x')
    with pytest.raises(ValueError):
        T.continuation_contract(dict(parent, planned_updates=P - 1), 'x')


def test_cursor_validation():
    T.validate_cursor(P, 5 * W, 5, 0, 10)
    T.validate_cursor(6 * U, 6 * W, 6, 0, 10)
    T.validate_cursor(6 * U + 7, 6 * W + 7 * 256, 6, 7, 10)
    T.validate_cursor(10 * U, 10 * W, 10, 0, 10)
    for args in ((P, 5 * W + 1, 5, 0, 10), (P + 1, 5 * W, 5, 0, 10), (10 * U + 1, 10 * W + 256, 10, 1, 10),
                 (P + U, 5 * W + U * 256, 5, U, 10), (6 * U, 6 * W, 6, 0, 5)):
        with pytest.raises(ValueError):
            T.validate_cursor(*args)


def _payload(**over):
    state = dict(contract=_parent_contract(), update=P, windows_seen=5 * W, next_epoch=5, next_batch=0, elapsed_training_seconds=1.)
    p = dict(step=P, training_state=state, optimizer=dict(state={0: 1}, param_groups=[]))
    for k, v in over.items():
        (state if k in state else p)[k] = v
    return p


def test_parent_state_validation():
    T.validate_parent_state(_payload())
    for over in (dict(step=P - 1), dict(update=P - 1), dict(next_epoch=4), dict(next_batch=1), dict(windows_seen=5 * W - 1),
                 dict(optimizer=dict(state={}, param_groups=[]))):
        with pytest.raises(ValueError):
            T.validate_parent_state(_payload(**over))
    bad = _payload(); bad['training_state']['contract'] = dict(bad['training_state']['contract'], epochs=10)
    with pytest.raises(ValueError):
        T.validate_parent_state(bad)


def _rows(path, updates):
    path.write_text(''.join(json.dumps(dict(update=u, loss=.1)) + '\n' for u in updates))


def test_reconcile_metrics(tmp_path):
    f = tmp_path / 'metrics.jsonl'
    _rows(f, range(P + 1, P + 14))
    T.reconcile_metrics(f, P + 10, first_update=P + 1)
    assert [json.loads(l)['update'] for l in f.read_text().splitlines()] == list(range(P + 1, P + 11))
    sup = list(tmp_path.glob('metrics.superseded.*.jsonl')); assert len(sup) == 1
    assert [json.loads(l)['update'] for l in sup[0].read_text().splitlines()] == [P + 11, P + 12, P + 13]
    T.reconcile_metrics(f, P + 10, first_update=P + 1)            # idempotent
    assert len(list(tmp_path.glob('metrics.superseded.*.jsonl'))) == 1
    _rows(f, [P + 1, P + 2, P + 4]);  pytest.raises(ValueError, T.reconcile_metrics, f, P + 4, P + 1)
    _rows(f, [P + 1, P + 2, P + 2]);  pytest.raises(ValueError, T.reconcile_metrics, f, P + 2, P + 1)
    g = tmp_path / 'none.jsonl'
    pytest.raises(ValueError, T.reconcile_metrics, g, P + 1, P + 1)
    T.reconcile_metrics(g, P, P + 1)


BASE = ['--output-dir', '/nonexistent/out', '--shared-dir', '/nonexistent', '--latent-dir', '/nonexistent', '--init-artifact', 'x']
CONT = ['--epochs', '10', '--lr-backbone', '3e-5', '--lr-robot-io', '3e-5', '--warmup-fraction', '0', '--save-epochs', '6,7,8,9,10',
        '--continuation-from', '/nonexistent/latest.pt', '--expected-parent-sha256', 'ab' * 32]


def _argcheck(extra):
    env = dict(os.environ); [env.pop(k, None) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    r = subprocess.run([sys.executable, str(SCRIPT)] + BASE + extra, capture_output=True, text=True, env=env, timeout=300)
    return r.returncode, r.stderr


@pytest.mark.parametrize('extra,msg', [
    (CONT, 'requires 4 or 8 GPUs'),                                    # every recipe check passes; stops at the world check
    ([], 'requires 4 or 8 GPUs'),                                      # the original five-epoch recipe is unchanged
    (CONT[:-2], 'go together'),
    (CONT[:1] + ['5'] + CONT[2:], 'requires robocasa_gr1'),
    (CONT[:3] + ['1e-4'] + CONT[4:], 'Continuation requires'),
    (CONT[:5] + ['1e-4'] + CONT[6:], 'Continuation requires'),
    (CONT[:7] + ['.05'] + CONT[8:], 'Continuation requires'),
    (CONT[:9] + ['6,7,8,9'] + CONT[10:], 'Continuation requires'),
    (CONT + ['--allow-partial-cache'], 'Continuation requires'),
    (['--epochs', '10'], 'requires robocasa_gr1'),
    (['--expected-parent-sha256', 'ab' * 32], 'go together'),
])
def test_argument_checks(extra, msg):
    code, err = _argcheck(extra)
    assert code == 2 and msg in err, err[-400:]
