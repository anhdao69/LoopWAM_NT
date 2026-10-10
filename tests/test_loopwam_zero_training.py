"""CPU contracts for the production DDP/ZeRO trainer integration."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch

SCRIPT = Path(os.environ.get('LOOPWAM_TRAIN_SCRIPT', Path(__file__).resolve().parents[1] / 'scripts/train_loopwam.py'))
spec = importlib.util.spec_from_file_location('loopwam_zero_trainer_tested', SCRIPT)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proprio_encoder = torch.nn.Linear(2, 1)
    def policy_parameters(self):
        return list(self.parameters())


@pytest.mark.parametrize('is_zero', [False, True])
def test_four_rank_partial_global_batches_have_exact_mean_gradients(is_zero):
    world, micro, global_batch, size = 4, 8, 128, 139
    batches = [list(trainer.ExactDistributedBatches(size, micro, rank, world)) for rank in range(world)]
    accum = global_batch // (world * micro)
    for group_start in range(0, len(batches[0]), accum):
        valid = min(global_batch, size - group_start * world * micro)
        rank_grads, all_values = [], []
        for rank in range(world):
            parameter = torch.tensor(1., requires_grad=True)
            class Engine:
                def backward(self, loss, *, scale_wrt_gas):
                    assert scale_wrt_gas is False
                    loss.backward()
            for step in range(group_start, min(group_start + accum, len(batches[rank]))):
                values = [float(index + 1) for index in batches[rank][step] if index >= 0]
                all_values.extend(values)
                loss = parameter * (sum(values) / micro)
                trainer.backward_training_microbatch(Engine(), loss, is_zero=is_zero,
                    world=world, microbatch=micro, valid_global=valid)
            rank_grads.append(parameter.grad)
        torch.testing.assert_close(torch.stack(rank_grads).mean(), torch.tensor(sum(all_values) / len(all_values)))
        assert len(all_values) == valid


def test_backend_factory_preserves_ddp_defaults_and_zero_fp32_optimizer(monkeypatch):
    import fastwam.training_backends as backends
    model = TinyPolicy()
    args = SimpleNamespace(backend='ddp', fused_optimizer=False, bucket_views=False, microbatch=8, global_batch=128)
    runner, optimizer = trainer.configure_training_backend(model, args, world=1)
    assert runner is model and optimizer.defaults['lr'] == 1e-4
    assert optimizer.defaults['betas'] == (.9, .95) and optimizer.defaults['fused'] is False
    calls = []
    def fake_initialize(module, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(optimizer=SimpleNamespace(check_grad_overflow=False))
    monkeypatch.setattr(backends, 'initialize_deepspeed_backend', fake_initialize)
    args.backend = 'zero2'
    runner, optimizer = trainer.configure_training_backend(model, args, world=4)
    assert calls == [dict(stage=2, microbatch=8, global_batch=128, world_size=4, learning_rate=1e-4, fused=True)]
    assert optimizer.check_grad_overflow is True


def test_step_sets_lr_before_zero_update_and_rejects_bad_global_norm():
    optimizer = SimpleNamespace(param_groups=[{'lr': 99.}], overflow=False)
    seen = []
    runner = SimpleNamespace(step=lambda: seen.append(optimizer.param_groups[0]['lr']), get_global_grad_norm=lambda: 4.)
    grad, branch = trainer.step_training_backend(runner, optimizer, [], None, .0002,
                                                 is_zero=True, smoke_grad_norms={'video': 4.})
    assert seen == [.0002] and grad == 4.
    assert branch['video'] == pytest.approx(1., abs=1e-6)
    for value in (-1., float('nan'), float('inf')):
        runner.get_global_grad_norm = lambda: value
        with pytest.raises(FloatingPointError):
            trainer.step_training_backend(runner, optimizer, [], None, .001, is_zero=True)
    optimizer.overflow = True
    with pytest.raises(FloatingPointError, match='rejected'):
        trainer.step_training_backend(runner, optimizer, [], None, .001, is_zero=True)


def test_smoke_gradient_coverage_checks_missing_and_nonfinite():
    model = TinyPolicy()
    with pytest.raises(AssertionError, match='Missing'):
        trainer.smoke_policy_gradients(model, is_zero=False)
    model.proprio_encoder(torch.ones(2, 2)).sum().backward()
    norms = trainer.smoke_policy_gradients(model, is_zero=False)
    assert set(norms) == {'proprio'} and norms['proprio'] > 0
    model.proprio_encoder.weight.grad.fill_(float('inf'))
    with pytest.raises(FloatingPointError):
        trainer.smoke_policy_gradients(model, is_zero=False)


def test_zero_checkpoint_collective_and_portable_state_are_separate(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(trainer.dist, 'barrier', lambda: calls.append('barrier'))
    class Engine:
        def save_checkpoint(self, directory, **kwargs):
            calls.append(('native', kwargs))
            (Path(directory) / kwargs['tag']).mkdir(parents=True, exist_ok=True)
    class Model:
        def save_checkpoint(self, path, **kwargs):
            calls.append(('portable', kwargs))
            torch.save(kwargs, path)
    state = dict(epoch=0, next_micro=4, update=1, windows_seen=128, contract={'global_batch': 128}, rng=[])
    trainer.save_training_checkpoint(Model(), Engine(), object(), tmp_path, 1, state,
                                     backend='zero1', rank=1, world=4)
    assert [call[0] for call in calls if isinstance(call, tuple)] == ['native']
    assert not (tmp_path / 'latest.pt').exists()
    for update in (1, 2, 3):
        trainer.save_training_checkpoint(Model(), Engine(), object(), tmp_path, update,
            {**state, 'update': update}, backend='zero1', rank=0, world=4)
    payload = torch.load(tmp_path / 'latest.pt', weights_only=False)
    assert payload['optimizer'] is None
    saved = payload['training_state']
    assert saved['contract'] == state['contract'] and saved['backend'] == 'zero1'
    assert saved['native_optimizer_checkpoint']['tag'] == 'step_00000003'
    assert sorted(path.name for path in (tmp_path / 'deepspeed').iterdir()) == ['step_00000002', 'step_00000003']
    native = [entry[1] for entry in calls if isinstance(entry, tuple) and entry[0] == 'native']
    assert all(entry['exclude_frozen_parameters'] for entry in native)


def test_cli_explicitly_rejects_zero_resume_before_gpu_initialization(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), '--backend', 'zero1', '--resume', 'unused.pt',
                             '--output-dir', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 2
    assert '--resume is currently supported only for DDP' in result.stderr


@pytest.mark.parametrize("backend", ["ddp", "zero1"])
def test_retained_epoch_survives_latest_replacement_and_native_pruning(tmp_path, backend):
    class Engine:
        def save_checkpoint(self, directory, **kwargs):
            (Path(directory) / kwargs['tag']).mkdir(parents=True, exist_ok=True)
    class Model:
        def save_checkpoint(self, path, **kwargs):
            temporary = path.with_suffix('.tmp')
            torch.save(kwargs, temporary)
            temporary.replace(path)
    for epoch in range(4, 9):
        state = dict(epoch=epoch-1, next_micro=1, update=epoch, windows_seen=epoch*128, rng=[])
        trainer.save_training_checkpoint(Model(), Engine(), None, tmp_path, epoch, state,
            backend=backend, rank=0, world=1, retain_epoch=epoch if epoch>=5 else None)
    assert not (tmp_path/'epoch_004.pt').exists()
    for epoch in range(5, 9):
        payload=torch.load(tmp_path/f'epoch_{epoch:03d}.pt', weights_only=False)
        assert payload['step']==epoch
        if backend!='ddp':
            native=payload['training_state']['native_optimizer_checkpoint']
            assert (Path(native['directory'])/native['tag']).is_dir()
    assert (tmp_path/'latest.pt').stat().st_ino==(tmp_path/'epoch_008.pt').stat().st_ino
