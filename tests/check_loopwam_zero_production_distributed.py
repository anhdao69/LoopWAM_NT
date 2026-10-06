#!/usr/bin/env python3
"""Four-GPU acceptance test of the production trainer's DDP/ZeRO helpers.

Run only after the parent assigns four idle GPUs:
  torchrun --standalone --nproc_per_node=4 tests/check_loopwam_zero_production_distributed.py --output-dir /tmp/unique-path

Checks full128/partial18/full128/partial6 batches, scheduled LR, shared-core
checkpointing, clipped updates and FP32 moments, CPU metadata preservation,
portable checkpoints and collective native ZeRO save/reload. Uses tiny tensors.
"""
import argparse
import contextlib
import copy
import gc
import faulthandler
import time
import traceback
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
trainer = load_module('production_trainer', os.environ.get('LOOPWAM_TRAIN_SCRIPT', ROOT / 'scripts/train_loopwam.py'))
reference = load_module('backend_reference', os.environ.get('LOOPWAM_REFERENCE_CHECK',
    Path(__file__).with_name('check_loopwam_backends_distributed.py')))


def trace(stage, **details):
    print(json.dumps(dict(event='backend_gate_trace', rank=int(os.environ.get('RANK', 0)),
                          monotonic=time.monotonic(), stage=stage, **details)), flush=True)


def traced_snapshot(model, optimizer, is_zero):
    trace('snapshot_begin', zero=is_zero)
    if not is_zero:
        result = reference.snapshot(model, optimizer, False)
    else:
        from deepspeed.utils import safe_get_full_optimizer_state
        named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        result = {'parameters': torch.cat([p.detach().flatten() for _, p in named]).cpu().clone()}
        for key in ('exp_avg', 'exp_avg_sq'):
            values = []
            for name, parameter in named:
                trace('moment_gather_begin', parameter=name, key=key)
                value = safe_get_full_optimizer_state(parameter, key)
                trace('moment_gather_return', parameter=name, key=key, found=value is not None)
                assert value is not None and value.dtype == torch.float32
                values.append(value.detach().flatten())
            result[key] = torch.cat(values).cpu().clone()
        assert all(torch.isfinite(value).all() for value in result.values())
    trace('snapshot_end', zero=is_zero)
    return result


def native_optimizer_reference(link, model, world):
    """Reconstruct the tiny gate's AdamW moments directly from native CPU shards.

    Saved param_shapes preserves DeepSpeed's flattened optimizer order, which
    need not match model order. Padding belongs at the end of each flat group.
    """
    directory = Path(link['directory']) / link['tag']
    metadata = torch.load(directory / 'mp_rank_00_model_states.pt', map_location='cpu', weights_only=False)
    groups = metadata['param_shapes']
    assert len(groups) == 1, 'This tiny gate intentionally uses one optimizer group.'
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    shapes = groups[0]
    assert set(shapes) == {name for name, _ in named}
    shards, paddings, learning_rates, steps = [], [], [], []
    for rank in range(world):
        path = directory / f'zero_pp_rank_{rank}_mp_rank_00_optim_states.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)['optimizer_state_dict']
        assert state['partition_count'] in (world, [world])
        base = state['base_optimizer_state']
        assert len(base['param_groups']) == 1
        group = base['param_groups'][0]
        assert len(group['params']) == 1
        shard = base['state'][group['params'][0]]
        shards.append(shard)
        paddings.append(state['group_paddings'][0])
        learning_rates.append(group['lr'])
        steps.append(float(shard['step']))
    assert len(set(learning_rates)) == len(set(steps)) == 1
    result = {'lr': learning_rates[0], 'step': steps[0]}
    real_size = sum(math.prod(shape) for shape in shapes.values())
    for key in ('exp_avg', 'exp_avg_sq'):
        parts = [shard[key] for shard in shards]
        assert all(part.dtype == torch.float32 and torch.isfinite(part).all() for part in parts)
        flat = torch.cat(parts)
        assert flat.numel() - sum(paddings) == real_size
        values, offset = {}, 0
        for name, shape in shapes.items():
            count = math.prod(shape)
            values[name] = flat[offset:offset + count]
            offset += count
        result[key] = torch.cat([values[name] for name, _ in named])
    return result


def assert_native_optimizer_restore(link, model, optimizer, snapshot, world):
    expected = native_optimizer_reference(link, model, world)
    for key in ('exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(snapshot[key], expected[key], rtol=0, atol=0)
    assert all(group['lr'] == expected['lr'] for group in optimizer.param_groups)
    assert all(float(state['step']) == expected['step'] for state in optimizer.optimizer.state.values())
    trace('native_optimizer_exact_match', step=expected['step'], lr=expected['lr'])


def link_restored_moments_for_inspection(optimizer):
    # DeepSpeed0.18.7 load_state_dict relinks FP32 parameter fragments, but
    # optimizer-state fragments are normally linked lazily by _optimizer_step.
    # A freshly restored engine has not stepped yet. Initialize only these
    # inspection views before calling safe_get_full_optimizer_state; no tensor
    # values or optimizer counters are modified by this installed DS method.
    trace('restored_moment_views_begin')
    optimizer._lazy_init_hp_params_optimizer_state()
    trace('restored_moment_views_end')


class Policy(reference.TinyRecurrentPolicy):
    def forward(self, sample):
        # Regresses DDP's former accidental scatter of cache metadata onto CUDA.
        assert sample['training_index'].device.type == 'cpu'
        device = self.encoder.weight.device
        return super().forward(sample['inputs'].to(device), sample['target'].to(device), sample['valid'].to(device))

    def save_checkpoint(self, path, optimizer=None, step=None, training_state=None):
        payload = dict(model=self.state_dict(), step=step, training_state=training_state)
        if optimizer is not None:
            payload['optimizer'] = optimizer.state_dict()
        torch.save(payload, path)


def groups():
    generator = torch.Generator().manual_seed(995)
    return [(torch.randn(size, 7, generator=generator), torch.randn(size, 3, generator=generator) + 5.)
            for size in (128, 18, 128, 6)]


def samples(group, rank, world):
    inputs, target = group
    for start in range(0, len(inputs), world * 2):
        ids = torch.arange(start + rank * 2, start + (rank + 1) * 2)
        valid = ids < len(inputs)
        clamped = ids.clamp_max(len(inputs) - 1)
        yield dict(inputs=inputs[clamped], target=target[clamped], valid=valid,
                   training_index=torch.where(valid, ids, -1))


def run_backend(name, initial, data, out, rank, world, device):
    args = SimpleNamespace(backend=name, fused_optimizer=True, bucket_views=True,
                           microbatch=2, global_batch=128)
    model = copy.deepcopy(initial).to(device).train()
    trace('initial_engine_begin', backend=name)
    runner, optimizer = trainer.configure_training_backend(model, args, world)
    trace('initial_engine_end', backend=name)
    parameters = model.policy_parameters()
    is_zero = name != 'ddp'
    if not is_zero:
        optimizer.zero_grad(set_to_none=True)
    states, seen_total, micro_total = [], 0, 0
    for update, group in enumerate(data):
        batches = list(samples(group, rank, world))
        for index, sample in enumerate(batches):
            boundary = index == len(batches) - 1
            if is_zero:
                runner.set_gradient_accumulation_boundary(boundary)
            sync = runner.no_sync() if not is_zero and not boundary else contextlib.nullcontext()
            with sync:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = runner(sample)
                assert torch.isfinite(loss)
                trainer.backward_training_microbatch(runner, loss, is_zero=is_zero, world=world,
                                                      microbatch=2, valid_global=len(group[0]))
            if is_zero and not boundary:
                runner.step()
        smoke = trainer.smoke_policy_gradients(model, is_zero=is_zero) if update == 0 else None
        lr = .001 * trainer.lr_factor(update, len(data), 2)
        grad, _ = trainer.step_training_backend(runner, optimizer, parameters, model, lr,
                                                is_zero=is_zero, smoke_grad_norms=smoke)
        assert math.isfinite(float(grad)) and float(grad) > 1., 'Test must exercise global clipping.'
        assert all(pg['lr'] == lr for pg in optimizer.param_groups)
        seen_total += len(group[0])
        micro_total += len(batches)
        snapshot = traced_snapshot(model, optimizer, is_zero)
        snapshot.update(grad_norm=float(grad), valid_global=len(group[0]), update=update + 1)
        states.append(snapshot)
        training_state = dict(epoch=0, next_micro=micro_total, update=update + 1,
                              windows_seen=seen_total, rng=[], contract={'global_batch': 128, 'world': world})
        trainer.save_training_checkpoint(model, runner, optimizer, out, update + 1, training_state,
                                         backend=name, rank=rank, world=world)
    portable = torch.load(out / 'latest.pt', map_location='cpu', weights_only=False)
    assert portable['training_state']['windows_seen'] == sum(len(group[0]) for group in data)
    assert ('optimizer' in portable) == (not is_zero)
    restored_policy = copy.deepcopy(initial)
    restored_policy.load_state_dict(portable['model'], strict=True)
    restored_values = torch.cat([p.detach().flatten() for p in restored_policy.policy_parameters()])
    torch.testing.assert_close(restored_values, states[-1]['parameters'], atol=0, rtol=0)
    native_link = portable['training_state'].get('native_optimizer_checkpoint')
    trace('delete_training_engine_begin', backend=name)
    del runner, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    trace('delete_training_engine_end', backend=name)
    if is_zero:
        recovered = copy.deepcopy(initial).to(device).train()
        trace('restore_engine_begin', backend=name)
        restored_engine, restored_optimizer = trainer.configure_training_backend(recovered, args, world)
        trace('restore_engine_end', backend=name)
        trace('native_load_begin', backend=name, **native_link)
        load_path, client = restored_engine.load_checkpoint(native_link['directory'], tag=native_link['tag'],
            load_module_strict=False, load_optimizer_states=True, load_lr_scheduler_states=False)
        trace('native_load_end', backend=name, path=load_path)
        assert load_path and client['training_state']['update'] == len(data)
        link_restored_moments_for_inspection(restored_optimizer)
        recovered_state = traced_snapshot(recovered, restored_optimizer, True)
        assert_native_optimizer_restore(native_link, recovered, restored_optimizer, recovered_state, world)
        for key in ('parameters', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(recovered_state[key], states[-1][key], atol=0, rtol=0)
        del restored_engine, restored_optimizer, recovered
        gc.collect()
        torch.cuda.empty_cache()
    dist.barrier()
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reload-only', choices=['zero1', 'zero2'], default=None,
                        help='Fresh-process native reload diagnostic from an existing gate output')
    args = parser.parse_args()
    faulthandler.dump_traceback_later(45, repeat=True)
    rank, world, local = (int(os.environ.get(key, default)) for key, default in
                          [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if world != 4:
        raise ValueError('This production gate requires four ranks.')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', device_id=device)
    try:
        torch.set_num_threads(2)
        if args.reload_only:
            name = args.reload_only
            torch.manual_seed(361)
            policy = Policy(checkpoint_blocks=True).to(device).train()
            payload = torch.load(args.output_dir / name / 'latest.pt', map_location='cpu', weights_only=False)
            link = payload['training_state']['native_optimizer_checkpoint']
            options = SimpleNamespace(backend=name, fused_optimizer=True, bucket_views=True, microbatch=2, global_batch=128)
            trace('reload_only_engine_begin', backend=name)
            engine, optimizer = trainer.configure_training_backend(policy, options, world)
            trace('reload_only_engine_end', backend=name)
            trace('native_load_begin', backend=name, **link)
            loaded, client = engine.load_checkpoint(link['directory'], tag=link['tag'], load_module_strict=False,
                load_optimizer_states=True, load_lr_scheduler_states=False)
            trace('native_load_end', backend=name, path=loaded)
            assert loaded
            link_restored_moments_for_inspection(optimizer)
            result = traced_snapshot(policy, optimizer, True)
            assert_native_optimizer_restore(link, policy, optimizer, result, world)
            expected = Policy(checkpoint_blocks=True)
            expected.load_state_dict(payload['model'], strict=True)
            expected_values = torch.cat([p.detach().flatten() for p in expected.policy_parameters()])
            torch.testing.assert_close(result['parameters'], expected_values, rtol=0, atol=0)
            trace('reload_only_pass', backend=name, restored_update=client['training_state']['update'])
            dist.barrier()
            return
        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            for name in ('ddp', 'zero1', 'zero2'):
                (args.output_dir / name).mkdir()
        dist.barrier()
        torch.manual_seed(361)
        scalar_report = reference.scalar_vs_fused_adamw(device)
        initial = Policy(checkpoint_blocks=True)
        data = groups()
        baseline = run_backend('ddp', initial, data, args.output_dir / 'ddp', rank, world, device)
        reports = {}
        for name in ('zero1', 'zero2'):
            actual = run_backend(name, initial, data, args.output_dir / name, rank, world, device)
            reports[name] = reference.compare(baseline, actual, name)
        if rank == 0:
            report = dict(event='four_rank_production_backend_pass', world_size=world,
                          real_group_sizes=[len(group[0]) for group in data],
                          scalar_vs_fused=scalar_report, maximum_errors=reports,
                          portable_and_native_reload=True, cpu_metadata_preserved=True)
            (args.output_dir / 'result.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
    except BaseException:
        # Print before NCCL teardown: a peer may still be inside a collective,
        # and teardown can otherwise conceal the original rank-local failure.
        trace('rank_exception')
        traceback.print_exc()
        raise
    finally:
        trace('destroy_process_group_begin')
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
