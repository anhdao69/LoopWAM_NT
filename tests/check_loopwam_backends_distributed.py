#!/usr/bin/env python3
"""Two-rank numerical acceptance gate for the installed DeepSpeed backends.

Run only in an assigned GPU allocation, for example:
  torchrun --standalone --nproc_per_node=2 tests/check_loopwam_backends_distributed.py

Uses real DDP and DeepSpeed engines, fused AdamW, FP32 policy/master/moments,
BF16 compute, shared recurrent parameters, and nonreentrant checkpointing.
Full and incomplete global batches are compared after every optimizer update.
This script does not load data, a Wan checkpoint, or an existing training run.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import math
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint

from fastwam.training_backends import initialize_deepspeed_backend, deepspeed_precision_report


class TinyRecurrentPolicy(nn.Module):
    def __init__(self, checkpoint_blocks=True):
        super().__init__()
        self.encoder = nn.Linear(7, 16)
        self.core = nn.Sequential(nn.Linear(16, 24), nn.GELU(approximate='tanh'), nn.Linear(24, 16))
        self.head = nn.Linear(16, 3)
        self.vae = nn.Linear(7, 7).to(torch.bfloat16).requires_grad_(False)
        self.checkpoint_blocks = checkpoint_blocks

    def policy_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def forward(self, inputs, target, valid):
        with torch.no_grad():
            inputs = self.vae(inputs.to(torch.bfloat16)).float()
        hidden = self.encoder(inputs)
        for _ in range(4):
            update = checkpoint(self.core, hidden, use_reentrant=False) if self.checkpoint_blocks else self.core(hidden)
            hidden = hidden + .2 * update
        prediction = self.head(hidden)
        error = (prediction.float() - target.float()).square().mean(-1)
        return (error * valid.float()).mean()


def scalar_vs_fused_adamw(device):
    """Compare actual multi-step optimizer arithmetic with deterministic gradients."""
    generator = torch.Generator().manual_seed(841)
    original = [torch.randn(shape, generator=generator).to(device) for shape in [(5, 7), (7,), (2, 3, 4)]]
    scalar = [nn.Parameter(value.clone()) for value in original]
    fused = [nn.Parameter(value.clone()) for value in original]
    options = dict(lr=.003, betas=(.9, .95), eps=1e-8, weight_decay=.01)
    first = torch.optim.AdamW(scalar, foreach=False, **options)
    second = torch.optim.AdamW(fused, fused=True, **options)
    max_error = 0.
    for step in range(7):
        for index, (left, right) in enumerate(zip(scalar, fused)):
            gradient = torch.randn(left.shape, generator=generator).to(device) * (index + 1)
            left.grad, right.grad = gradient.clone(), gradient.clone()
        for optimizer, parameters in [(first, scalar), (second, fused)]:
            optimizer.param_groups[0]['lr'] = options['lr'] * (.3 + .1 * step)
            torch.nn.utils.clip_grad_norm_(parameters, .05, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for left, right in zip(scalar, fused):
            torch.testing.assert_close(left, right, atol=3e-7, rtol=3e-6)
            max_error = max(max_error, (left - right).abs().max().item())
            for key in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(first.state[left][key], second.state[right][key], atol=1e-8, rtol=3e-6)
                assert second.state[right][key].dtype == torch.float32
    return {'steps': 7, 'max_parameter_absolute_error': max_error}


def fixed_groups():
    generator = torch.Generator().manual_seed(194)
    # With global=16, micro=2, world=2, tails10 and6 need fewer than4 microsteps.
    return [(torch.randn(size, 7, generator=generator), torch.randn(size, 3, generator=generator) * 3)
            for size in (16, 10, 16, 6)]


def microbatches(group, rank, world, device):
    inputs, target = group
    microbatch = 2
    for start in range(0, len(inputs), world * microbatch):
        ids = torch.arange(start + rank * microbatch, start + (rank + 1) * microbatch)
        valid = ids < len(inputs)
        ids = ids.clamp_max(len(inputs) - 1)
        yield inputs[ids].to(device), target[ids].to(device), valid.to(device)


def snapshot(model, optimizer, is_deepspeed):
    """All ranks must call: reading full ZeRO moments involves collectives."""
    if is_deepspeed:
        from deepspeed.utils import safe_get_full_optimizer_state
    parameters = model.policy_parameters()
    values = {'parameters': torch.cat([p.detach().flatten() for p in parameters]).cpu().clone()}
    for key in ('exp_avg', 'exp_avg_sq'):
        moments = [safe_get_full_optimizer_state(p, key) if is_deepspeed else optimizer.state[p][key]
                   for p in parameters]
        assert all(value is not None and value.dtype == torch.float32 for value in moments)
        values[key] = torch.cat([value.detach().flatten() for value in moments]).cpu().clone()
    assert all(torch.isfinite(value).all() for value in values.values())
    return values


def run_backend(name, initial, groups, rank, world, device):
    model = copy.deepcopy(initial).to(device).train()
    stage = {'zero1': 1, 'zero2': 2}.get(name)
    if stage is None:
        optimizer = torch.optim.AdamW(model.policy_parameters(), lr=.003, betas=(.9, .95),
                                     eps=1e-8, weight_decay=.01, fused=True)
        runner = DDP(model, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True)
        optimizer.zero_grad(set_to_none=True)
    else:
        runner = initialize_deepspeed_backend(model, stage=stage, microbatch=2,
            global_batch=16, world_size=world, learning_rate=.003, gradient_clipping=.05,
            bucket_size=128, overlap_comm=True)
        optimizer = runner.optimizer
    result = []
    for group_index, group in enumerate(groups):
        batches = list(microbatches(group, rank, world, device))
        valid_global = len(group[0])
        seen = torch.zeros((), dtype=torch.long, device=device)
        for micro, (inputs, target, valid) in enumerate(batches):
            boundary = micro + 1 == len(batches)
            if stage is not None:
                runner.set_gradient_accumulation_boundary(boundary)
            sync_context = runner.no_sync() if stage is None and not boundary else contextlib.nullcontext()
            with sync_context:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = runner(inputs, target, valid)
                if not torch.isfinite(loss):
                    raise AssertionError(f'{name}: nonfinite loss')
                weighted = loss * (world * 2 / valid_global)
                if stage is None:
                    weighted.backward()
                else:
                    runner.backward(weighted, scale_wrt_gas=False)
            seen += valid.sum()
            if stage is not None:
                # Must call this each microbatch; only the declared boundary updates.
                old_steps = runner.global_steps
                runner.step()
                assert runner.global_steps == old_steps + int(boundary)
        dist.all_reduce(seen)
        assert int(seen) == valid_global
        if stage is None:
            norm = float(torch.nn.utils.clip_grad_norm_(model.policy_parameters(), .05, error_if_nonfinite=True))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            norm = float(runner.get_global_grad_norm())
            report = deepspeed_precision_report(runner)
            assert report['optimizer_state_initialized']
        assert math.isfinite(norm) and norm >= 0, (name, norm)
        assert norm > .05, 'Test must actually exercise gradient clipping.'
        assert model.vae.weight.dtype == torch.bfloat16 and not model.vae.weight.requires_grad
        for parameter in model.policy_parameters():
            assert parameter.dtype == torch.float32
        result.append(dict(snapshot(model, optimizer, stage is not None), grad_norm=norm,
                           valid_global=valid_global, update=group_index + 1))
    dist.barrier()
    del runner, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def compare(reference, actual, name):
    errors = {'parameters': 0., 'exp_avg': 0., 'exp_avg_sq': 0.}
    absolute_tolerance = {'parameters': 3e-5, 'exp_avg': 3e-7, 'exp_avg_sq': 3e-9}
    for expected, observed in zip(reference, actual):
        assert expected['valid_global'] == observed['valid_global']
        for key in errors:
            # BF16 kernels and FP32 collective summation may round differently.
            torch.testing.assert_close(observed[key], expected[key], atol=absolute_tolerance[key], rtol=5e-4,
                                       msg=f'{name} update={expected["update"]} {key}')
            errors[key] = max(errors[key], (observed[key] - expected[key]).abs().max().item())
        if not math.isclose(observed['grad_norm'], expected['grad_norm'], rel_tol=5e-4, abs_tol=3e-5):
            raise AssertionError(f'{name}: global pre-clip norm differs: {observed["grad_norm"]} vs {expected["grad_norm"]}')
    return errors


def check_nonfinite_guard(name, initial, rank, world, device):
    """A bad early microbatch must prevent the entire accumulated update.

    Inject NaN on rank zero only; a packed device flag and MIN reduction make
    every rank reject before the boundary optimizer step. Backward still runs
    collectively, matching a runner that postpones host synchronization.
    """
    model = copy.deepcopy(initial).to(device).train()
    before = torch.cat([p.detach().flatten() for p in model.policy_parameters()]).clone()
    stage = {'zero1': 1, 'zero2': 2}.get(name)
    if stage is None:
        optimizer = torch.optim.AdamW(model.policy_parameters(), lr=.003, fused=True)
        runner = DDP(model, device_ids=[device.index], broadcast_buffers=False, gradient_as_bucket_view=True)
    else:
        runner = initialize_deepspeed_backend(model, stage=stage, microbatch=2,
            global_batch=8, world_size=world, learning_rate=.003, bucket_size=128)
        optimizer = runner.optimizer
    group = tuple(value[:8] for value in fixed_groups()[0])
    batches = list(microbatches(group, rank, world, device))
    finite = torch.ones((), dtype=torch.uint8, device=device)
    rejected = False
    for micro, (inputs, target, valid) in enumerate(batches):
        boundary = micro + 1 == len(batches)
        if stage is not None:
            runner.set_gradient_accumulation_boundary(boundary)
        sync = runner.no_sync() if stage is None and not boundary else contextlib.nullcontext()
        with sync:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = runner(inputs, target, valid)
            if rank == 0 and micro == 0:
                loss = loss * float('nan')
            finite.logical_and_(torch.isfinite(loss.detach()))
            weighted = loss * (world * 2 / 8)
            if stage is None:
                weighted.backward()
            else:
                runner.backward(weighted, scale_wrt_gas=False)
        if boundary:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite):
                rejected = True
                break  # Deliberately before optimizer.step / boundary engine.step.
            raise AssertionError('Injected nonfinite loss escaped the device guard.')
        if stage is not None:
            runner.step()
    assert rejected
    if stage is not None:
        assert runner.global_steps == 0
    after = torch.cat([p.detach().flatten() for p in model.policy_parameters()])
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    dist.barrier()
    del runner, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return {'injected_rank': 0, 'rejected_on_all_ranks': True, 'optimizer_updates': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', nargs='+', type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument('--no-checkpoint-blocks', action='store_true')
    args = parser.parse_args()
    rank, world, local = (int(os.environ.get(key, default)) for key, default in [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if world != 2:
        raise ValueError('This acceptance test requires exactly two ranks.')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', device_id=device)
    try:
        torch.set_num_threads(2)
        torch.manual_seed(361)
        scalar_report = scalar_vs_fused_adamw(device)
        initial = TinyRecurrentPolicy(checkpoint_blocks=not args.no_checkpoint_blocks)
        groups = fixed_groups()
        reference = run_backend('ddp', initial, groups, rank, world, device)
        reports = {}
        guards = {'ddp': check_nonfinite_guard('ddp', initial, rank, world, device)}
        for stage in args.stages:
            name = f'zero{stage}'
            actual = run_backend(name, initial, groups, rank, world, device)
            reports[name] = compare(reference, actual, name)
            guards[name] = check_nonfinite_guard(name, initial, rank, world, device)
            dist.barrier()
        if rank == 0:
            print(json.dumps({'event': 'backend_correctness_pass', 'world_size': world,
                'global_sample_counts': [len(group[0]) for group in groups],
                'checkpoint_blocks': not args.no_checkpoint_blocks,
                'scalar_vs_fused_adamw': scalar_report, 'max_absolute_errors': reports,
                'nonfinite_accumulated_loss_guards': guards}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
