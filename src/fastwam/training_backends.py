"""Precision-preserving DeepSpeed backends for LoopWAM throughput comparisons.

Audited against installed DeepSpeed 0.18.7 engine.py, torch_autocast.py and
zero/stage_1_and_2.py. ``bf16.enabled`` would cast the entire model; native
``torch_autocast`` instead leaves policy/master weights FP32 and respects the
frozen VAE's existing dtype. External autocast is disabled by DeepSpeed unless
its ``torch_autocast`` configuration is also enabled.

The caller owns batches, exact sample accounting, LR scheduling and checkpoints::

    engine = initialize_deepspeed_backend(model, stage=2, microbatch=8,
                                         global_batch=128, world_size=2)
    for sample in microbatches:
        engine.set_gradient_accumulation_boundary(boundary)  # before forward
        loss, logs = engine(sample)  # engine enables BF16 autocast
        engine.backward(loss * (world_size * microbatch / valid_global),
                        scale_wrt_gas=False)
        engine.step()  # every microbatch; updates/clears grads only at boundary

Do not use DDP/no_sync, loss.backward, manual clipping, optimizer.step or
optimizer.zero_grad on this path. ZeRO-2 reduces/partitions every microbatch;
ZeRO-1 reduces at accumulation boundaries. The sample weighting above assumes
padded tail entries already contribute zero loss, as in train_loopwam.py.
"""
from __future__ import annotations

import math

import torch


def make_deepspeed_config(stage: int, microbatch: int, global_batch: int = 128,
                          world_size: int = 2, *, gradient_clipping: float = 1.0,
                          bucket_size: int = 50_000_000,
                          overlap_comm: bool = True) -> dict:
    """ZeRO-1/2, FP32 storage/gradient communication, BF16 forward computation.

    Bucket sizes are tensor elements (50M FP32 elements = 200MB). An empty
    lower_precision_safe_modules list prevents automatic BF16 communication
    attributes on Linear/Conv parameters, keeping this comparison at FP32
    communication precision like the reference DDP run.
    """
    if isinstance(stage, bool) or stage not in (1, 2):
        raise ValueError('Only ZeRO stages 1 and 2 are supported.')
    for name, value in [('microbatch', microbatch), ('global_batch', global_batch),
                        ('world_size', world_size), ('bucket_size', bucket_size)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if global_batch % (world_size * microbatch):
        raise ValueError('Global batch must be divisible by world_size * microbatch.')
    if not math.isfinite(gradient_clipping) or gradient_clipping <= 0:
        raise ValueError('gradient_clipping must be finite and positive.')
    return {
        'train_batch_size': global_batch,
        'train_micro_batch_size_per_gpu': microbatch,
        'gradient_accumulation_steps': global_batch // (world_size * microbatch),
        'gradient_clipping': gradient_clipping,
        'fp16': {'enabled': False},
        'bf16': {'enabled': False},
        'torch_autocast': {'enabled': True, 'dtype': 'bfloat16',
                           'lower_precision_safe_modules': []},
        'data_types': {'grad_accum_dtype': 'fp32'},
        'communication_data_type': 'fp32',
        'gradient_predivide_factor': 1.0,
        'zero_allow_untested_optimizer': True,
        'zero_optimization': {
            'stage': stage, 'overlap_comm': bool(overlap_comm),
            'contiguous_gradients': True, 'reduce_scatter': True,
            'allgather_partitions': True, 'reduce_bucket_size': bucket_size,
            'allgather_bucket_size': bucket_size,
            'ignore_unused_parameters': False,
        },
        'steps_per_print': 1_000_000,
        'wall_clock_breakdown': False,
    }


def policy_parameters_fp32(model):
    """Check that the optimizer owns every trainable tensor exactly once."""
    parameters = list(model.policy_parameters())
    ids = [id(parameter) for parameter in parameters]
    if not parameters or len(ids) != len(set(ids)):
        raise ValueError('Policy optimizer parameters must be nonempty and deduplicated.')
    if set(ids) != {id(parameter) for parameter in model.parameters() if parameter.requires_grad}:
        raise ValueError('Policy optimizer parameters must cover all and only trainable tensors.')
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError('Policy parameters must remain FP32 for this backend comparison.')
    return parameters


def initialize_deepspeed_backend(model, *, stage: int, microbatch: int,
                                 global_batch: int = 128, world_size: int = 2,
                                 learning_rate: float = 1e-4, gradient_clipping: float = 1.0,
                                 bucket_size: int = 50_000_000,
                                 overlap_comm: bool = True, fused: bool = True):
    """Return the actual DeepSpeedEngine wrapping an already constructed policy.

    Call torch.distributed.init_process_group first (also for world_size=1).
    ``engine.optimizer`` is the ZeRO wrapper; its nested ``optimizer`` is the
    supplied torch.optim.AdamW with FP32 flattened master partitions. Fresh
    optimizer state is intentional; this factory does not resume old DDP state.
    """
    import deepspeed

    if not torch.distributed.is_initialized():
        raise RuntimeError('Initialize torch.distributed before the DeepSpeed backend.')
    if torch.distributed.get_world_size() != world_size:
        raise ValueError('Configured world_size differs from initialized process group.')
    config = make_deepspeed_config(stage, microbatch, global_batch, world_size,
        gradient_clipping=gradient_clipping, bucket_size=bucket_size, overlap_comm=overlap_comm)
    parameters = policy_parameters_fp32(model)
    dtypes_before = {id(value): value.dtype for value in model.parameters()}
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, betas=(.9, .95),
        eps=1e-8, weight_decay=.01, fused=fused)
    engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=parameters,
        optimizer=optimizer, config=config, dist_init_required=False)
    if any(value.dtype != dtypes_before[id(value)] for value in model.parameters()):
        raise RuntimeError('DeepSpeed changed a policy or frozen-module parameter dtype.')
    if any(parameter.dtype != torch.float32
           for parameter in engine.optimizer.single_partition_of_fp32_groups):
        raise RuntimeError('ZeRO master parameter partitions must be FP32.')
    # Keep exact resolved settings available to run manifests without relying on
    # private DeepSpeed config serialization or a separate caller-side copy.
    engine.loopwam_backend_config = config
    return engine


def deepspeed_precision_report(engine) -> dict:
    """Inspect storage after the first update without materializing full states.

    This checks dtype only. Numerical gradient/loss checks remain the caller's
    responsibility; ZeRO exposes the preceding pre-clip norm via
    engine.get_global_grad_norm() after engine.step().
    """
    policy = policy_parameters_fp32(engine.module)
    optimizer = engine.optimizer
    master = optimizer.single_partition_of_fp32_groups
    state = optimizer.optimizer.state
    moments = [value for values in state.values() for key, value in values.items()
               if key in ('exp_avg', 'exp_avg_sq')]
    if any(value.dtype != torch.float32 for value in [*master, *moments]):
        raise RuntimeError('AdamW master weights and moments must remain FP32.')
    return {'policy_dtype': 'float32', 'policy_parameters': sum(p.numel() for p in policy),
            'master_dtype': 'float32', 'local_master_elements': sum(p.numel() for p in master),
            'optimizer_moment_dtypes': sorted({str(value.dtype) for value in moments}),
            'optimizer_state_initialized': bool(moments),
            'compute_dtype': 'bfloat16', 'communication_dtype': 'float32'}
