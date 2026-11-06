#!/usr/bin/env python3
"""Original LoopWAM v0 4/4 on RoboCasa-GR1 (24 tasks): exact 256-window global batches, 5 epochs.

Port of scripts/train_loopwam_robotwin_full.py (RoboTwin v0 run, runtime de52d2f) with the same
model and training contract: 3/6/3, video/action loops 4/4, original LoopMoT, fresh Wan2.1 donor
init, FP32 weights and Adam moments with BF16 autocast, AdamW lr 1e-4 (all params) betas
(0.9, 0.95) eps 1e-8 weight decay 0.01, clip 1.0, linear warmup over 5% then cosine to 1% of the
peak, seed 42, every window once per epoch in randperm(seed + epoch) order, per-window keyed noise.
Only the global batch changes: 256 real windows per update (world x microbatch x accumulation).

Continuation (epochs 6-10, same design as the RoboTwin v0 epochs 6-10 posttrain): --continuation-from the completed
five-epoch latest.pt (update 117,580, optimizer included) with --expected-parent-sha256, --epochs 10, lr 3e-5 for all
params, no warmup, cosine over the 117,580 continuation updates to 1% (3e-7), saves epochs 6-10. Everything else must
equal the parent contract (checked); data order and noise continue as zero-based epochs 5-9 of the same recipe.
Later launches add --resume (out/latest.pt); --max-updates ending on an epoch boundary saves that epoch and exits.
"""
import argparse, contextlib, gc, hashlib, json, math, os, queue, random, subprocess, threading, time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from fastwam.datasets.robocasa_gr1 import (GR1Windows, gr1_spec, LATENT_SHAPE, OBS_TOKENS, ROBOT_DIM, VIDEO_HW,
                                           latent_provenance_gr1, stats_sha256)
from fastwam.datasets.loopwam_latent_cache import LoopWAMLatentCache
from fastwam.datasets.loopwam_robot_batches import epoch_rank_batches, training_budget
from fastwam.models.wan22.loopwam_gr1 import create_loopwam_gr1, RobotTrainForward, PARAMETERS, verify_optimizer_restore
from fastwam.models.wan22.loopwam_init import sha256_file
import fastwam.models.wan22.mot as mot_module
from fastwam.models.wan22.wan_video_dit import rope_apply, rope_apply_real

GLOBAL_BATCH = 256
WINDOWS, UPDATES_PER_EPOCH = 6020058, 23516          # robocasa_gr1 windows per epoch, ceil(WINDOWS / 256)
PARENT_UPDATES = 5 * UPDATES_PER_EPOCH                # 117,580: the completed five-epoch run
CONTINUATION_LR, CONTINUATION_SAVES = 3e-5, '6,7,8,9,10'


def continuation_contract(parent, sha):
    """The epochs 6-10 contract: the parent's, with only the continuation schedule fields changed."""
    if parent['epochs'] != 5 or parent['planned_updates'] != PARENT_UPDATES or parent['planned_windows'] != 5 * WINDOWS:
        raise ValueError('Continuation requires the complete five-epoch contract')
    return dict(parent, epochs=10, planned_updates=10 * UPDATES_PER_EPOCH, planned_windows=10 * WINDOWS,
                lr_backbone=CONTINUATION_LR, lr_robot_io=CONTINUATION_LR, warmup_fraction=0., save_epochs=CONTINUATION_SAVES,
                phase_start_update=PARENT_UPDATES, phase_start_epoch=5, parent_checkpoint_sha256=sha)


def continuation_factor(update):
    """Cosine from 1 to the 1% floor over the 117,580 continuation updates (3e-5 -> 3e-7), no warmup."""
    step = update - PARENT_UPDATES
    if not 0 <= step < PARENT_UPDATES:
        raise ValueError('Continuation LR step outside epochs 6-10')
    return .01 + .99 * .5 * (1 + math.cos(math.pi * step / (PARENT_UPDATES - 1)))


def validate_cursor(update, windows, epoch, batch, epochs):
    if not 0 <= epoch <= epochs or not 0 <= batch < UPDATES_PER_EPOCH or (epoch == epochs and batch):
        raise ValueError('Invalid epoch/batch cursor')
    if update != epoch * UPDATES_PER_EPOCH + batch or windows != epoch * WINDOWS + GLOBAL_BATCH * batch:
        raise ValueError('Invalid update/window cursor')


def validate_parent_state(payload):
    state = payload.get('training_state', {}); old = state.get('contract', {})
    if old.get('epochs') != 5 or old.get('planned_updates') != PARENT_UPDATES or old.get('planned_windows') != 5 * WINDOWS:
        raise ValueError('Parent is not a five-epoch run')
    if (payload.get('step'), state.get('update'), state.get('windows_seen'), state.get('next_epoch'), state.get('next_batch')) != \
            (PARENT_UPDATES, PARENT_UPDATES, 5 * WINDOWS, 5, 0):
        raise ValueError('Parent must be exactly the completed epoch-5 boundary')
    if not payload.get('optimizer', {}).get('state'):
        raise ValueError('Parent optimizer state missing')
    validate_cursor(state['update'], state['windows_seen'], 5, 0, 5)


def reconcile_metrics(path, saved_update, first_update=1):
    """On resume keep metrics up to the checkpoint; rolled-back records move to metrics.superseded.*.jsonl."""
    path = Path(path)
    if not path.exists():
        if saved_update >= first_update:
            raise ValueError('Missing pre-checkpoint metrics')
        return
    keep, extra = [], []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            extra.append(line); continue
        if row['update'] <= saved_update:
            keep.append(row)
        else:
            extra.append(line)
    if [r['update'] for r in keep] != list(range(first_update, saved_update + 1)):
        raise ValueError('Metrics are missing or duplicated before the resume checkpoint')
    if extra:
        path.with_name(f'metrics.superseded.{time.time_ns()}.jsonl').write_text('\n'.join(extra) + '\n')
        tmp = path.with_suffix('.jsonl.tmp'); tmp.write_text(''.join(json.dumps(x) + '\n' for x in keep)); tmp.replace(path)


def write(path, payload):
    path = Path(path); tmp = path.with_suffix('.json.tmp'); tmp.write_text(json.dumps(payload, indent=2)); tmp.replace(path)


def profile_summary(prof, wall_seconds, path):
    """Kernel-only GPU time by category (CPU ops also carry their kernels' time, so they are excluded) plus top CPU ops."""
    events = list(prof.key_averages())
    kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0]
    busy = sum(e.self_device_time_total for e in kernels); cats = {}
    for e in kernels:
        n = e.key.lower()
        c = ('attention' if any(k in n for k in ('flash', 'fmha', 'attention')) else 'nccl' if 'nccl' in n else
             'gemm' if any(k in n for k in ('gemm', 'cutlass', 'xmma', 'nvjet', 'sm90_', 'cublas')) else 'triton_fused' if 'triton' in n else
             'complex' if 'complex' in n else 'eager_elementwise' if 'elementwise' in n or 'reduce_kernel' in n else
             'optimizer' if any(k in n for k in ('adam', 'foreach', 'multi_tensor')) else 'copy' if any(k in n for k in ('copy', 'memcpy', 'memset')) else 'other')
        cats[c] = cats.get(c, 0) + e.self_device_time_total
    lines = [json.dumps(dict(wall_ms=round(1e3 * wall_seconds, 1), gpu_kernel_ms=round(busy / 1e3, 1),
                             categories_pct={k: round(100 * v / busy, 1) for k, v in sorted(cats.items(), key=lambda x: -x[1])}))]
    lines += [f'{100 * e.self_device_time_total / busy:6.2f}% {e.self_device_time_total / 1e3:9.1f}ms {e.count:6d} {e.key[:120]}'
              for e in sorted(kernels, key=lambda e: -e.self_device_time_total)[:30]]
    lines += ['--- top CPU ops by self CPU time'] + [f'{e.self_cpu_time_total / 1e3:9.1f}ms {e.count:6d} {e.key[:120]}'
              for e in sorted(events, key=lambda e: -e.self_cpu_time_total)[:25]]
    Path(path).write_text('\n'.join(lines) + '\n'); print(lines[0], flush=True)


def lr_factor(step, total, warmup):
    """RoboTwin v0 shape: linear warmup, cosine decay to 1% of the peak."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = min(1., (step - warmup) / max(total - warmup - 1, 1))
    return .01 + .99 * .5 * (1 + math.cos(math.pi * progress))


def slot_seed(seed, update, stream, w):
    digest = hashlib.sha256(f'{seed}:{update}:{stream}:{w}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1)


def slot_noise(model, seeds, device):
    """Noise and timesteps fixed by (seed, epoch, window): identical across runs and layouts."""
    nv, na, uv, ua = [], [], [], []
    for s in seeds:
        g = torch.Generator(device=device).manual_seed(s)
        nv.append(torch.randn(LATENT_SHAPE, generator=g, device=device))
        na.append(torch.randn((32, ROBOT_DIM), generator=g, device=device))
        u = torch.rand((2,), generator=g, device=device)
        uv.append(u[0]); ua.append(u[1])
    sv, sa = model.train_video_scheduler, model.train_action_scheduler
    tv = sv._phi(torch.stack(uv), sv.shift) * float(sv.num_train_timesteps)
    ta = sa._phi(torch.stack(ua), sa.shift) * float(sa.num_train_timesteps)
    return dict(video=torch.stack(nv), action=torch.stack(na), t_video=tv, t_action=ta)


class Prefetcher:
    def __init__(self, w, cache, rank, world, args, start_epoch=0, start_batch=0):
        self.q = queue.Queue(maxsize=3); self.error = None

        def worker():
            try:
                for epoch in range(start_epoch, args.epochs):
                    first = start_batch if epoch == start_epoch else 0
                    for batch, ids in enumerate(epoch_rank_batches(len(w), GLOBAL_BATCH, world, rank, args.seed + epoch, first), first):
                        n = min(GLOBAL_BATCH, len(w) - batch * GLOBAL_BATCH)
                        micros = []; built = time.perf_counter()
                        for lo in range(0, math.ceil(n / world), args.microbatch):
                            chosen = ids[lo:lo + args.microbatch]; dummy = not chosen; idx = chosen or [0]
                            w.warm_text([w.prompt(i) for i in idx]); f = [w.fields(i) for i in idx]
                            sample = {k: torch.stack([x[k] for x in f])
                                      for k in ('action', 'proprio', 'action_is_pad', 'image_is_pad', 'context', 'context_mask')}
                            if args.text_tokens < sample['context'].shape[1]:
                                # Exact: every text consumer is masked (per-token MLP, position-free masked cross-attention).
                                if sample['context_mask'][:, args.text_tokens:].any():
                                    raise ValueError(f'Prompt longer than --text-tokens {args.text_tokens}')
                                sample['context'] = sample['context'][:, :args.text_tokens].contiguous()
                                sample['context_mask'] = sample['context_mask'][:, :args.text_tokens].contiguous()
                            sample = {k: v.pin_memory() for k, v in sample.items()}
                            if not np.all(cache._valid[idx] == 1):
                                raise ValueError('Latent cache miss')
                            sample['latents_bits'] = torch.from_numpy(np.array(cache._latents[idx]).view(np.int16)).pin_memory()
                            micros.append((sample, idx, dummy))
                        self.q.put((epoch, batch, n, micros, time.perf_counter() - built))
                self.q.put(None)
            except BaseException as e:
                self.error = e; self.q.put(None)
        threading.Thread(target=worker, daemon=True).start()

    def get(self):
        item = self.q.get()
        if item is None and self.error:
            raise self.error
        return item


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='robocasa_gr1'); p.add_argument('--run-name', default='LoopWAM-v0-RoboCasa-GR1')
    p.add_argument('--output-dir', required=True); p.add_argument('--shared-dir', required=True)
    p.add_argument('--latent-dir', required=True, help='staged latent cache (node-local)')
    p.add_argument('--init-artifact', required=True); p.add_argument('--vae-path', default='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--memory-tokens', type=int, default=0); p.add_argument('--history-frame', type=int, default=0)
    p.add_argument('--epochs', type=int, default=5); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--global-batch', type=int, default=GLOBAL_BATCH); p.add_argument('--microbatch', type=int, default=8)
    p.add_argument('--lr-backbone', type=float, default=1e-4); p.add_argument('--lr-robot-io', type=float, default=1e-4)
    p.add_argument('--warmup-fraction', type=float, default=.05); p.add_argument('--save-epochs', default='3,4,5')
    p.add_argument('--checkpoint-every', type=int, default=1000); p.add_argument('--text-threads', type=int, default=4)
    p.add_argument('--checkpoint-blocks', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--keep-last', type=int, default=0, help='with --checkpoint-blocks: keep activations of the last N of 30 virtual layers')
    p.add_argument('--rope', choices=('complex', 'real'), default='complex',
                   help='real: the same float64 RoPE without complex tensors (bit-identical forward), fusable by inductor')
    p.add_argument('--text-tokens', type=int, default=128, help='drop masked text padding beyond this length (exact; batches '
                   'with longer prompts are refused)')
    p.add_argument('--manual-gc', type=int, default=0, help='N>0: freeze setup objects, disable automatic Python GC and collect '
                   'on every rank at the same update every N updates (no rank-at-a-time GC stalls in synchronous DDP)')
    p.add_argument('--profile-updates', default='', help='A:B -> rank-0 torch.profiler over updates A..B-1 (bounded no-save runs only)')
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--compile-mode', default='default', choices=('default', 'max-autotune-no-cudagraphs'))
    p.add_argument('--smoke-tail', action='store_true', help='One no-save epoch-tail update for DDP/compile validation only')
    p.add_argument('--max-updates', type=int); p.add_argument('--no-save', action='store_true'); p.add_argument('--resume', action='store_true')
    p.add_argument('--allow-partial-cache', action='store_true', help='bounded smokes only; every batch still requires valid latents')
    p.add_argument('--continuation-from', help='completed five-epoch latest.pt (update 117,580): starts the epochs 6-10 phase')
    p.add_argument('--expected-parent-sha256', help='SHA256 of the --continuation-from checkpoint')
    return p


def check_recipe(p, a):
    """Recipe/argument checks (argparse errors), shared by the trainer and the CPU dry run. Returns: continuation?"""
    cont = bool(a.continuation_from)
    if (a.dataset, a.epochs, a.seed, a.global_batch, a.memory_tokens, a.history_frame) != \
            ('robocasa_gr1', 10 if cont else 5, 42, GLOBAL_BATCH, 0, 0):
        p.error('GR1 v0 recipe requires robocasa_gr1, 5 epochs (10 with --continuation-from), seed 42, batch 256, memory0/history0')
    if not cont and (a.lr_backbone, a.lr_robot_io, a.warmup_fraction) != (1e-4, 1e-4, .05):
        p.error('GR1 v0 recipe requires lr 1e-4, warmup 5%')
    if cont and (a.lr_backbone, a.lr_robot_io, a.warmup_fraction, a.save_epochs, a.smoke_tail, a.allow_partial_cache) != \
            (CONTINUATION_LR, CONTINUATION_LR, 0., CONTINUATION_SAVES, False, False):
        p.error('Continuation requires lr 3e-5 (backbone and robot I/O), warmup 0, --save-epochs 6,7,8,9,10, full cache, no tail smoke')
    if cont != bool(a.expected_parent_sha256):
        p.error('--continuation-from and --expected-parent-sha256 go together')
    if a.keep_last and not a.checkpoint_blocks or not 0 <= a.keep_last <= 30:
        p.error('--keep-last requires --checkpoint-blocks and 0..30')
    if not 1 <= a.text_tokens <= 128:
        p.error('--text-tokens must be in 1..128')
    if a.profile_updates and not (a.no_save and a.max_updates):
        p.error('--profile-updates is for bounded no-save runs')
    if a.smoke_tail and (a.max_updates != 1 or not a.no_save or a.resume):
        p.error('Tail smoke requires max-updates1/no-save/fresh')
    return cont


def make_contract(a, w, m, hashes, cache, world, accumulation, per_epoch, total, tail):
    """Training contract stored in every checkpoint (shared by the trainer and the CPU dry run)."""
    contract = dict(model='original LoopWAM v0', partition=[3, 6, 3], video_loops=4, action_loops=4, memory_tokens=0, history_frame=0,
                    robot_dim=ROBOT_DIM, latent_shape=list(LATENT_SHAPE), observation_tokens=OBS_TOKENS,
                    init_sha256=hashes['init'], vae_sha256=hashes['vae'], normalization_sha256=stats_sha256(w.stats),
                    normalization_mode='min/max', latent_cache_provenance_sha256=cache.provenance_sha256,
                    data_manifest_sha256=hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest(),
                    epochs=a.epochs, seed=a.seed, global_batch=GLOBAL_BATCH, world=world, microbatch=a.microbatch, accumulation=accumulation,
                    shuffle='randperm(seed+zero_based_epoch)', windows_per_epoch=len(w), planned_windows=len(w) * a.epochs,
                    planned_updates=total, updates_per_epoch=per_epoch, last_batch_windows=tail,
                    lr_backbone=a.lr_backbone, lr_robot_io=a.lr_robot_io, betas=[.9, .95], eps=1e-8, weight_decay=.01, clip=1.,
                    warmup_fraction=a.warmup_fraction, cosine_floor=.01,
                    precision='fp32 weights/moments, bf16 compute', save_epochs=a.save_epochs, checkpoint_every=a.checkpoint_every,
                    kernels=dict(rope=a.rope, attention='structured', compile=a.compile, compile_mode=a.compile_mode,
                                 checkpoint_blocks=a.checkpoint_blocks, keep_last=a.keep_last,
                                 text_tokens=a.text_tokens),
                    smoke_tail=a.smoke_tail)
    if a.continuation_from:
        contract.update(phase_start_update=PARENT_UPDATES, phase_start_epoch=5, parent_checkpoint_sha256=hashes['parent'])
    return contract


def main():
    p = build_parser()
    a = p.parse_args()
    cont = check_recipe(p, a)
    rank, world, local = [int(os.getenv(k, d)) for k, d in [('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)]]
    if world not in (4, 8) or GLOBAL_BATCH % world:
        p.error('This recipe requires 4 or 8 GPUs')
    per_rank = GLOBAL_BATCH // world
    if a.microbatch < 1 or per_rank % a.microbatch:
        p.error(f'microbatch must divide the {per_rank} windows per rank')
    accumulation = per_rank // a.microbatch
    os.environ['TRITON_CACHE_DIR'] = f'/tmp/gr1_v0_{os.getenv("SLURM_JOB_ID", "local")}_rank{rank}'
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.environ['TRITON_CACHE_DIR'] + '_inductor'
    torch.cuda.set_device(local); device = torch.device('cuda', local); torch.set_num_threads(2)
    from datetime import timedelta
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=60))
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True); shared = Path(a.shared_dir)
    resume = out / 'latest.pt'
    if not a.resume and any((out / n).exists() for n in ('manifest.json', 'metrics.jsonl', 'latest.pt', 'COMPLETE')):
        raise ValueError('Fresh run refuses existing records')
    if a.resume and not resume.exists():
        raise ValueError('Missing resume checkpoint')
    w = GR1Windows(gr1_spec(stats_path=shared / 'data/dataset_stats_29d.json'), cache_dir=shared / 'data', text_threads=a.text_threads)
    m = w.manifest(); assert len(w.groups) == 24 and len(w.episodes) == 24000
    old = json.loads((shared / 'data/data_manifest.json').read_text()); assert old == m, 'Data manifest changed'
    hashes = [dict(init=sha256_file(a.init_artifact), vae=sha256_file(a.vae_path),
                   **({'parent': sha256_file(a.continuation_from)} if cont else {})) if rank == 0 else None]
    dist.broadcast_object_list(hashes, src=0); hashes = hashes[0]
    if cont and hashes['parent'] != a.expected_parent_sha256:
        raise ValueError('Parent checkpoint SHA256 differs from --expected-parent-sha256')
    cache = LoopWAMLatentCache(Path(a.latent_dir), len(w), latent_provenance_gr1(w, hashes['vae']), latent_shape=LATENT_SHAPE)
    if a.allow_partial_cache:
        if not a.max_updates:
            p.error('--allow-partial-cache is only for bounded smokes')
    else:
        assert np.all(cache._valid == 1), 'Incomplete latent cache'
    per_epoch, total, tail = training_budget(len(w), GLOBAL_BATCH, a.epochs)
    assert len(w) == WINDOWS and per_epoch == UPDATES_PER_EPOCH and total == UPDATES_PER_EPOCH * a.epochs, 'Unexpected data size'
    contract = make_contract(a, w, m, hashes, cache, world, accumulation, per_epoch, total, tail)
    mot_module.rope_apply = rope_apply_real if a.rope == 'real' else rope_apply
    model, payload = create_loopwam_gr1(a.vae_path, init_artifact=a.init_artifact,
                                        checkpoint_path=resume if a.resume else a.continuation_from if cont else None,
                                        device=device, checkpoint_blocks=a.checkpoint_blocks, seed=a.seed, artifact_sha256=hashes['init'])
    assert model.architecture_metadata['init_artifact_sha256'] == hashes['init']
    model.train(); model.mot.structured_attention = True; model.mot.structured_attention_observation_tokens = OBS_TOKENS
    model.mot.uncheckpointed_virtual_layers = a.keep_last
    if a.compile:
        torch._dynamo.config.cache_size_limit = 256; torch._dynamo.config.accumulated_cache_size_limit = 8192
        torch._inductor.config.comprehensive_padding = False
        for name in ('_joint_block', '_video_only_block'):
            setattr(model.mot, name, torch.compile(getattr(model.mot, name), mode=a.compile_mode))
    params = model.policy_parameters(); assert sum(x.numel() for x in params) == PARAMETERS
    io_ids = {id(p) for layer in (model.action_expert.action_encoder, model.action_expert.head, model.proprio_encoder) for p in layer.parameters()}
    groups = [dict(params=[p for p in params if id(p) not in io_ids], lr=a.lr_backbone), dict(params=[p for p in params if id(p) in io_ids], lr=a.lr_robot_io)]
    opt = torch.optim.AdamW(groups, betas=(.9, .95), eps=1e-8, weight_decay=.01, fused=True)
    runner = DDP(RobotTrainForward(model), device_ids=[local], broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True)
    update = windows_seen = start_epoch = start_batch = 0; elapsed_prior = 0.
    if a.resume or cont:
        state = payload['training_state']
        if a.resume:
            assert state['contract'] == contract, 'Resume contract mismatch'
        else:
            validate_parent_state(payload)
            assert continuation_contract(state['contract'], hashes['parent']) == contract, 'Continuation contract mismatch'
        opt.load_state_dict(payload['optimizer']); verify_optimizer_restore(opt, payload['optimizer'])
        update = state['update']; windows_seen = state['windows_seen']
        start_epoch = state['next_epoch']; start_batch = state['next_batch']; elapsed_prior = state['elapsed_training_seconds'] if a.resume else 0.
        validate_cursor(update, windows_seen, start_epoch, start_batch, a.epochs)
        if update >= total:
            raise ValueError('Nothing left to train: the checkpoint is at the end of the planned updates')
        resume_ids = next(epoch_rank_batches(len(w), GLOBAL_BATCH, world, rank, a.seed + start_epoch, start_batch), [])
        next_factor = continuation_factor(update) if cont else lr_factor(update, total, int(total * a.warmup_fraction))
        if rank == 0:
            if a.resume:
                reconcile_metrics(out / 'metrics.jsonl', update, first_update=PARENT_UPDATES + 1 if cont else 1)
            write(out / f'resume_audit_u{update}.json', dict(passed=True, optimizer_exact=True, next_epoch=start_epoch, next_batch=start_batch,
                  next_rank0_window_ids=resume_ids, next_rank0_noise_seeds=[slot_seed(a.seed, start_epoch, i, 0) for i in resume_ids],
                  next_lr=a.lr_backbone * next_factor, continuation=cont, parent_checkpoint_sha256=hashes.get('parent')))
    del payload
    launch_update = update
    if a.max_updates is not None and not launch_update < a.max_updates <= total:
        raise ValueError(f'--max-updates {a.max_updates} must be in ({launch_update}, {total}]')
    if a.smoke_tail:
        start_batch = per_epoch - 1
    if rank == 0:
        write(out / ('manifest.json' if not a.resume else f'resume_manifest_u{update}.json'), dict(vars(a), contract=contract, world_size=world,
              policy_parameters=PARAMETERS, memory_parameters=0, resumed_from_update=update if a.resume or cont else None,
              git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), slurm_job_id=os.getenv('SLURM_JOB_ID'),
              cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'), latent_dir=str(Path(a.latent_dir).resolve())))
        (out / 'data').mkdir(exist_ok=True); write(out / 'data/data_manifest.json', m); write(out / 'data/dataset_stats.json', w.stats)
    pref = Prefetcher(w, cache, rank, world, a, start_epoch, start_batch); log = (out / 'metrics.jsonl').open('a') if rank == 0 else None
    start = time.time(); warmup = int(total * a.warmup_fraction); last_save = update; next_epoch, next_batch = start_epoch, start_batch

    def save(epoch=None):
        if rank == 0:
            state = dict(contract=contract, update=update, windows_seen=windows_seen, next_epoch=next_epoch, next_batch=next_batch,
                         elapsed_training_seconds=elapsed_prior + time.time() - start)
            if epoch is None:
                model.save_robot(out / 'latest.pt', optimizer=opt, step=update, training_state=state)
            else:
                (out / 'checkpoints').mkdir(exist_ok=True); model.save_robot(out / 'checkpoints' / f'epoch_{epoch:02d}.pt', step=update, training_state=state)
        dist.barrier()
    gc_seconds = [0., 0]

    def gc_timer(phase, info):  # GC holds the GIL: any collection (main or prefetch thread) stalls this rank's launches
        if phase == 'start':
            gc_seconds[1] = time.perf_counter()
        else:
            gc_seconds[0] += time.perf_counter() - gc_seconds[1]
    gc.callbacks.append(gc_timer)
    if a.manual_gc:
        gc.collect(); gc.freeze(); gc.disable()
    status = 'running'; prof = None
    prof_range = [int(x) for x in a.profile_updates.split(':')] if a.profile_updates else None
    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    while True:
        t = time.time(); item = pref.get(); data_wait = time.time() - t
        if item is None:
            status = 'complete'; break
        epoch, batch, n, micros, build_seconds = item
        if a.manual_gc and update and update % a.manual_gc == 0:
            gc.collect()
        if rank == 0 and prof_range and update == prof_range[0]:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]); prof.__enter__()
            prof_start = time.time()
        events[0].record()
        if (a.resume or cont) and epoch == start_epoch and batch == start_batch:
            assert [i for _, ids, dummy in micros if not dummy for i in ids] == resume_ids, 'Resumed data differs from expected next batch'
        factor = continuation_factor(update) if cont else lr_factor(update, total, warmup)
        for group, lr in zip(opt.param_groups, [a.lr_backbone, a.lr_robot_io]):
            group['lr'] = lr * factor
        sums = torch.zeros(4, device=device)
        for mi, (cpu, ids, dummy) in enumerate(micros):
            sample = {k: v.to(device, non_blocking=True) for k, v in cpu.items() if k != 'latents_bits'}
            sample['latents'] = cpu['latents_bits'].to(device, non_blocking=True).view(torch.bfloat16).float()
            sample['video'] = torch.zeros((), device=device).expand(len(ids), 3, 9, *VIDEO_HW)
            noise = slot_noise(model, [slot_seed(a.seed, epoch, i, 0) for i in ids], device)
            ctx = runner.no_sync() if mi < len(micros) - 1 else contextlib.nullcontext()
            with ctx:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, logs, _ = runner([dict(sample=sample, noise=noise)])
                (loss * (0. if dummy else world / n)).backward()
            if mi == 0:
                events[3].record()  # first microstep: pure compute (no_sync), comparable across ranks
            if not dummy:
                sums += torch.stack([loss.detach(), logs['video'], logs['action'], torch.as_tensor(len(ids), device=device)])
        events[1].record(); issue_seconds = time.time() - t - data_wait
        if update == launch_update and any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise ValueError('Missing/nonfinite gradient')
        grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); opt.step(); opt.zero_grad(set_to_none=True)
        events[2].record()
        if update == launch_update:
            moments = sorted({str(v.dtype) for state in opt.state.values() for k, v in state.items() if k in ('exp_avg', 'exp_avg_sq')})
            assert moments == ['torch.float32'] and all(p.dtype == torch.float32 for p in params)
            if rank == 0:
                write(out / 'precision.json', dict(policy='float32', compute='bfloat16', optimizer_moment_dtypes=moments, all_gradients_present_and_finite=True))
        dist.all_reduce(sums); assert int(sums[3]) == n
        torch.cuda.synchronize(); seconds = time.time() - t; update += 1; windows_seen += n
        diag = torch.tensor([events[0].elapsed_time(events[3]) / 1e3, issue_seconds, build_seconds, gc_seconds[0]], device=device)
        gc_seconds[0] = 0.; ranks = torch.empty(world, 4, device=device); dist.all_gather_into_tensor(ranks, diag); ranks = ranks.tolist()
        done = (batch + 1 == per_epoch); next_epoch, next_batch = (epoch + 1, 0) if done else (epoch, batch + 1)
        if rank == 0:
            rec = dict(update=update, epoch=epoch + 1, epochs_complete=epoch + int(done), valid_windows=n, windows_seen=windows_seen, seconds=seconds,
                       data_wait_seconds=data_wait, prefetch_build_seconds=build_seconds, host_issue_seconds=issue_seconds,
                       gpu_micro_seconds=events[0].elapsed_time(events[1]) / 1e3, gpu_optimizer_seconds=events[1].elapsed_time(events[2]) / 1e3,
                       rank_first_micro_seconds=[round(x[0], 4) for x in ranks], rank_host_issue_seconds=[round(x[1], 4) for x in ranks],
                       rank_prefetch_build_seconds=[round(x[2], 4) for x in ranks], rank_gc_seconds=[round(x[3], 4) for x in ranks],
                       loss=float(sums[0] / n), loss_video=float(sums[1] / n), loss_action=float(sums[2] / n),
                       grad_norm=float(grad), lr_backbone=opt.param_groups[0]['lr'],
                       peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9, peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9)
            log.write(json.dumps(rec) + '\n'); log.flush()
            if update <= 5 or update % 10 == 0:
                print(json.dumps(rec), flush=True)
            write(out / 'timing.json', dict(status='running', update=update, planned_updates=total, windows_seen=windows_seen,
                  elapsed_training_seconds=elapsed_prior + time.time() - start, last_update_seconds=seconds, heartbeat_unix=time.time()))
        if prof is not None and update == prof_range[1]:
            prof.__exit__(None, None, None)
            profile_summary(prof, (time.time() - prof_start) / (prof_range[1] - prof_range[0]), out / f'profile_rank0_u{prof_range[0]}_{prof_range[1]}.txt'); prof = None
        if not a.no_save:
            if done and epoch + 1 in {int(v) for v in a.save_epochs.split(',')}:
                save(epoch + 1)
            if update - last_save >= a.checkpoint_every or done:
                save(); last_save = update
        if a.max_updates and update >= a.max_updates:
            if not a.no_save and last_save != update:
                save()
            status = 'max_updates'; break
    if status == 'complete':
        assert update == total and windows_seen == len(w) * a.epochs and next_epoch == a.epochs
    if rank == 0:
        result = dict(status=status, update=update, planned_updates=total, windows_seen=windows_seen, elapsed_training_seconds=elapsed_prior + time.time() - start)
        write(out / 'timing.json', result)
        if status == 'complete':
            write(out / 'COMPLETE', result)
        log.close(); print(json.dumps(result), flush=True)
    dist.barrier(); dist.destroy_process_group(); os._exit(0)


if __name__ == '__main__':
    main()
