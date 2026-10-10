#!/usr/bin/env python3
"""ChronoLoop fine-tuning from the full-suite v0 4/4 parent on all four LIBERO suites.

torchrun --standalone --nproc_per_node=4 scripts/train_chronoloop.py --run-name CL-A \
    --memory-tokens 16 --mem-source learned --mem-write loop --action-loops 4 --output-dir DIR

Every run reads the same traversal schedule (plan 3.2): 32 streams x 4 consecutive windows
= 128 window slots per update (global batch 128). Rank r owns streams r, r+R, ...; the
schedule, the noise of every window slot, and the stream state are world-size independent,
so a run may resume on a different GPU count. Loss = parent v0 flow-matching loss averaged
over the valid windows of the update. FP32 master weights and AdamW moments, BF16 autocast.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import threading
import queue
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from fastwam.datasets.chronoloop_data import (FullLiberoWindows, StreamScheduler, load_schedule, read_latents,
                                              write_schedule)
from fastwam.datasets.loopwam_latent_cache import LoopWAMLatentCache, latent_cache_provenance
from fastwam.datasets.loopwam_long import FULL_LIBERO_SUITES, build_full_libero_datasets
from fastwam.models.wan22.chronoloop import ChronoConfig, create_chronoloop
from fastwam.models.wan22.loopwam_init import sha256_file

PARENT_PARAMETERS = 584536135


def lr_factor(step, total, warmup):
    """Parent shape: linear warmup, cosine decay to 1% of the peak."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = min(1., (step - warmup) / max(total - warmup - 1, 1))
    return .01 + .99 * .5 * (1 + math.cos(math.pi * progress))


def slot_seed(seed, update, stream, w):
    digest = hashlib.sha256(f'{seed}:{update}:{stream}:{w}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1)


def slot_noise(model, seeds, device):
    """Noise and timesteps fixed by (seed, update, stream, window): identical across runs and layouts."""
    nv, na, uv, ua = [], [], [], []
    for s in seeds:
        g = torch.Generator(device=device).manual_seed(s)
        nv.append(torch.randn((16, 3, 28, 56), generator=g, device=device))
        na.append(torch.randn((32, 7), generator=g, device=device))
        u = torch.rand((2,), generator=g, device=device)
        uv.append(u[0]); ua.append(u[1])
    sv, sa = model.train_video_scheduler, model.train_action_scheduler
    tv = sv._phi(torch.stack(uv), sv.shift) * float(sv.num_train_timesteps)
    ta = sa._phi(torch.stack(ua), sa.shift) * float(sa.num_train_timesteps)
    return dict(video=torch.stack(nv), action=torch.stack(na), t_video=tv, t_action=ta)


class Prefetcher:
    """Background thread: advances the scheduler and assembles CPU window batches in order."""

    def __init__(self, scheduler, windows, cache, my_streams, stream_micro, history, depth=3):
        self.scheduler, self.windows, self.cache = scheduler, windows, cache
        self.my_streams, self.stream_micro, self.history = my_streams, stream_micro, history
        self.q = queue.Queue(maxsize=depth)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.error = None
        self.thread.start()

    def _window_rows(self, slot_list):
        idx = [x[0] for x in slot_list]
        fields = [self.windows.fields(i) for i in idx]
        batch = {k: torch.stack([f[k] for f in fields]) for k in
                 ('action', 'proprio', 'action_is_pad', 'image_is_pad', 'proprio_is_pad', 'context', 'context_mask')}
        if not (self.cache._valid[np.asarray(idx)] == 1).all():
            raise RuntimeError('Latent cache miss')
        bits = np.ascontiguousarray(self.cache._latents[np.asarray(idx)]).view(np.int16)
        batch['latents_bits'] = torch.from_numpy(bits).pin_memory()
        if self.history:
            # Query k-3 (30 steps back); the traversal's first query is repeated when k < 3.
            hidx = [i - 10 * min(q, 3) for i, _, q in slot_list]
            hb = np.ascontiguousarray(self.cache._latents[np.asarray(hidx)][:, :, :1]).view(np.int16)
            batch['history_bits'] = torch.from_numpy(hb).pin_memory()
        batch['training_index'] = torch.as_tensor(idx)
        return batch

    def _run(self):
        try:
            while not self.scheduler.exhausted:
                slots, resets = self.scheduler.next_update()
                state = self.scheduler.state_dict()
                valid_global = sum(x is not None for row in slots for x in row)
                suites = [0] * 4
                for row in slots:
                    for x in row:
                        if x is not None:
                            d = int(self.windows.episode_of[x[0]])
                            suites[FULL_LIBERO_SUITES.index(self.windows.episodes[d][0])] += 1
                micros = []
                for lo in range(0, len(self.my_streams), self.stream_micro):
                    streams = self.my_streams[lo:lo + self.stream_micro]
                    windows = []
                    for w in range(len(slots[0])):
                        rows = [j for j, k in enumerate(streams) if slots[k][w] is not None]
                        if not rows:
                            continue
                        sl = [slots[streams[j]][w] for j in rows]
                        windows.append(dict(rows=rows, streams=[streams[j] for j in rows], w=w,
                                            batch=self._window_rows(sl), slots=sl))
                    micros.append(dict(streams=streams, resets=[resets[k] for k in streams], windows=windows))
                self.q.put(dict(micros=micros, valid_global=valid_global, suites=suites, state=state))
            self.q.put(None)
        except Exception as exc:  # surfaced in the main thread
            self.error = exc
            self.q.put(None)

    def get(self):
        item = self.q.get()
        if item is None and self.error is not None:
            raise self.error
        return item


def to_device_sample(batch, device):
    sample = {k: batch[k].to(device, non_blocking=True) for k in
              ('action', 'proprio', 'action_is_pad', 'image_is_pad', 'context', 'context_mask')}
    sample['latents'] = batch['latents_bits'].to(device, non_blocking=True).view(torch.bfloat16).float()
    b = sample['latents'].shape[0]
    sample['video'] = torch.zeros((), device=device).expand(b, 3, 9, 224, 448)
    sample['training_index'] = batch['training_index']
    if 'history_bits' in batch:
        sample['history_latents'] = batch['history_bits'].to(device, non_blocking=True).view(torch.bfloat16).float()
    return sample


def concat_samples(samples):
    out = {}
    for key in samples[0]:
        if key == 'video':
            out[key] = torch.zeros((), device=samples[0][key].device).expand(
                sum(s[key].shape[0] for s in samples), *samples[0][key].shape[1:])
        else:
            out[key] = torch.cat([s[key] for s in samples])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--memory-tokens', type=int, default=0)
    p.add_argument('--mem-source', default='none', choices=['none', 'learned', 'reset', 'oracle'])
    p.add_argument('--mem-write', default='none', choices=['none', 'loop', 'external'])
    p.add_argument('--action-loops', type=int, default=4)
    p.add_argument('--history-frame', type=int, default=0)
    p.add_argument('--init', default=os.environ.get('CHRONO_PARENT'))
    p.add_argument('--shared-dir', default=os.environ.get('CHRONO_SHARED'))
    p.add_argument('--dataset-dir', default='data/lerobot_v30')
    p.add_argument('--text-cache-dir', default='data/text_embeds_cache/libero')
    p.add_argument('--vae-path', default='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--streams', type=int, default=32)
    p.add_argument('--tbptt', type=int, default=4)
    p.add_argument('--global-batch', type=int, default=128)
    p.add_argument('--stream-micro', type=int, default=8, help='streams per micro-batch on a rank')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr-backbone', type=float, default=3e-5)
    p.add_argument('--lr-memory', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup-fraction', type=float, default=0.03)
    p.add_argument('--save-epochs', default='8,9,10')
    p.add_argument('--save-updates', default='2000')
    p.add_argument('--checkpoint-every', type=int, default=500)
    p.add_argument('--checkpoint-blocks', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True,
                   help='torch.compile each transformer-block function (same math; ~1.8x faster)')
    p.add_argument('--max-updates', type=int, default=None, help='smoke/benchmark only')
    p.add_argument('--no-save', action='store_true', help='benchmark only')
    p.add_argument('--time-limit-hours', type=float, default=None,
                   help='checkpoint and exit cleanly before this wall time (allocation end)')
    args = p.parse_args()
    if args.streams * args.tbptt != args.global_batch:
        p.error('global batch must equal streams x tbptt (plan: 32 x 4 = 128)')
    cfg = ChronoConfig(memory_tokens=args.memory_tokens, mem_source=args.mem_source, mem_write=args.mem_write,
                       history_frame=args.history_frame, action_loops=args.action_loops)
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    if args.streams % world:
        raise ValueError('streams must divide evenly across ranks (each rank owns whole streams)')
    wall_start = time.time()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    shared = Path(args.shared_dir)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(2)

    # ---------------------------------------------------------------- data
    train, _, manifest = build_full_libero_datasets(args.dataset_dir, args.text_cache_dir, str(shared / 'data'))
    windows = FullLiberoWindows(train)
    if rank == 0:
        windows.verify_against_reference(np.linspace(0, len(windows) - 1, 24).astype(int))
    schedule_path = shared / f'schedule_e{args.epochs}_s{args.seed}_n{args.streams}_t{args.tbptt}.json'
    if rank == 0 and not schedule_path.exists():
        write_schedule(schedule_path, windows, args.epochs, args.seed, args.streams, args.tbptt)
    if world > 1: dist.barrier()
    schedule = load_schedule(schedule_path)
    total = schedule['summary']['updates']
    epoch_done_at = {int(k): v for k, v in schedule['summary']['epoch_complete_update'].items()}
    vae_sha = [sha256_file(args.vae_path) if rank == 0 else None]
    if world > 1: dist.broadcast_object_list(vae_sha, src=0)
    cache = LoopWAMLatentCache(shared / 'latents', len(train), latent_cache_provenance(manifest, vae_sha[0]))
    if rank == 0 and int(np.count_nonzero(np.asarray(cache._valid) == 1)) != len(train):
        raise RuntimeError('Latent cache incomplete; run chronoloop_precompute_latents.py first')

    # ---------------------------------------------------------------- model
    resume = out / 'latest.pt'
    resume = resume if resume.exists() else None
    parent_sha = [None]
    if rank == 0 and not resume:
        parent_sha[0] = sha256_file(args.init)
    if world > 1: dist.broadcast_object_list(parent_sha, src=0)
    model, payload = create_chronoloop(cfg, args.vae_path, parent_path=args.init, checkpoint_path=resume,
                                       device=device, checkpoint_blocks=args.checkpoint_blocks,
                                       parent_sha256=parent_sha[0])
    model.train()
    if args.compile:
        torch._dynamo.config.cache_size_limit = 256
        torch._dynamo.config.accumulated_cache_size_limit = 8192
        # Batch size can change inside one TBPTT graph (traversals end mid-segment); padded
        # strides then break checkpoint recomputation of the dynamic-shape graph (inductor 2.7).
        torch._inductor.config.comprehensive_padding = False
        for name in ('_joint_block', '_video_only_block', '_chrono_block'):
            # Bound-method attributes: partial(self.<name>, i) picks up the compiled version.
            setattr(model.mot, name, torch.compile(getattr(model.mot, name)))
    model.mot.structured_attention = True
    model.mot.structured_attention_observation_tokens = 392
    if cfg.memory_tokens and not resume:
        # gamma / e from obs tokens after prepare on 32 fixed windows (identical on every rank).
        idx = np.linspace(0, len(windows) - 1, 32).astype(int)
        lat = read_latents(cache, idx, device)
        with torch.no_grad():
            x = model.video_expert.patchify(lat[:, :, :1])
            model.memory.initialize_from_obs(x.flatten(2).transpose(1, 2), seed=args.seed)
    count_backbone = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and '.memory.' not in f'.{n}')
    count_memory = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and '.memory.' in f'.{n}')
    if count_backbone != PARENT_PARAMETERS:
        raise AssertionError(f'Unexpected backbone parameter count {count_backbone}')
    mem_params = [p for n, p in model.named_parameters() if p.requires_grad and '.memory.' in f'.{n}']
    bb_params = [p for n, p in model.named_parameters() if p.requires_grad and '.memory.' not in f'.{n}']
    groups = [dict(params=bb_params, lr=args.lr_backbone, weight_decay=args.weight_decay, name='backbone')]
    if mem_params:
        groups.append(dict(params=mem_params, lr=args.lr_memory, weight_decay=0.0, name='memory'))
    opt = torch.optim.AdamW(groups, betas=(.9, .95), eps=1e-8, fused=True)
    runner = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False,
                 gradient_as_bucket_view=True) if world > 1 else model
    all_params = bb_params + mem_params

    # ---------------------------------------------------------------- state
    contract = dict(run=args.run_name, chrono=asdict(cfg), schedule_sha256=schedule['sha256'], streams=args.streams,
                    tbptt=args.tbptt, global_batch=args.global_batch, epochs=args.epochs, seed=args.seed,
                    lr_backbone=args.lr_backbone, lr_memory=args.lr_memory, weight_decay=args.weight_decay,
                    warmup_fraction=args.warmup_fraction, planned_updates=total,
                    normalization_sha256=manifest['normalization_sha256'])
    scheduler = StreamScheduler(schedule['traversals'], schedule['episode_starts'], schedule['episode_lengths'],
                                args.streams, args.tbptt)
    my_streams = list(range(rank, args.streams, world))
    stream_state = {}                       # stream id -> detached s (GPU, fp32)
    update, windows_seen = 0, 0
    elapsed_prior = 0.
    if resume:
        ts = payload['training_state']
        if ts['contract'] != contract:
            raise ValueError(f'Resume contract changed: {ts["contract"]} vs {contract}')
        opt.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(ts['scheduler'])
        update, windows_seen, elapsed_prior = ts['update'], ts['windows_seen'], ts.get('elapsed_training_seconds', 0.)
        stream_state = {k: v.to(device) for k, v in ts['stream_state'].items() if k in my_streams}
        if rank == 0:
            print(json.dumps(dict(event='resumed', update=update, from_world=ts.get('world'), world=world)), flush=True)
    del payload
    warmup = int(total * args.warmup_fraction)
    sequential = cfg.carries_state
    save_epochs = {int(x) for x in args.save_epochs.split(',') if x}
    save_updates = {int(x) for x in args.save_updates.split(',') if x}
    snapshot = Path('REVISION').exists()   # frozen code snapshot made by chronoloop_submit.sh
    git_rev = Path('REVISION').read_text().strip() if snapshot else \
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    git_dirty = False if snapshot else bool(subprocess.check_output(
        ['git', 'status', '--porcelain', '--', 'src', 'scripts'], text=True).strip())
    if rank == 0:
        manifest_out = dict(vars(args), contract=contract, chrono_flags_sha256=cfg.flags_hash(), world_size=world,
            streams_per_rank=len(my_streams), sequential_tbptt=sequential, backbone_parameters=count_backbone,
            memory_parameters=count_memory, parent_sha256=model.parent_sha256, schedule_file=str(schedule_path),
            schedule_summary=schedule['summary'], suite_windows_per_epoch=schedule['suite_windows_per_epoch'],
            git_revision=git_rev, git_dirty=git_dirty, code_snapshot=snapshot, code_dir=os.getcwd(),
            host=socket.gethostname(), slurm_job_id=os.getenv('SLURM_JOB_ID'), torch=torch.__version__,
            gpu=torch.cuda.get_device_name(), resumed_from_update=update if resume else None)
        (out / ('manifest.json' if not resume else f'resume_manifest_u{update}.json')).write_text(json.dumps(manifest_out, indent=2, default=str))
        print(json.dumps(dict(event='ready', run=args.run_name, planned_updates=total, update=update, world=world,
                              memory_parameters=count_memory, sequential=sequential)), flush=True)

    def gather_stream_state():
        local_state = {k: v.detach().cpu() for k, v in stream_state.items()}
        if world == 1:
            return local_state
        parts = [None] * world
        dist.all_gather_object(parts, local_state)
        merged = {}
        for part in parts: merged.update(part)
        return merged

    def save(kind, name=None):
        merged = gather_stream_state()
        if rank == 0:
            state = dict(contract=contract, scheduler=scheduler_state, update=update, windows_seen=windows_seen,
                         stream_state=merged, world=world, elapsed_training_seconds=elapsed_prior + (time.time() - train_start),
                         rng=dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state()))
            t0 = time.time()
            if kind == 'resume':
                model.save_chrono(out / 'latest.pt', optimizer=opt, step=update, training_state=state)
            else:
                (out / 'checkpoints').mkdir(exist_ok=True)
                model.save_chrono(out / 'checkpoints' / name, step=update,
                                  training_state=dict(update=update, windows_seen=windows_seen, contract=contract),
                                  weights_only=True)
            log.write(json.dumps(dict(event='checkpoint', kind=kind, name=name or 'latest.pt', update=update,
                                      seconds=time.time() - t0)) + '\n'); log.flush()
        if world > 1: dist.barrier()

    log = (out / 'metrics.jsonl').open('a') if rank == 0 else None
    prefetch = Prefetcher(scheduler, windows, cache, my_streams, args.stream_micro, cfg.history_frame > 0)
    scheduler_state = scheduler.state_dict()
    train_start = time.time()
    last_save = start_update = update
    epoch_sizes = np.bincount(schedule['traversals'][:, 0])

    def epochs_complete(state):
        done = 0
        for e, size in enumerate(epoch_sizes):
            if state['finished_per_epoch'].get(e, 0) != size:
                break
            done = e + 1
        return done
    stop_reason = None
    while True:
        item = prefetch.get()
        if item is None:
            stop_reason = 'complete'
            break
        t0 = time.time()
        valid_global = item['valid_global']
        for group in opt.param_groups:
            base = args.lr_memory if group['name'] == 'memory' else args.lr_backbone
            group['lr'] = base * lr_factor(update, total, warmup)
        sums = {}
        local_valid = 0
        micros = item['micros']
        for mi, micro in enumerate(micros):
            last = mi == len(micros) - 1
            entries = []
            for win in micro['windows']:
                sample = to_device_sample(win['batch'], device)
                seeds = [slot_seed(args.seed, update, k, win['w']) for k in win['streams']]
                entries.append(dict(sample=sample, noise=slot_noise(model, seeds, device),
                                    rows=torch.as_tensor(win['rows'], device=device)))
                local_valid += len(win['rows'])
            dummy = not entries
            if dummy:
                # This rank has no valid window in this update: zero-weight forward keeps DDP in step.
                win_slots = [(0, -1, 0)]
                b = Prefetcher._window_rows(prefetch, win_slots)
                entries = [dict(sample=to_device_sample(b, device), noise=slot_noise(model, [0], device),
                                rows=torch.zeros(1, dtype=torch.long, device=device))]
            if sequential:
                s0 = torch.stack([torch.zeros(cfg.memory_tokens, 1536, device=device) if (r or k not in stream_state)
                                  else stream_state[k] for k, r in zip(micro['streams'], micro['resets'])]) if not dummy else \
                    torch.zeros(1, cfg.memory_tokens, 1536, device=device)
                segment = entries
            else:
                s0 = None
                segment = [dict(sample=concat_samples([e['sample'] for e in entries]),
                                noise={k: torch.cat([e['noise'][k] for e in entries]) for k in entries[0]['noise']},
                                rows=None)]
            ctx = runner.no_sync() if world > 1 and not last else contextlib.nullcontext()
            with ctx:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss_sum, logs, s_final = runner(segment, s0, sequential)
                scale = 0. if dummy else world / valid_global
                (loss_sum * scale).backward()
            if sequential and not dummy:
                for j, k in enumerate(micro['streams']):
                    stream_state[k] = s_final[j]
            if not dummy:
                sums['loss'] = sums.get('loss', 0.) + loss_sum.detach().float()
                for key, value in logs.items():
                    sums[key] = sums.get(key, 0.) + value
        grad = torch.nn.utils.clip_grad_norm_(all_params, 1.0, error_if_nonfinite=True)
        opt.step(); opt.zero_grad(set_to_none=True)
        update += 1
        windows_seen += valid_global
        scheduler_state = item['state']
        # Streams that finished their traversal start fresh next time (reset flag), so stale
        # state is never read; drop it to keep checkpoints small.
        for k, (t, _) in enumerate(scheduler_state['states']):
            if t < 0 and k in stream_state:
                del stream_state[k]
        keys = sorted(sums)
        packed = torch.stack([torch.as_tensor(sums[k], device=device, dtype=torch.float32) for k in keys] +
                             [torch.tensor(float(local_valid), device=device)]) if keys else \
            torch.tensor([float(local_valid)], device=device)
        if world > 1:
            # Ranks may lack keys only when dummy; all ranks have the same keys otherwise.
            sizes = [None] * world
            dist.all_gather_object(sizes, keys)
            if any(s != keys for s in sizes):
                union = sorted(set().union(*map(set, sizes)))
                vals = {k: sums.get(k, torch.zeros((), device=device)) for k in union}
                packed = torch.stack([torch.as_tensor(vals[k], device=device, dtype=torch.float32) for k in union] +
                                     [torch.tensor(float(local_valid), device=device)])
                keys = union
            dist.all_reduce(packed)
        torch.cuda.synchronize()
        seconds = time.time() - t0
        if rank == 0:
            vals = dict(zip(keys, packed[:-1].tolist()))
            if int(packed[-1].item()) != valid_global:
                raise AssertionError('Valid window accounting mismatch')
            rec = dict(update=update, epochs_complete=epochs_complete(scheduler_state),
                       valid_windows=valid_global, windows_seen=windows_seen, suite_windows=item['suites'],
                       seconds=seconds, lr_backbone=opt.param_groups[0]['lr'], grad_norm=float(grad),
                       loss=vals.get('loss', 0.) / valid_global, loss_video=vals.get('video', 0.) / valid_global,
                       loss_action=vals.get('action', 0.) / valid_global,
                       video_raw=vals.get('video_raw', 0.) / valid_global, action_raw=vals.get('action_raw', 0.) / valid_global,
                       peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
            if 'mem_s_rms' in vals:
                rec['mem_s_rms'] = vals['mem_s_rms'] / valid_global
                rec['mem_saturated_frac'] = vals['mem_saturated'] / valid_global
            if model.memory is not None:
                mem = model.memory
                if hasattr(mem, 'a'):
                    rec['mem_lambda_mean'] = float(torch.sigmoid(mem.a).mean())
                rec['tanh_alpha_video_absmean_per_block'] = [round(float(x), 6) for x in torch.tanh(mem.alpha_video).abs().mean(1)]
                rec['tanh_alpha_action_absmean_per_block'] = [round(float(x), 6) for x in torch.tanh(mem.alpha_action).abs().mean(1)]
                if update == 1000 and float(torch.tanh(mem.alpha_video).abs().max()) < 1e-3 and float(torch.tanh(mem.alpha_action).abs().max()) < 1e-3:
                    rec['flag'] = 'gate_still_closed_after_1000_updates'
            log.write(json.dumps(rec) + '\n'); log.flush()
            if update % 10 == 0 or update <= 5:
                print(json.dumps({k: rec[k] for k in ('update', 'valid_windows', 'seconds', 'loss_video', 'loss_action', 'grad_norm')}), flush=True)
            elapsed = elapsed_prior + time.time() - train_start
            (out / 'timing.json').write_text(json.dumps(dict(status='running', update=update, planned_updates=total,
                windows_seen=windows_seen, elapsed_training_seconds=elapsed, last_update_seconds=seconds,
                eta_hours=(total - update) * (time.time() - train_start) / max(1, update - start_update) / 3600,
                heartbeat_unix=time.time(), slurm_job_id=os.getenv('SLURM_JOB_ID')), indent=2))
        if args.no_save:
            pass
        else:
            for e, at in epoch_done_at.items():
                if at == update and e in save_epochs:
                    save('weights', f'epoch_{e:02d}.pt')
            if update in save_updates:
                save('weights', f'update_{update:06d}.pt')
            if update - last_save >= args.checkpoint_every or update == total:
                save('resume'); last_save = update
        if args.max_updates is not None and update >= args.max_updates:
            if not args.no_save and last_save != update:
                save('resume'); last_save = update
            stop_reason = 'max_updates'
            break
        if args.time_limit_hours is not None:
            remaining = torch.tensor(float(args.time_limit_hours * 3600 - (time.time() - wall_start)), device=device)
            if world > 1: dist.all_reduce(remaining, op=dist.ReduceOp.MIN)
            if remaining.item() < 900:
                if not args.no_save and last_save != update:
                    save('resume'); last_save = update
                stop_reason = 'time_limit'
                break
    if rank == 0:
        status = dict(status='complete' if stop_reason == 'complete' else stop_reason, update=update,
                      planned_updates=total, windows_seen=windows_seen,
                      elapsed_training_seconds=elapsed_prior + time.time() - train_start,
                      slurm_job_id=os.getenv('SLURM_JOB_ID'))
        (out / 'timing.json').write_text(json.dumps(status, indent=2))
        if stop_reason == 'complete':
            (out / 'COMPLETE').write_text(json.dumps(status))
        print(json.dumps(dict(event='stop', **status)), flush=True)
        log.close()
    if dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()
    os._exit(0)


if __name__ == '__main__':
    main()
