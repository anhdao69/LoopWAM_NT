#!/usr/bin/env python3
"""Auditable LoopWAM LIBERO-Long training with exact window accounting.

Run with torchrun --standalone --nproc_per_node=2 scripts/train_loopwam.py ...
Policy parameters and AdamW states are FP32; all model forwards use BF16 autocast.
A final partial global batch is weighted by its real sample count, without repeats.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data._utils.collate import default_collate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

from fastwam.datasets.loopwam_long import build_long_datasets, build_full_libero_datasets
from fastwam.models.wan22.loopwam import create_loopwam


class ExactDistributedBatches(Sampler):
    """Same microstep count on all ranks; dummy tail entries carry zero weight.

    Dataset wrapper turns -1 into a marked duplicate. Those entries never contribute
    to the loss/denominator or reported data exposure.
    """
    def __init__(self, size, batch_size, rank, world, seed=42):
        self.size, self.batch_size, self.rank, self.world, self.seed = size,batch_size,rank,world,seed
        self.epoch = 0
        self.start_batch = 0
    def __len__(self):
        return math.ceil(self.size/(self.batch_size*self.world))
    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed+self.epoch)).tolist()
        unit = self.batch_size*self.world
        order += [-1] * ((-len(order)) % unit)
        for start in range(self.start_batch*unit, len(order), unit):
            offset = start+self.rank*self.batch_size
            yield order[offset:offset+self.batch_size]


class MarkedDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data=data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sample = dict(self.data[max(index,0)])
        sample['sample_valid'] = index >= 0
        sample['training_index'] = index
        if index < 0:
            sample['action_is_pad'] = torch.ones_like(sample['action_is_pad'],dtype=torch.bool)
            sample['image_is_pad'] = torch.ones_like(sample['image_is_pad'],dtype=torch.bool)
        return sample


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step+1)/max(warmup,1)
    progress = min(1., (step-warmup)/max(total-warmup-1,1))
    return .01+.99*.5*(1+math.cos(math.pi*progress))


def training_contract(args, world, train_windows, planned_updates, data_manifest):
    contract = dict(world=world, microbatch=args.microbatch, global_batch=args.global_batch,
                    seed=args.seed, train_windows=train_windows, planned_updates=planned_updates,
                    version=args.version, normalization_sha256=data_manifest['normalization_sha256'])
    if hasattr(args, 'action_kv_mode'):
        contract['action_kv_mode'] = args.action_kv_mode
    if data_manifest.get('dataset_scope') == 'full_libero':
        contract.update(dataset_scope='full_libero', epochs=args.epochs,
                        suites=list(data_manifest['suites']))
    if getattr(args, 'action_loops', None) is not None or getattr(args, 'video_loops', None) is not None:
        contract.update(video_loops=getattr(args,'video_loops',None) or 4,
                        action_loops=args.action_loops or getattr(args,'video_loops',None) or 4, loop_alignment='late')
    return contract


@torch.no_grad()
def evaluate_open_loop(model, dataset, count, seed):
    """Fixed held-out observations and action-sampler seeds; no rollout claim."""
    model.eval()
    errors=[]
    indices=torch.linspace(0,len(dataset)-1,min(count,len(dataset))).long().tolist()
    with torch.random.fork_rng(devices=[model.device]):
        for index in indices:
            sample=dataset[index]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model.infer_action(prompt=None,input_image=sample['video'][:,0],action_horizon=32,
                    proprio=sample['proprio'][0],context=sample['context'],context_mask=sample['context_mask'],
                    num_inference_steps=10,seed=seed+index)['action']
            valid=~sample['action_is_pad']
            error=(prediction-sample['action']).float().square().mean(-1)
            errors.append(float(error[valid].mean()))
    model.train()
    return dict(heldout_action_mse=float(np.mean(errors)),samples=len(indices),indices=indices,
                sampler_steps=10,seed=seed)


def configure_training_backend(model, args, world):
    """Create a fresh optimizer; ZeRO keeps model/master/moments in FP32."""
    from fastwam.training_backends import initialize_deepspeed_backend, policy_parameters_fp32, policy_optimizer_parameters
    parameters = policy_parameters_fp32(model)
    if args.backend == 'ddp':
        optimizer = torch.optim.AdamW(policy_optimizer_parameters(model), lr=1e-4, betas=(.9, .95), eps=1e-8,
                                      weight_decay=.01, foreach=False, fused=args.fused_optimizer)
        runner = DDP(model, device_ids=None, broadcast_buffers=False, find_unused_parameters=False,
                     gradient_as_bucket_view=args.bucket_views) if world > 1 else model
    else:
        runner = initialize_deepspeed_backend(model, stage=int(args.backend[-1]),
            microbatch=args.microbatch, global_batch=args.global_batch, world_size=world,
            learning_rate=1e-4, fused=True)
        optimizer = runner.optimizer
        # DeepSpeed defaults this off for FP32 model weights. Enable its
        # partition-aware check so finite-loss/nonfinite-gradient failures skip
        # the update, then fail the runner before any checkpoint is published.
        optimizer.check_grad_overflow = True
    return runner, optimizer


def backward_training_microbatch(runner, loss, *, is_zero, world, microbatch, valid_global):
    weighted = loss * (world * microbatch / valid_global)
    if is_zero:
        runner.backward(weighted, scale_wrt_gas=False)
    else:
        weighted.backward()


def _gradient_group(name):
    prefix = name.split('.', 1)[0]
    return {'video_expert': 'video', 'action_expert': 'action', 'proprio_encoder': 'proprio'}.get(prefix, prefix)


def smoke_policy_gradients(model, *, is_zero):
    """All ranks participate; call once before the first smoke optimizer step."""
    if is_zero:
        from deepspeed.utils import safe_get_full_grad
    missing, finite, norms = [], [], {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = safe_get_full_grad(parameter) if is_zero else parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        finite.append(torch.isfinite(gradient).all())
        norms.setdefault(_gradient_group(name), []).append(gradient.detach().float().norm())
    if missing:
        raise AssertionError(f'Missing gradients: {missing}')
    if not finite or not bool(torch.stack(finite).all()):
        raise FloatingPointError('Nonfinite policy gradients at smoke boundary')
    return {name: float(torch.linalg.vector_norm(torch.stack(values))) for name, values in norms.items()}


def step_training_backend(runner, optimizer, parameters, model, learning_rate, *, is_zero,
                          smoke_grad_norms=None):
    """Apply one boundary update with the scheduled LR and global clipping."""
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
    if is_zero:
        runner.step()
        if optimizer.overflow:
            raise FloatingPointError('DeepSpeed rejected nonfinite accumulated gradients')
        grad = float(runner.get_global_grad_norm())
        if not math.isfinite(grad) or grad < 0:
            raise FloatingPointError('Nonfinite DeepSpeed global gradient norm')
        # Full branch-gradient gathers are a smoke-only diagnostic.
        coefficient = min(1., 1. / (grad + 1e-6))
        branch_grad = None if smoke_grad_norms is None else {
            name: value * coefficient for name, value in smoke_grad_norms.items()}
    else:
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        norms = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                norms.setdefault(_gradient_group(name), []).append(parameter.grad.detach().norm())
        branch_grad = {name: float(torch.linalg.vector_norm(torch.stack(values)))
                       for name, values in norms.items()}
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return grad, branch_grad


def save_training_checkpoint(model, runner, optimizer, output_dir, update, state, *,
                             backend, rank, world):
    """Publish portable evaluation weights plus native partitioned recovery state.

    Native DeepSpeed checkpoints are collective. Keep the newest two completed
    tags; the portable latest.pt contains the exact native directory/tag link,
    but intentionally contains no misleading rank-local optimizer state.
    """
    output_dir = Path(output_dir)
    state = {**state, 'backend': backend}
    is_zero = backend != 'ddp'
    if is_zero:
        tag = f'step_{update:08d}'
        native_dir = output_dir / 'deepspeed'
        state['native_optimizer_checkpoint'] = {'directory': str(native_dir.resolve()), 'tag': tag}
        runner.save_checkpoint(str(native_dir), tag=tag, client_state={'training_state': state},
                               exclude_frozen_parameters=True)
    if rank == 0:
        model.save_checkpoint(output_dir / 'latest.pt', optimizer=None if is_zero else optimizer,
                              step=update, training_state=state)
        (output_dir / 'trainer_state.json').write_text(json.dumps({
            **{key: state[key] for key in ('epoch', 'next_micro', 'update', 'windows_seen', 'backend')},
            'micro': state['next_micro'] - 1}, indent=2))
        if is_zero:
            completed = sorted(path for path in native_dir.glob('step_*')
                               if path.is_dir() and len(path.name) == 13 and path.name[5:].isdigit())
            for old in completed[:-2]:
                shutil.rmtree(old)
    if world > 1:
        dist.barrier()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',help='Standalone LoopWAM YAML configuration')
    p.add_argument('--video-loops',type=int,default=None,help='Number of video core repetitions')
    p.add_argument('--action-kv-mode', choices=['aligned','concat','mix'], default='aligned')
    p.add_argument('--action-loops',type=int,default=None,help='v0 action core repetitions')
    p.add_argument('--backend', choices=['ddp','zero1','zero2'], default='ddp')
    p.add_argument('--version', choices=['dense_s12','dense_s30','v0','v1','v2'],default='v0')
    p.add_argument('--init-artifact',default='checkpoints/LoopWAM/wan21_compact_donors.pt')
    p.add_argument('--vae-path',default='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--dataset-dir',default='data/lerobot_v30/libero_10_no_noops_lerobot')
    p.add_argument('--dataset-scope',choices=['long_split','full_libero'],default='long_split',
                   help='full_libero uses all four suite subdirectories and all demonstrations')
    p.add_argument('--text-cache-dir',default='data/text_embeds_cache/libero')
    p.add_argument('--output-dir',default=None)
    p.add_argument('--microbatch',type=int,default=1)
    p.add_argument('--global-batch',type=int,default=128)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--max-updates',type=int,default=None,help='Smoke only; recorded separately from epoch budget')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--checkpoint-blocks',action='store_true')
    p.add_argument('--fused-optimizer',action='store_true')
    p.add_argument('--bucket-views',action='store_true')
    p.add_argument('--structured-attention',action='store_true')
    p.add_argument('--latent-cache-dir',default=None)
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--resume',default=None)
    p.add_argument('--validation-samples',type=int,default=8)
    p.add_argument('--smoke',action='store_true',help='Check all intended gradients and VAE anchor before training')
    pre,_=p.parse_known_args()
    if pre.config:
        from omegaconf import OmegaConf
        config=OmegaConf.to_container(OmegaConf.load(pre.config),resolve=True)
        allowed={action.dest for action in p._actions}
        if set(config)-allowed: raise ValueError(f'Unknown config keys: {set(config)-allowed}')
        p.set_defaults(**config)
    args=p.parse_args()
    if not args.output_dir: p.error('--output-dir is required')
    is_zero = args.backend != 'ddp'
    if is_zero and args.resume: p.error('--resume is currently supported only for DDP; ZeRO runs must start fresh')
    if is_zero: args.fused_optimizer = True
    if min(args.microbatch,args.global_batch,args.epochs,args.save_every)<=0: p.error('Batch, epochs, save interval must be positive')
    if args.max_updates is not None and args.max_updates<=0: p.error('--max-updates must be positive')
    if args.workers<0 or args.validation_samples<0: p.error('Worker and validation counts cannot be negative')
    if args.dataset_scope == 'full_libero' and args.validation_samples != 0:
        p.error('Full LIBERO uses every demonstration for training; set --validation-samples 0 and use simulator evaluation')
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    if world>1 or is_zero:
        dist.init_process_group('nccl',device_id=torch.device('cuda',local))
    if args.global_batch % (world*args.microbatch):
        raise ValueError('Global batch must be divisible by world size times microbatch')
    accum=args.global_batch//(world*args.microbatch)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.jsonl').exists() and not args.resume:
        raise ValueError('Output already contains a run; use a new directory or --resume')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    setup_start=time.perf_counter()
    build_datasets = build_full_libero_datasets if args.dataset_scope == 'full_libero' else build_long_datasets
    # Only rank0 persists deterministic split files; other ranks read the same output.
    if rank==0:
        train,val,data_manifest=build_datasets(args.dataset_dir,args.text_cache_dir,str(out/'data'),seed=args.seed)
    if world>1: dist.barrier()
    if rank!=0:
        train,val,data_manifest=build_datasets(args.dataset_dir,args.text_cache_dir,str(out/'data'),seed=args.seed)
    sampler=ExactDistributedBatches(len(train),args.microbatch,rank,world,args.seed)
    loader=DataLoader(MarkedDataset(train),batch_sampler=sampler,num_workers=args.workers,
        pin_memory=True,persistent_workers=args.workers>0, generator=torch.Generator().manual_seed(args.seed+rank))
    updates_per_epoch=math.ceil(len(train)/args.global_batch)
    total=updates_per_epoch*args.epochs
    model=create_loopwam(args.init_artifact,args.vae_path,version=args.version,device=f'cuda:{local}',
        model_dtype=torch.float32,checkpoint_blocks=args.checkpoint_blocks,checkpoint_path=args.resume,action_loops=args.action_loops,loops=args.video_loops,action_kv_mode=args.action_kv_mode)
    params=model.policy_parameters()
    count=sum(p.numel() for p in params)
    from fastwam.training_backends import expected_policy_parameters
    expected_count = expected_policy_parameters(args.version, args.action_kv_mode, model.mot.loops)
    if count!=expected_count:
        raise AssertionError(f'Unexpected policy parameter count {count}')
    model.train()
    model.mot.structured_attention=args.structured_attention
    model.mot.structured_attention_observation_tokens=392
    if hasattr(model.mot, 'collect_diagnostics'): model.mot.collect_diagnostics=True
    runner,opt=configure_training_backend(model,args,world)
    resume_state=None
    if args.resume:
        payload=model.load_checkpoint(args.resume)
        resume_state=payload.get('training_state')
        if resume_state and resume_state.get('backend','ddp')!='ddp':
            raise ValueError('Portable ZeRO weights cannot resume a DDP optimizer; native DeepSpeed checkpoints are stored separately')
        if 'optimizer' not in payload: raise ValueError('Checkpoint lacks a complete DDP optimizer state')
        opt.load_state_dict(payload['optimizer'])
        if not resume_state: raise ValueError('Checkpoint lacks resumable training state')
        expected=training_contract(args,world,len(train),total,data_manifest)
        if {'action_kv_mode':'aligned', **resume_state['contract']}!=expected: raise ValueError('Resume training contract changed')
    # Different independent noise streams after identical construction on all ranks.
    torch.manual_seed(args.seed+rank)
    from fastwam.models.wan22.loopwam_init import sha256_file
    asset_hashes=[None]
    if rank==0: asset_hashes[0]=dict(vae=sha256_file(args.vae_path), initialization=sha256_file(args.init_artifact))
    if world>1: dist.broadcast_object_list(asset_hashes,src=0)
    if args.latent_cache_dir:
        from fastwam.datasets.loopwam_latent_cache import LoopWAMLatentCache, latent_cache_provenance
        provenance=latent_cache_provenance(data_manifest,asset_hashes[0]['vae'])
        if rank==0: LoopWAMLatentCache(args.latent_cache_dir,len(train),provenance,create=True).close()
        if world>1: dist.barrier()
        model.training_latent_cache=LoopWAMLatentCache(args.latent_cache_dir,len(train),provenance)

    manifest=dict(vars(args),initialization_mode='resume_checkpoint' if args.resume else 'canonical_wan_artifact_fresh_optimizer',world_size=world,gradient_accumulation=accum,policy_parameters=count,loops=model.mot.loops,action_core_loops=model.mot.action_loops,loop_alignment="late",
        asset_sha256=asset_hashes[0],initialization_metadata=model.architecture_metadata,
        train_windows=len(train),val_windows=len(val),updates_per_epoch=updates_per_epoch,
        planned_updates=total,planned_windows=args.epochs*len(train),data=data_manifest,
        policy_dtype='float32',optimizer_state_dtype='float32',compute_dtype='bfloat16',
        optimizer='AdamW replicated (DDP)' if not is_zero else f'AdamW partitioned ({args.backend})',
        deepspeed_config=getattr(runner,'loopwam_backend_config',None),
        gradient_overflow_check=True if is_zero else 'clip_grad_norm_error_if_nonfinite',
        deepspeed_version=__import__('deepspeed').__version__ if is_zero else None,torch_version=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_files_sha256={str(path): __import__('hashlib').sha256(path.read_bytes()).hexdigest() for path in
            [Path(__file__),*Path('src/fastwam/models/wan22').glob('loop*.py'),Path('src/fastwam/datasets/loopwam_long.py'),Path('src/fastwam/models/wan22/fastwam.py'),Path('src/fastwam/datasets/loopwam_latent_cache.py'),Path('src/fastwam/training_backends.py')]},
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        slurm_job_id=os.getenv('SLURM_JOB_ID'),setup_seconds=time.perf_counter()-setup_start)
    if rank==0:
        manifest_path=out/('resume_manifest.json' if args.resume else 'manifest.json')
        manifest_path.write_text(json.dumps(manifest,indent=2,default=str))
        print(json.dumps({'event':'ready', 'version':args.version,'parameters':count,'train_windows':len(train),
            'planned_updates':total,'global_batch':args.global_batch,'microbatch':args.microbatch,'world_size':world,
            'accumulation':accum,'initialization_mode':manifest['initialization_mode'],'backend':args.backend,'optimizer_state_entries_before_training':len(opt.optimizer.state if is_zero else opt.state),'setup_seconds':manifest['setup_seconds']}),flush=True)
    if args.smoke:
        sample=next(iter(loader))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            clip=model._encode_video_latents(sample['video'].cuda())
            anchor=model._encode_input_image_latents_tensor(sample['video'][:,:,0].cuda())
        if tuple(clip.shape[1:])!=(16,3,28,56): raise AssertionError(f'VAE shape {clip.shape}')
        torch.testing.assert_close(clip[:,:,:1],anchor,rtol=0,atol=0)
        if rank==0: print(json.dumps({'event':'vae_anchor_pass','shape':list(clip.shape)}),flush=True)
    if not is_zero: opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); train_start=time.perf_counter(); update=0; windows_seen=0
    timings=[]; stop=False
    checkpoint_seconds_total=0.; checkpoint_count=0
    start_epoch=0; start_micro=0
    if resume_state:
        start_epoch=resume_state['epoch']; start_micro=resume_state['next_micro']
        update=resume_state['update']; windows_seen=resume_state['windows_seen']
        if start_micro==len(loader): start_epoch+=1; start_micro=0
        if start_epoch>=args.epochs: raise ValueError('Checkpoint has already completed the requested epoch budget')
        if args.max_updates is not None and update>=args.max_updates:
            raise ValueError('Checkpoint has already reached --max-updates')
        rng=resume_state['rng'][rank]
        torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda'])
        random.setstate(rng['python']); np.random.set_state(rng['numpy'])
    contract=training_contract(args,world,len(train),total,data_manifest)
    logfile=(out/'metrics.jsonl').open('a') if rank==0 else None
    for epoch in range(start_epoch,args.epochs):
        sampler.epoch=epoch
        sampler.start_batch=start_micro if epoch==start_epoch else 0
        group_start=time.perf_counter(); group_logs={}; group_windows=0
        group_finite=torch.ones((),device='cuda',dtype=torch.bool)
        for micro,sample in enumerate(loader,start=sampler.start_batch):
            group=micro//accum
            valid_global=min(args.global_batch,len(train)-group*args.global_batch)
            boundary=(micro+1)%accum==0 or micro+1==len(loader)
            if is_zero: runner.set_gradient_accumulation_boundary(boundary)
            context=runner.no_sync() if not is_zero and world>1 and not boundary else contextlib.nullcontext()
            with context:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,logs=runner(sample)
                group_finite &= torch.isfinite(loss.detach())
                # Per-rank model loss is a microbatch mean, including dummy zero losses.
                # DDP averages gradients across ranks; this yields the exact global mean.
                backward_training_microbatch(runner,loss,is_zero=is_zero,world=world,
                    microbatch=args.microbatch,valid_global=valid_global)
            group_windows+=int(sample['sample_valid'].sum())
            for key,value in logs.items():
                group_logs[key]=group_logs.get(key,torch.zeros((),device=loss.device))+value*args.microbatch
            if not boundary:
                if is_zero: runner.step()
                continue
            if world>1: dist.all_reduce(group_finite,op=dist.ReduceOp.MIN)
            if not bool(group_finite): raise FloatingPointError('Nonfinite accumulated training loss')
            smoke_norms=smoke_policy_gradients(model,is_zero=is_zero) if args.smoke and update==0 else None
            learning_rate=1e-4*lr_factor(update,total,int(total*.05))
            grad,branch_grad=step_training_backend(runner,opt,params,model,learning_rate,
                is_zero=is_zero,smoke_grad_norms=smoke_norms)
            if is_zero and update==0:
                from fastwam.training_backends import deepspeed_precision_report
                precision=deepspeed_precision_report(runner)
                if not precision['optimizer_state_initialized']: raise AssertionError('AdamW moments absent after first update')
                if rank==0: print(json.dumps(dict(event='backend_precision',backend=args.backend,**precision)),flush=True)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-group_start
            elapsed_t=torch.tensor(elapsed,device='cuda'); windows_t=torch.tensor(group_windows,device='cuda')
            memory=torch.tensor([torch.cuda.max_memory_allocated()/1e9,torch.cuda.max_memory_reserved()/1e9],device='cuda')
            if world>1:
                dist.all_reduce(elapsed_t,op=dist.ReduceOp.MAX); dist.all_reduce(windows_t)
                dist.all_reduce(memory,op=dist.ReduceOp.MAX)
                packed_logs=torch.stack(list(group_logs.values()))
                dist.all_reduce(packed_logs)
                group_logs=dict(zip(group_logs,packed_logs.unbind()))
            elapsed=float(elapsed_t); actual_windows=int(windows_t)
            if actual_windows!=valid_global: raise AssertionError('Global window accounting mismatch')
            update+=1; windows_seen+=actual_windows; timings.append(elapsed)
            record=dict(update=update,epoch=epoch+1,windows=actual_windows,windows_seen=windows_seen,
                effective_passes=windows_seen/len(train),seconds=elapsed,lr=learning_rate,grad_norm=float(grad),
                branch_grad_after_clip=branch_grad,latent_cache_rank_local=getattr(getattr(model,'training_latent_cache',None),'stats',None),peak_allocated_gb=float(memory[0]),peak_reserved_gb=float(memory[1]),
                **{key:float(value)/valid_global for key,value in group_logs.items()})
            if rank==0:
                print(json.dumps(record),flush=True); logfile.write(json.dumps(record)+'\n'); logfile.flush()
                # Warm first update excluded once enough timed updates exist.
                measured=timings[1:] if len(timings)>1 else timings
                summary=dict(status='running',completed_updates=update,windows_seen=windows_seen,
                    planned_updates=total,elapsed_training_seconds=time.perf_counter()-train_start,
                    checkpoint_seconds_total=checkpoint_seconds_total,checkpoint_count=checkpoint_count,
                    measured_mean_update_seconds=float(np.mean(measured)),
                    estimated_total_training_hours=float(np.mean(measured))*total/3600,
                    estimate_note='Projection from measured updates; excludes setup/checkpoint overhead',
                    latest=record)
                (out/'timing.json').write_text(json.dumps(summary,indent=2))
            epoch_done=micro+1==len(loader)
            if epoch_done and args.latent_cache_dir and rank==0:
                coverage=int(np.count_nonzero(np.memmap(Path(args.latent_cache_dir)/'valid.uint8',mode='r',dtype=np.uint8)==1))
                if not args.resume and coverage!=len(train): raise AssertionError('Fresh full epoch did not populate every real latent cache entry')
                logfile.write(json.dumps(dict(event='latent_cache_coverage',epoch=epoch+1,valid_windows=coverage,total_windows=len(train)))+'\n'); logfile.flush()
            if epoch_done and args.validation_samples>0 and rank==0:
                validation=evaluate_open_loop(model,val,args.validation_samples,args.seed)
                logfile.write(json.dumps(dict(event='validation',update=update,epoch=epoch+1,**validation))+'\n'); logfile.flush()
                print(json.dumps(dict(event='validation',update=update,**validation)),flush=True)
            if epoch_done and args.action_kv_mode != 'aligned' and rank==0:
                from loopwam_kv_diagnostics import collect_kv_diagnostics
                diagnostic=collect_kv_diagnostics(model,val,epoch+1,args.seed)
                logfile.write(json.dumps(diagnostic)+'\n');logfile.flush()
                print(json.dumps(diagnostic),flush=True)
            save_due=update%args.save_every==0 or epoch_done
            stop=args.max_updates is not None and update>=args.max_updates
            if save_due or stop:
                rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())
                rngs=[None]*world
                if world>1: dist.all_gather_object(rngs,rng)
                else: rngs=[rng]
                state=dict(epoch=epoch,next_micro=micro+1,update=update,windows_seen=windows_seen,rng=rngs,contract=contract)
                checkpoint_start=time.perf_counter()
                save_training_checkpoint(model,runner,opt,out,update,state,
                    backend=args.backend,rank=rank,world=world)
                checkpoint_seconds_total+=time.perf_counter()-checkpoint_start
                checkpoint_count+=1
            group_start=time.perf_counter(); group_logs={}; group_windows=0
            group_finite.fill_(True)
            if stop: break
        if stop: break
    if rank==0:
        summary['status']='smoke_complete' if args.max_updates is not None else 'complete'
        summary['elapsed_training_seconds']=time.perf_counter()-train_start
        summary['checkpoint_seconds_total']=checkpoint_seconds_total
        summary['checkpoint_count']=checkpoint_count
        (out/'timing.json').write_text(json.dumps(summary,indent=2))
        logfile.close()
        print(json.dumps({'event':summary['status'],**summary}),flush=True)
    if dist.is_initialized(): dist.destroy_process_group()


if __name__=='__main__': main()
