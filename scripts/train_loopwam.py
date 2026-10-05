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
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data._utils.collate import default_collate
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler

from fastwam.datasets.loopwam_long import build_long_datasets
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
        if index < 0:
            sample['action_is_pad'] = torch.ones_like(sample['action_is_pad'],dtype=torch.bool)
            sample['image_is_pad'] = torch.ones_like(sample['image_is_pad'],dtype=torch.bool)
        return sample


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step+1)/max(warmup,1)
    progress = min(1., (step-warmup)/max(total-warmup-1,1))
    return .01+.99*.5*(1+math.cos(math.pi*progress))


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


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',help='Standalone LoopWAM YAML configuration')
    p.add_argument('--version', choices=['v0','v1','v2'],default='v0')
    p.add_argument('--init-artifact',default='checkpoints/LoopWAM/wan21_compact_donors.pt')
    p.add_argument('--vae-path',default='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--dataset-dir',default='data/lerobot_v30/libero_10_no_noops_lerobot')
    p.add_argument('--text-cache-dir',default='data/text_embeds_cache/libero')
    p.add_argument('--output-dir',default=None)
    p.add_argument('--microbatch',type=int,default=1)
    p.add_argument('--global-batch',type=int,default=128)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--max-updates',type=int,default=None,help='Smoke only; recorded separately from epoch budget')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--checkpoint-blocks',action='store_true')
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
    if min(args.microbatch,args.global_batch,args.epochs,args.save_every)<=0: p.error('Batch, epochs, save interval must be positive')
    if args.max_updates is not None and args.max_updates<=0: p.error('--max-updates must be positive')
    if args.workers<0 or args.validation_samples<0: p.error('Worker and validation counts cannot be negative')
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    if world>1:
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
    # Only rank0 persists deterministic split files; other ranks read the same output.
    if rank==0:
        train,val,data_manifest=build_long_datasets(args.dataset_dir,args.text_cache_dir,str(out/'data'),seed=args.seed)
    if world>1: dist.barrier()
    if rank!=0:
        train,val,data_manifest=build_long_datasets(args.dataset_dir,args.text_cache_dir,str(out/'data'),seed=args.seed)
    sampler=ExactDistributedBatches(len(train),args.microbatch,rank,world,args.seed)
    loader=DataLoader(MarkedDataset(train),batch_sampler=sampler,num_workers=args.workers,
        pin_memory=True,persistent_workers=args.workers>0, generator=torch.Generator().manual_seed(args.seed+rank))
    updates_per_epoch=math.ceil(len(train)/args.global_batch)
    total=updates_per_epoch*args.epochs
    model=create_loopwam(args.init_artifact,args.vae_path,version=args.version,device=f'cuda:{local}',
        model_dtype=torch.float32,checkpoint_blocks=args.checkpoint_blocks,checkpoint_path=args.resume)
    params=model.policy_parameters()
    count=sum(p.numel() for p in params)
    if count!=584536135:
        raise AssertionError(f'Unexpected policy parameter count {count}')
    opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False)
    model.train()
    if hasattr(model.mot, 'collect_diagnostics'): model.mot.collect_diagnostics=True
    resume_state=None
    if args.resume:
        payload=model.load_checkpoint(args.resume,optimizer=opt)
        resume_state=payload.get('training_state')
        if not resume_state: raise ValueError('Checkpoint lacks resumable training state')
        expected=dict(world=world,microbatch=args.microbatch,global_batch=args.global_batch,seed=args.seed,
                      train_windows=len(train),planned_updates=total,version=args.version,
                      normalization_sha256=data_manifest['normalization_sha256'])
        if resume_state['contract']!=expected: raise ValueError('Resume training contract changed')
    runner=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False) if world>1 else model
    # Different independent noise streams after identical construction on all ranks.
    torch.manual_seed(args.seed+rank)
    from fastwam.models.wan22.loopwam_init import sha256_file
    asset_hashes=[None]
    if rank==0: asset_hashes[0]=dict(vae=sha256_file(args.vae_path), initialization=sha256_file(args.init_artifact))
    if world>1: dist.broadcast_object_list(asset_hashes,src=0)
    manifest=dict(vars(args),world_size=world,gradient_accumulation=accum,policy_parameters=count,
        asset_sha256=asset_hashes[0],initialization_metadata=model.architecture_metadata,
        train_windows=len(train),val_windows=len(val),updates_per_epoch=updates_per_epoch,
        planned_updates=total,planned_windows=args.epochs*len(train),data=data_manifest,
        policy_dtype='float32',optimizer_state_dtype='float32',compute_dtype='bfloat16',
        optimizer='AdamW replicated (DDP)',torch_version=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_files_sha256={str(path): __import__('hashlib').sha256(path.read_bytes()).hexdigest() for path in
            [Path(__file__),*Path('src/fastwam/models/wan22').glob('loop*.py'),Path('src/fastwam/datasets/loopwam_long.py')]},
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        slurm_job_id=os.getenv('SLURM_JOB_ID'),setup_seconds=time.perf_counter()-setup_start)
    if rank==0:
        manifest_path=out/('resume_manifest.json' if args.resume else 'manifest.json')
        manifest_path.write_text(json.dumps(manifest,indent=2,default=str))
        print(json.dumps({'event':'ready', 'version':args.version,'parameters':count,'train_windows':len(train),
            'planned_updates':total,'global_batch':args.global_batch,'microbatch':args.microbatch,'world_size':world,
            'accumulation':accum,'setup_seconds':manifest['setup_seconds']}),flush=True)
    if args.smoke:
        sample=next(iter(loader))
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            clip=model._encode_video_latents(sample['video'].cuda())
            anchor=model._encode_input_image_latents_tensor(sample['video'][:,:,0].cuda())
        if tuple(clip.shape[1:])!=(16,3,28,56): raise AssertionError(f'VAE shape {clip.shape}')
        torch.testing.assert_close(clip[:,:,:1],anchor,rtol=0,atol=0)
        if rank==0: print(json.dumps({'event':'vae_anchor_pass','shape':list(clip.shape)}),flush=True)
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); train_start=time.perf_counter(); update=0; windows_seen=0
    timings=[]; stop=False
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
    contract=dict(world=world,microbatch=args.microbatch,global_batch=args.global_batch,seed=args.seed,
                  train_windows=len(train),planned_updates=total,version=args.version,
                  normalization_sha256=data_manifest['normalization_sha256'])
    logfile=(out/'metrics.jsonl').open('a') if rank==0 else None
    for epoch in range(start_epoch,args.epochs):
        sampler.epoch=epoch
        sampler.start_batch=start_micro if epoch==start_epoch else 0
        group_start=time.perf_counter(); group_logs={}; group_windows=0
        for micro,sample in enumerate(loader,start=sampler.start_batch):
            group=micro//accum
            valid_global=min(args.global_batch,len(train)-group*args.global_batch)
            boundary=(micro+1)%accum==0 or micro+1==len(loader)
            context=runner.no_sync() if world>1 and not boundary else contextlib.nullcontext()
            with context:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,logs=runner(sample)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                # Per-rank model loss is a microbatch mean, including dummy zero losses.
                # DDP averages gradients across ranks; this yields the exact global mean.
                (loss*(world*args.microbatch/valid_global)).backward()
            group_windows+=int(sample['sample_valid'].sum())
            for key,value in logs.items():
                group_logs[key]=group_logs.get(key,torch.zeros((),device=loss.device))+value*args.microbatch
            if not boundary: continue
            missing=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
            if args.smoke and missing: raise AssertionError(f'Missing gradients: {missing}')
            grad=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            branch_grad={name: torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in module.parameters() if p.grad is not None])).item()
                         for name,module in [('video',model.video_expert),('action',model.action_expert),('proprio',model.proprio_encoder)]}
            learning_rate=1e-4*lr_factor(update,total,int(total*.05))
            for group_opt in opt.param_groups: group_opt['lr']=learning_rate
            opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-group_start
            elapsed_t=torch.tensor(elapsed,device='cuda'); windows_t=torch.tensor(group_windows,device='cuda')
            memory=torch.tensor([torch.cuda.max_memory_allocated()/1e9,torch.cuda.max_memory_reserved()/1e9],device='cuda')
            if world>1:
                dist.all_reduce(elapsed_t,op=dist.ReduceOp.MAX); dist.all_reduce(windows_t)
                dist.all_reduce(memory,op=dist.ReduceOp.MAX)
                for value in group_logs.values(): dist.all_reduce(value)
            elapsed=float(elapsed_t); actual_windows=int(windows_t)
            if actual_windows!=valid_global: raise AssertionError('Global window accounting mismatch')
            update+=1; windows_seen+=actual_windows; timings.append(elapsed)
            record=dict(update=update,epoch=epoch+1,windows=actual_windows,windows_seen=windows_seen,
                effective_passes=windows_seen/len(train),seconds=elapsed,lr=learning_rate,grad_norm=float(grad),
                branch_grad_after_clip=branch_grad,peak_allocated_gb=float(memory[0]),peak_reserved_gb=float(memory[1]),
                **{key:float(value)/valid_global for key,value in group_logs.items()})
            if rank==0:
                print(json.dumps(record),flush=True); logfile.write(json.dumps(record)+'\n'); logfile.flush()
                # Warm first update excluded once enough timed updates exist.
                measured=timings[1:] if len(timings)>1 else timings
                summary=dict(status='running',completed_updates=update,windows_seen=windows_seen,
                    planned_updates=total,elapsed_training_seconds=time.perf_counter()-train_start,
                    measured_mean_update_seconds=float(np.mean(measured)),
                    estimated_total_training_hours=float(np.mean(measured))*total/3600,
                    estimate_note='Projection from measured updates; excludes setup/checkpoint overhead',
                    latest=record)
                (out/'timing.json').write_text(json.dumps(summary,indent=2))
            epoch_done=micro+1==len(loader)
            if epoch_done and args.validation_samples>0 and rank==0:
                validation=evaluate_open_loop(model,val,args.validation_samples,args.seed)
                logfile.write(json.dumps(dict(event='validation',update=update,epoch=epoch+1,**validation))+'\n'); logfile.flush()
                print(json.dumps(dict(event='validation',update=update,**validation)),flush=True)
            save_due=update%args.save_every==0 or epoch_done
            stop=args.max_updates is not None and update>=args.max_updates
            if save_due or stop:
                rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())
                rngs=[None]*world
                if world>1: dist.all_gather_object(rngs,rng)
                else: rngs=[rng]
                if rank==0:
                    state=dict(epoch=epoch,next_micro=micro+1,update=update,windows_seen=windows_seen,rng=rngs,contract=contract)
                    model.save_checkpoint(out/'latest.pt',optimizer=opt,step=update,training_state=state)
                    (out/'trainer_state.json').write_text(json.dumps(dict(epoch=epoch,micro=micro,update=update,windows_seen=windows_seen)))
                if world>1: dist.barrier()
            group_start=time.perf_counter(); group_logs={}; group_windows=0
            if stop: break
        if stop: break
    if rank==0:
        summary['status']='smoke_complete' if args.max_updates is not None else 'complete'
        summary['elapsed_training_seconds']=time.perf_counter()-train_start
        (out/'timing.json').write_text(json.dumps(summary,indent=2))
        logfile.close()
        print(json.dumps({'event':summary['status'],**summary}),flush=True)
    if world>1: dist.destroy_process_group()


if __name__=='__main__': main()
