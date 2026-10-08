#!/usr/bin/env python3
"""Fresh real-data, global128 DDP/ZeRO throughput trials; no checkpoint resume."""
import argparse, contextlib, hashlib, json, os, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from train_loopwam import ExactDistributedBatches, MarkedDataset
from fastwam.datasets.loopwam_long import build_long_datasets
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.training_backends import initialize_deepspeed_backend, deepspeed_precision_report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--version',choices=['dense_s12','dense_s30','v0','v1','v2'],default='v0')
    p.add_argument('--video-loops',type=int,default=None,help='Number of video core repetitions')
    p.add_argument('--action-loops',type=int,default=None,help='v0 action core repetitions')
    p.add_argument('--backend',choices=['ddp','zero1','zero2'],default='ddp')
    p.add_argument('--microbatch',type=int,required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--updates',type=int,default=5)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--unfused',action='store_true')
    p.add_argument('--structured-attention',action='store_true')
    p.add_argument('--checkpoint-blocks',action='store_true')
    p.add_argument('--latent-cache-dir',default=None)
    a=p.parse_args()
    if a.workers < 0 or a.updates < 3: p.error('workers must be nonnegative and updates >=3')
    rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local); torch.set_num_threads(4)
    dist.init_process_group('nccl',device_id=torch.device('cuda',local))
    assert 128%(world*a.microbatch)==0
    accum=128//(world*a.microbatch)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): raise ValueError('Use a fresh benchmark directory')
    if rank==0: build_long_datasets('data/lerobot_v30/libero_10_no_noops_lerobot','data/text_embeds_cache/libero',str(out/'data'))
    dist.barrier()
    train,_,data_manifest=build_long_datasets('data/lerobot_v30/libero_10_no_noops_lerobot','data/text_embeds_cache/libero',str(out/'data'))
    loader=DataLoader(MarkedDataset(train),batch_sampler=ExactDistributedBatches(len(train),a.microbatch,rank,world),num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,generator=torch.Generator().manual_seed(42+rank))
    torch.manual_seed(42)
    model=create_loopwam('checkpoints/LoopWAM/wan21_compact_donors.pt','checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',version=a.version,device=f'cuda:{local}',checkpoint_blocks=a.checkpoint_blocks,action_loops=a.action_loops,loops=a.video_loops)
    model.train(); model.mot.collect_diagnostics=True
    model.mot.structured_attention=a.structured_attention
    model.mot.structured_attention_observation_tokens=392
    vae_events=[]
    original_encode=model._encode_video_latents
    def timed_encode(*args,**kwargs):
        start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
        start.record(); result=original_encode(*args,**kwargs); end.record()
        vae_events.append((start,end)); return result
    model._encode_video_latents=timed_encode
    if a.latent_cache_dir:
        from fastwam.datasets.loopwam_latent_cache import LoopWAMLatentCache, latent_cache_provenance
        from fastwam.models.wan22.loopwam_init import sha256_file
        hashes=[sha256_file('checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth') if rank==0 else None]
        dist.broadcast_object_list(hashes,src=0)
        provenance=latent_cache_provenance(data_manifest,hashes[0])
        if rank==0: LoopWAMLatentCache(a.latent_cache_dir,len(train),provenance,create=True).close()
        dist.barrier()
        model.training_latent_cache=LoopWAMLatentCache(a.latent_cache_dir,len(train),provenance)

    params=model.policy_parameters()
    if a.backend=='ddp':
        opt=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01,fused=not a.unfused,foreach=False)
        runner=DDP(model,device_ids=None,broadcast_buffers=False,gradient_as_bucket_view=True)
        opt.zero_grad(set_to_none=True)
    else:
        runner=initialize_deepspeed_backend(model,stage=int(a.backend[-1]),microbatch=a.microbatch,world_size=world,fused=not a.unfused)
        runner.optimizer.check_grad_overflow=True
    torch.manual_seed(42+rank)
    it=iter(loader); records=[]
    for update in range(a.updates):
        vae_events.clear()
        dist.barrier(); torch.cuda.synchronize(); start=time.perf_counter()
        finite=torch.ones((),device='cuda',dtype=torch.bool); losses=torch.zeros((),device='cuda')
        stages={k:[] for k in ['data','forward','backward','optimizer']}
        def event():
            e=torch.cuda.Event(enable_timing=True); e.record(); return e
        for micro in range(accum):
            t=time.perf_counter(); sample=next(it); stages['data'].append(time.perf_counter()-t)
            boundary=micro==accum-1
            if a.backend!='ddp': runner.set_gradient_accumulation_boundary(boundary)
            context=runner.no_sync() if a.backend=='ddp' and not boundary else contextlib.nullcontext()
            with context:
                e0=event()
                with torch.autocast('cuda',dtype=torch.bfloat16): loss,logs=runner(sample)
                e1=event(); finite &= torch.isfinite(loss.detach()); losses+=loss.detach()/accum
                if a.backend=='ddp': (loss/accum).backward()
                else: runner.backward(loss/accum,scale_wrt_gas=False)
                e2=event()
            stages['forward'].append((e0,e1)); stages['backward'].append((e1,e2))
            if boundary:
                dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not bool(finite): raise FloatingPointError('Nonfinite loss')
            if a.backend=='ddp':
                if boundary:
                    grad=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True)
                    opt.step(); opt.zero_grad(set_to_none=True)
            else:
                runner.step()
                if boundary:
                    grad=runner.get_global_grad_norm()
                    if runner.optimizer.overflow:
                        raise FloatingPointError('Nonfinite ZeRO gradient; benchmark rejected')
            stages['optimizer'].append((e2,event()))
        torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        vals=torch.tensor([seconds,torch.cuda.max_memory_allocated()/1e9,torch.cuda.max_memory_reserved()/1e9],device='cuda')
        dist.all_reduce(vals,op=dist.ReduceOp.MAX)
        dist.all_reduce(losses); losses/=world
        record=dict(update=update+1,seconds=vals[0].item(),allocated_gb=vals[1].item(),reserved_gb=vals[2].item(),loss=losses.item(),grad_norm=float(grad),data_wait_seconds=sum(stages['data']),vae_cuda_seconds=sum(x.elapsed_time(y) for x,y in vae_events)/1000,**{k+'_cuda_seconds':sum(x.elapsed_time(y) for x,y in stages[k])/1000 for k in ['forward','backward','optimizer']})
        assert torch.isfinite(torch.tensor(record['grad_norm'])) and record['grad_norm']>=0
        records.append(record)
        if rank==0: print(json.dumps(record),flush=True)
    precision=deepspeed_precision_report(runner) if a.backend!='ddp' else {'policy_dtype':'float32','optimizer_moment_dtypes':sorted({str(v.dtype) for s in opt.state.values() for k,v in s.items() if k in ['exp_avg','exp_avg_sq']})}
    if rank==0:
        mean=sum(r['seconds'] for r in records[2:])/len(records[2:])
        result=dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*Path('src/fastwam/models/wan22').glob('loop*.py'),Path('src/fastwam/models/wan22/fastwam.py'),Path('src/fastwam/training_backends.py')]},version=a.version,workers=a.workers,world_size=world,loops=model.mot.loops,action_loops=model.mot.action_loops,backend=a.backend,microbatch=a.microbatch,accumulation=accum,global_batch=128,fused=not a.unfused,structured_attention=a.structured_attention,latent_cache_dir=a.latent_cache_dir,cache_stats=getattr(getattr(model,'training_latent_cache',None),'stats',None),records=records,steady_seconds=mean,projected_10epochs_hours=mean*7250/3600,precision=precision)
        result.update(checkpoint_blocks=a.checkpoint_blocks,policy_parameters=sum(p.numel() for p in params),effective_depth=model.mot.effective_depth(),physical_depth=model.mot.num_layers)
        (out/'result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)
    dist.destroy_process_group()

if __name__=='__main__': main()
