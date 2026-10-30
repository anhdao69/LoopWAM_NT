#!/usr/bin/env python3
"""Original LoopWAM v0; fixed128 real RoboTwin windows, exact5epoch budget."""
import argparse,contextlib,hashlib,json,math,os,queue,random,subprocess,threading,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from fastwam.datasets.chronoloop_robot import RobotWindows,robot_spec,LATENT_SHAPE,OBS_TOKENS,latent_provenance
from fastwam.datasets.loopwam_latent_cache import LoopWAMLatentCache
from fastwam.datasets.loopwam_robot_batches import epoch_rank_batches,training_budget
from fastwam.models.wan22.loopwam_robot import create_loopwam_robot,RobotTrainForward,PARAMETERS,verify_optimizer_restore
from fastwam.models.wan22.loopwam_init import sha256_file
from fastwam.robotwin_pipeline import continuation_contract,continuation_factor,validate_parent_state,validate_cursor,reconcile_metrics
ROBOT_DIM=14

def write(path,payload):
    path=Path(path);tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(path)
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
        nv.append(torch.randn(LATENT_SHAPE, generator=g, device=device))
        na.append(torch.randn((32, ROBOT_DIM), generator=g, device=device))
        u = torch.rand((2,), generator=g, device=device)
        uv.append(u[0]); ua.append(u[1])
    sv, sa = model.train_video_scheduler, model.train_action_scheduler
    tv = sv._phi(torch.stack(uv), sv.shift) * float(sv.num_train_timesteps)
    ta = sa._phi(torch.stack(ua), sa.shift) * float(sa.num_train_timesteps)
    return dict(video=torch.stack(nv), action=torch.stack(na), t_video=tv, t_action=ta)

class Prefetcher:
    def __init__(self,w,cache,rank,world,args,start_epoch=0,start_batch=0):
        self.q=queue.Queue(maxsize=2);self.error=None
        def worker():
            try:
                for epoch in range(start_epoch,args.epochs):
                    first=start_batch if epoch==start_epoch else 0
                    for batch,ids in enumerate(epoch_rank_batches(len(w),args.global_batch,world,rank,args.seed+epoch,first),first):
                        n=min(args.global_batch,len(w)-batch*args.global_batch)
                        micros=[]
                        for lo in range(0,math.ceil(n/world),args.microbatch):
                            chosen=ids[lo:lo+args.microbatch];dummy=not chosen;idx=chosen or [0]
                            w.warm_text([w.prompt(i) for i in idx]);f=[w.fields(i) for i in idx]
                            sample={k:torch.stack([x[k] for x in f]).pin_memory() for k in ('action','proprio','action_is_pad','image_is_pad','context','context_mask')}
                            if not np.all(cache._valid[idx]==1):raise ValueError('Latent cache miss')
                            sample['latents_bits']=torch.from_numpy(np.array(cache._latents[idx]).view(np.int16)).pin_memory()
                            micros.append((sample,idx,dummy))
                        self.q.put((epoch,batch,n,micros))
                self.q.put(None)
            except BaseException as e:self.error=e;self.q.put(None)
        threading.Thread(target=worker,daemon=True).start()
    def get(self):
        item=self.q.get()
        if item is None and self.error:raise self.error
        return item

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='robotwin50');p.add_argument('--run-name',default='LoopWAM-v0-full-RoboTwin')
    p.add_argument('--output-dir',required=True);p.add_argument('--shared-dir',required=True)
    p.add_argument('--init-artifact',required=True);p.add_argument('--vae-path',default='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--memory-tokens',type=int,default=0);p.add_argument('--history-frame',type=int,default=0)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--global-batch',type=int,default=128);p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--lr-backbone',type=float,default=1e-4);p.add_argument('--lr-robot-io',type=float,default=1e-4)
    p.add_argument('--warmup-fraction',type=float,default=.05);p.add_argument('--save-epochs',default='3,4,5')
    p.add_argument('--checkpoint-every',type=int,default=1000);p.add_argument('--text-threads',type=int,default=4)
    p.add_argument('--checkpoint-blocks',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--compile',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--smoke-tail',action='store_true',help='One no-save epoch-tail update for DDP/compile validation only')
    p.add_argument('--max-updates',type=int);p.add_argument('--no-save',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--continuation-from',help='Final epoch5 optimizer checkpoint; starts a separate epochs6-10 phase')
    a=p.parse_args()
    expected_epochs=10 if a.continuation_from else 5
    if (a.dataset,a.epochs,a.seed,a.global_batch,a.memory_tokens,a.history_frame)!=('robotwin50',expected_epochs,42,128,0,0):p.error('Full original-v0 recipe requires robotwin50,5epochs or continuation10,seed42,batch128,memory0,history0')
    if a.continuation_from and (a.lr_backbone,a.lr_robot_io,a.warmup_fraction,a.save_epochs,a.smoke_tail)!=(3e-5,3e-5,0.,'6,7,8,9,10',False):p.error('Continuation requires LR3e-5, no warmup, saves6-10, no tail shortcut')
    if a.smoke_tail and (a.max_updates!=1 or not a.no_save or a.resume):p.error('Tail smoke requires max-updates1/no-save/fresh')
    if a.microbatch not in (1,2,4,8,16):p.error('microbatch must divide16')
    rank,world,local=[int(os.getenv(k,d)) for k,d in [('RANK',0),('WORLD_SIZE',1),('LOCAL_RANK',0)]]
    if world!=8:p.error('This production recipe requires8 GPUs')
    os.environ['TRITON_CACHE_DIR']=f'/tmp/robot_v0_{os.getenv("SLURM_JOB_ID","local")}_rank{rank}'
    os.environ['TORCHINDUCTOR_CACHE_DIR']=os.environ['TRITON_CACHE_DIR']+'_inductor'
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    from datetime import timedelta
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=60))
    torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);shared=Path(a.shared_dir)
    resume=out/'latest.pt'
    if not a.resume and any((out/n).exists() for n in ('manifest.json','metrics.jsonl','latest.pt','COMPLETE')):raise ValueError('Fresh run refuses existing records')
    if a.resume and not resume.exists():raise ValueError('Missing resume checkpoint')
    w=RobotWindows(robot_spec(a.dataset),cache_dir=shared/'data',text_threads=a.text_threads)
    m=w.manifest();assert len(w)==6075103 and len(w.episodes)==27500 and len(w.groups)==50
    old=json.loads((shared/'data/data_manifest.json').read_text());assert old==m,'Data manifest changed'
    hashes=[dict(init=sha256_file(a.init_artifact),vae=sha256_file(a.vae_path),**({'parent':sha256_file(a.continuation_from)} if a.continuation_from else {})) if rank==0 else None];dist.broadcast_object_list(hashes,src=0);hashes=hashes[0]
    cache=LoopWAMLatentCache(shared/'latents',len(w),latent_provenance(w,hashes['vae']),latent_shape=LATENT_SHAPE)
    assert np.all(cache._valid==1),'Incomplete latent cache'
    per_epoch,total,tail=training_budget(len(w),a.global_batch,a.epochs);assert (per_epoch,total,tail)==(47462,47462*a.epochs,95)
    contract=dict(model='original LoopWAM v0',partition=[3,6,3],video_loops=4,action_loops=4,memory_tokens=0,history_frame=0,
        init_sha256=hashes['init'],vae_sha256=hashes['vae'],normalization_sha256=m['normalization_sha256'],data_manifest_sha256=hashlib.sha256(json.dumps(m,sort_keys=True).encode()).hexdigest(),
        epochs=a.epochs,seed=a.seed,global_batch=128,world=8,microbatch=a.microbatch,shuffle='randperm(seed+zero_based_epoch)',
        windows_per_epoch=len(w),planned_windows=len(w)*a.epochs,planned_updates=total,updates_per_epoch=per_epoch,last_batch_windows=95,
        lr_backbone=a.lr_backbone,lr_robot_io=a.lr_robot_io,betas=[.9,.95],eps=1e-8,weight_decay=.01,clip=1.,warmup_fraction=a.warmup_fraction,cosine_floor=.01,
        precision='fp32 weights/moments, bf16 compute',save_epochs=a.save_epochs,smoke_tail=a.smoke_tail)
    if a.continuation_from:
        contract.update(phase_start_update=237310,phase_start_epoch=5,parent_checkpoint_sha256=hashes['parent'])
    model,payload=create_loopwam_robot(a.vae_path,init_artifact=a.init_artifact,checkpoint_path=resume if a.resume else a.continuation_from,
        device=device,checkpoint_blocks=a.checkpoint_blocks,seed=a.seed,artifact_sha256=hashes['init'])
    assert model.architecture_metadata['init_artifact_sha256']==hashes['init']
    model.train();model.mot.structured_attention=True;model.mot.structured_attention_observation_tokens=OBS_TOKENS
    if a.compile:
        torch._dynamo.config.cache_size_limit=256;torch._dynamo.config.accumulated_cache_size_limit=8192
        torch._inductor.config.comprehensive_padding=False
        for name in ('_joint_block','_video_only_block'):setattr(model.mot,name,torch.compile(getattr(model.mot,name)))
    params=model.policy_parameters();assert sum(x.numel() for x in params)==PARAMETERS
    io_ids={id(p) for layer in (model.action_expert.action_encoder,model.action_expert.head,model.proprio_encoder) for p in layer.parameters()}
    groups=[dict(params=[p for p in params if id(p) not in io_ids],lr=a.lr_backbone),dict(params=[p for p in params if id(p) in io_ids],lr=a.lr_robot_io)]
    opt=torch.optim.AdamW(groups,betas=(.9,.95),eps=1e-8,weight_decay=.01,fused=True)
    runner=DDP(RobotTrainForward(model),device_ids=[local],broadcast_buffers=False,find_unused_parameters=False,gradient_as_bucket_view=True)
    update=windows_seen=start_epoch=start_batch=0;elapsed_prior=0.
    if a.resume or a.continuation_from:
        state=payload['training_state']
        if a.resume:assert state['contract']==contract,'Resume contract mismatch'
        else:
            validate_parent_state(payload,contract)
            assert continuation_contract(state['contract'],hashes['parent'])==contract,'Continuation contract mismatch'
        opt.load_state_dict(payload['optimizer']);verify_optimizer_restore(opt,payload['optimizer'])
        update=state['update'];windows_seen=state['windows_seen']
        start_epoch=state['next_epoch'];start_batch=state['next_batch'];elapsed_prior=state['elapsed_training_seconds'] if a.resume else 0.
        validate_cursor(update,windows_seen,start_epoch,start_batch)
        resume_ids=next(epoch_rank_batches(len(w),128,world,rank,a.seed+start_epoch,start_batch),[])
        next_factor=continuation_factor(update) if a.continuation_from and update<total else lr_factor(update,total,int(total*a.warmup_fraction))
        if rank==0:
            if a.resume:reconcile_metrics(out/'metrics.jsonl',update,first_update=237311 if a.continuation_from else 1)
            write(out/f'resume_audit_u{update}.json',dict(passed=True,optimizer_exact=True,next_epoch=start_epoch,next_batch=start_batch,next_rank0_window_ids=resume_ids,next_rank0_noise_seeds=[slot_seed(a.seed,start_epoch,i,0) for i in resume_ids],next_lr=a.lr_backbone*next_factor))
    del payload
    if a.smoke_tail:start_batch=per_epoch-1
    if rank==0:
        write(out/('manifest.json' if not a.resume else f'resume_manifest_u{update}.json'),dict(vars(a),contract=contract,world_size=world,policy_parameters=PARAMETERS,memory_parameters=0,
            resumed_from_update=update if a.resume or a.continuation_from else None,git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),slurm_job_id=os.getenv('SLURM_JOB_ID')))
        (out/'data').mkdir(exist_ok=True);write(out/'data/data_manifest.json',m);write(out/'data/dataset_stats.json',w.stats)
    pref=Prefetcher(w,cache,rank,world,a,start_epoch,start_batch);log=(out/'metrics.jsonl').open('a') if rank==0 else None
    start=time.time();warmup=int(total*a.warmup_fraction);last_save=update;next_epoch,next_batch=start_epoch,start_batch
    initial_update=update
    def save(epoch=None):
        if rank==0:
            state=dict(contract=contract,update=update,windows_seen=windows_seen,next_epoch=next_epoch,next_batch=next_batch,elapsed_training_seconds=elapsed_prior+time.time()-start)
            if epoch is None:model.save_robot(out/'latest.pt',optimizer=opt,step=update,training_state=state)
            else:
                (out/'checkpoints').mkdir(exist_ok=True);model.save_robot(out/'checkpoints'/f'epoch_{epoch:02d}.pt',step=update,training_state=state)
        dist.barrier()
    status='running'
    while True:
        t=time.time();item=pref.get()
        if item is None:status='complete';break
        epoch,batch,n,micros=item
        if (a.resume or a.continuation_from) and epoch==start_epoch and batch==start_batch:
            assert [i for _,ids,dummy in micros if not dummy for i in ids]==resume_ids,'Resumed data differs from expected next batch'
        factor=continuation_factor(update) if a.continuation_from else lr_factor(update,total,warmup)
        for group,lr in zip(opt.param_groups,[a.lr_backbone,a.lr_robot_io]):group['lr']=lr*factor
        sums=torch.zeros(4,device=device);valid_local=0
        for mi,(cpu,ids,dummy) in enumerate(micros):
            sample={k:v.to(device,non_blocking=True) for k,v in cpu.items() if k!='latents_bits'}
            sample['latents']=cpu['latents_bits'].to(device,non_blocking=True).view(torch.bfloat16).float()
            sample['video']=torch.zeros((),device=device).expand(len(ids),3,9,384,320)
            noise=slot_noise(model,[slot_seed(a.seed,epoch,i,0) for i in ids],device)
            ctx=runner.no_sync() if mi<len(micros)-1 else contextlib.nullcontext()
            with ctx:
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,logs,_=runner([dict(sample=sample,noise=noise)])
                (loss*(0. if dummy else world/n)).backward()
            if not dummy:
                valid_local+=len(ids);sums+=torch.stack([loss.detach(),logs['video'],logs['action'],torch.as_tensor(len(ids),device=device)])
        if update==initial_update and any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):raise ValueError('Missing/nonfinite gradient')
        grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
        if update==initial_update:
            moments=sorted({str(v.dtype) for state in opt.state.values() for k,v in state.items() if k in ('exp_avg','exp_avg_sq')})
            assert moments==['torch.float32'] and all(p.dtype==torch.float32 for p in params)
            if rank==0:write(out/'precision.json',dict(policy='float32',compute='bfloat16',optimizer_moment_dtypes=moments,all_gradients_present_and_finite=True))
        dist.all_reduce(sums);assert int(sums[3])==n
        torch.cuda.synchronize();seconds=time.time()-t;update+=1;windows_seen+=n
        done=(batch+1==per_epoch);next_epoch,next_batch=(epoch+1,0) if done else (epoch,batch+1)
        if rank==0:
            rec=dict(update=update,epoch=epoch+1,epochs_complete=epoch+int(done),valid_windows=n,windows_seen=windows_seen,seconds=seconds,
                loss=float(sums[0]/n),loss_video=float(sums[1]/n),loss_action=float(sums[2]/n),grad_norm=float(grad),lr_backbone=opt.param_groups[0]['lr'],peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
            log.write(json.dumps(rec)+'\n');log.flush()
            if update<=5 or update%10==0:print(json.dumps(rec),flush=True)
            write(out/'timing.json',dict(status='running',update=update,planned_updates=total,windows_seen=windows_seen,elapsed_training_seconds=elapsed_prior+time.time()-start,last_update_seconds=seconds,heartbeat_unix=time.time()))
        if not a.no_save:
            if done and epoch+1 in {int(v) for v in a.save_epochs.split(',')}:save(epoch+1)
            if update-last_save>=a.checkpoint_every or done:save();last_save=update
        if a.max_updates and update>=a.max_updates:
            if not a.no_save and last_save!=update:save()
            status='max_updates';break
    if status=='complete':assert update==total and windows_seen==len(w)*a.epochs and next_epoch==a.epochs
    if rank==0:
        result=dict(status=status,update=update,planned_updates=total,windows_seen=windows_seen,elapsed_training_seconds=elapsed_prior+time.time()-start)
        write(out/'timing.json',result)
        if status=='complete':write(out/'COMPLETE',result)
        log.close();print(json.dumps(result),flush=True)
    dist.barrier();dist.destroy_process_group();os._exit(0)
if __name__=='__main__':main()
