#!/usr/bin/env python3
"""Full-LIBERO RT training: two GPUs, exact tail weights and resumable epochs."""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from train_loopwam import ExactDistributedBatches,lr_factor
from transport_experiments import EXPERIMENTS
from fastwam.datasets.transport_cached import TransportCachedDataset
from fastwam.models.wan22.loopwam import create_loopwam
from fastwam.models.wan22.transport_policy import TransportPolicy
from fastwam.models.wan22.residual_transport import SCHEDULES,merge_lora

ROOT=Path(__file__).resolve().parents[1]
ASSETS=Path("/mnt/data/vmo-ai-task/anhdh35/FastWAM")
PARENT_RUN=ASSETS/"runs/loopwam_nt/v0_full_libero_job4659_20261006"

def sample_noise(indices,epoch,seed):
    noise=[]; anchor=[]; times=[]
    for index in indices:
        g=torch.Generator().manual_seed(seed+1000003*epoch+97*(max(int(index),0)+1))
        noise.append(torch.randn(32,7,generator=g))
        anchor.append(torch.randn(32,7,generator=g))
        times.append(torch.rand(1,generator=g))
    return torch.stack(noise),torch.stack(anchor),torch.cat(times)

def atomic_save(value,path):
    path=Path(path); temp=path.with_suffix(path.suffix+".tmp")
    torch.save(value,temp); os.replace(temp,path)

def rng_state():
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),
                python=random.getstate(),numpy=np.random.get_state())

def save_recovery(model,opt,ema,out,state,rank,world):
    states=[None]*world
    dist.all_gather_object(states,rng_state()) if world>1 else states.__setitem__(0,rng_state())
    if rank==0:
        weights={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
        atomic_save(dict(format="rt-recovery-v1",weights=weights,optimizer=opt.state_dict(),
                         ema=ema,rng=states,**state),out/"latest.pt")
        (out/"status.json").write_text(json.dumps({k:v for k,v in state.items() if k!="contract"},indent=2))
    if world>1: dist.barrier()

def export_epoch(model,out,epoch,config,update,contract,early=False):
    parent=model.parent
    # A native parent-compatible backbone with merged LoRA plus RT-specific metadata.
    action=merge_lora(copy.deepcopy(model.student.mixtures["action"]).cpu())
    mot={k:v.detach().cpu() for k,v in parent.mot.state_dict().items()}
    for k,v in action.state_dict().items(): mot["mixtures.action."+k]=v
    payload=dict(format_version="loopwam-s-v1",version="v0",architecture=parent.architecture_metadata,
        video_loops=4,action_loops=4,loop_alignment="late",trained_max_loops=4,inference_loops=4,
        exit_weight_scale=parent.exit_weight_scale,mot=mot,
        proprio_encoder={k:v.cpu() for k,v in parent.proprio_encoder.state_dict().items()},
        step=update,transport_config=config,
        student_action={k:v.detach().cpu() for k,v in model.student.mixtures["action"].state_dict().items()},
        inference_action_format="unmerged-exact",transport={k:v.detach().cpu() for k,v in model.transport.state_dict().items()},
        step_condition=None if model.step_condition is None else {k:v.detach().cpu() for k,v in model.step_condition.state_dict().items()},
        training_state=dict(epoch=epoch,update=update,merged_lora=True,contract=contract,
                            windows_seen=update*contract["global_batch"] if early else epoch*contract["windows"]))
    path=out/("step_00002000.pt" if early else f"epoch_{epoch:02d}.pt")
    atomic_save(payload,path)
    # Durable pending files are discovered/retried by uploader and job wrapper.
    (out/(path.name+".upload_pending")).touch()

def validate_stacked_checkpoint(payload):
    config=payload.get("transport_config",{})
    expected=EXPERIMENTS["RT-B2a"]
    keys=("kind","scope","regime","schedules","step_cond","anchor")
    if payload.get("training_state",{}).get("epoch")!=10 or any(config.get(k)!=expected[k] for k in keys):
        raise ValueError("RT+B2 must initialize from the final RT-B2a checkpoint")

def initialize_stacked(model,initial,device):
    validate_stacked_checkpoint(initial)
    # Load merged full action weights before installing the new RT LoRA adapters.
    from fastwam.models.wan22.residual_transport import LoRALinear
    action_state=initial["mot"]
    target=model.student.mixtures["action"]
    plain=copy.deepcopy(model.parent.action_expert)
    plain.load_state_dict({k.removeprefix("mixtures.action."):v for k,v in action_state.items()
                           if k.startswith("mixtures.action.")},strict=True)
    from fastwam.models.wan22.residual_transport import install_core_lora
    plain.requires_grad_(False); install_core_lora(plain)
    model.student.mixtures["action"]=plain.to(device)
    model.step_condition.load_state_dict(initial["step_condition"])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-name",choices=EXPERIMENTS,required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--teacher",default=str(PARENT_RUN/"train/latest.pt"))
    p.add_argument("--vae-path",default=str(ASSETS/"checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"))
    p.add_argument("--manifest",default=str(PARENT_RUN/"train/data/data_manifest.json"))
    p.add_argument("--latent-dir",default=str(PARENT_RUN/"latents"))
    p.add_argument("--init-checkpoint")
    p.add_argument("--microbatch",type=int,default=8)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--global-batch",type=int,default=128)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--save-every",type=int,default=100)
    p.add_argument("--max-updates",type=int)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--no-checkpoint-blocks",action="store_true")
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    rank=int(os.getenv("RANK","0")); world=int(os.getenv("WORLD_SIZE","1"))
    local=int(os.getenv("LOCAL_RANK","0"))
    if world!=2 and not a.smoke: p.error("Production runs require exactly two GPUs")
    if a.global_batch!=128 and not a.smoke: p.error("Production global batch must be 128")
    if a.epochs!=10 and not a.smoke: p.error("Production training must cover ten epochs")
    if a.global_batch%(world*a.microbatch): p.error("Invalid accumulation layout")
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(local); torch.set_num_threads(4)
    if world>1: dist.init_process_group("nccl",device_id=torch.device("cuda",local))
    device=torch.device("cuda",local)
    config=copy.deepcopy(EXPERIMENTS[a.run_name])
    seed=config["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    if (out/"latest.pt").exists() and not a.resume:
        raise ValueError("Run already exists; explicitly resume instead of overwriting")
    if config.get("init_run") and not a.init_checkpoint:
        raise ValueError("RT+B2 requires the final RT-B2a checkpoint")
    data=TransportCachedDataset(a.manifest,a.latent_dir)
    if len(data)!=277713: raise ValueError("Unexpected full LIBERO window count")
    parent=create_loopwam(checkpoint_path=a.teacher,vae_path=a.vae_path,device=str(device))
    parent.requires_grad_(False).eval()
    model=TransportPolicy(parent,**{k:config[k] for k in ("kind","scope","step_cond","regime","anchor")},
        checkpoint_blocks=not a.no_checkpoint_blocks,loss_weights=config.get("loss_weights")).to(device)
    if a.init_checkpoint:
        initial=torch.load(a.init_checkpoint,map_location="cpu",weights_only=False)
        validate_stacked_checkpoint(initial)
        initialize_stacked(model,initial,device)
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(model.optimizer_groups(),betas=(.9,.95),eps=1e-8,fused=True)
    runner=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True) if world>1 else model
    sampler=ExactDistributedBatches(len(data),a.microbatch,rank,world,seed)
    loader=DataLoader(data,batch_sampler=sampler,num_workers=a.workers,pin_memory=True,
        persistent_workers=a.workers>0,generator=torch.Generator().manual_seed(seed+rank))
    total=math.ceil(len(data)/a.global_batch)*a.epochs
    accum=a.global_batch//(world*a.microbatch)
    from fastwam.models.wan22.loopwam_init import sha256_file
    identities=[None]
    if rank==0: identities[0]=dict(teacher=sha256_file(a.teacher),initial=None if a.init_checkpoint is None else sha256_file(a.init_checkpoint),vae=sha256_file(a.vae_path))
    if world>1: dist.broadcast_object_list(identities,src=0)
    contract=dict(asset_sha256=identities[0],config=config,windows=len(data),world=world,microbatch=a.microbatch,
        global_batch=a.global_batch,epochs=a.epochs,planned_updates=total,teacher=str(Path(a.teacher).resolve()),
        normalization_sha256=data.manifest["normalization_sha256"],
        checkpoint_blocks=not a.no_checkpoint_blocks,deterministic_algorithms=True)
    ema={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    update=0; epoch_start=0; micro_start=0; windows_seen=0
    if a.resume:
        state=torch.load(out/"latest.pt",map_location="cpu",weights_only=False)
        if state["contract"]!=contract: raise ValueError("Resume contract changed")
        with torch.no_grad():
            for n,pv in model.named_parameters():
                if pv.requires_grad: pv.copy_(state["weights"][n])
        opt.load_state_dict(state["optimizer"])
        ema={n:v.to(device) for n,v in state["ema"].items()}
        update=state["update"]; epoch_start=state["epoch"]; micro_start=state["next_micro"]
        windows_seen=state["windows_seen"]
        rng=state["rng"][rank]; torch.set_rng_state(rng["cpu"]); torch.cuda.set_rng_state(rng["cuda"])
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
    if a.resume and epoch_start in (8,9,10) and micro_start==0:
        if rank==0 and not (out/f"epoch_{epoch_start:02d}.pt").exists():
            export_epoch(model,out,epoch_start,config,update,contract)
        if world>1: dist.barrier()
    scale=data.transform.normalizer["action"].scale.to(device)
    offset=data.transform.normalizer["action"].offset.to(device)
    if rank==0:
        document=dict(**contract,run=a.run_name,arguments=vars(a),trainable_parameters=sum(p.numel() for p in params),
            slurm_job_id=os.getenv("SLURM_JOB_ID"),git_revision=(ROOT/"REVISION").read_text().strip() if (ROOT/"REVISION").exists() else subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            schedules={s:list(SCHEDULES[s]) for s in config["schedules"]},gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__,hf_folder=f"https://huggingface.co/anhdao69/ResidualTrans/tree/main/{a.run_name}")
        (out/"config.json").write_text(json.dumps(document,indent=2))
        # Preserve canonical bytes: normalization hashes bind the original formatting.
        import shutil
        shutil.copyfile(data.manifest["normalization_path"],out/"dataset_stats.json")
        (out/"data").mkdir(exist_ok=True)
        shutil.copyfile(data.manifest["normalization_path"],out/"data/dataset_stats.json")
        shutil.copyfile(a.manifest,out/"data/data_manifest.json")
        print(json.dumps(dict(event="ready",**document)),flush=True)
    stop_requested=[False]
    signal.signal(signal.SIGUSR1,lambda *_: stop_requested.__setitem__(0,True))
    model.train(); opt.zero_grad(set_to_none=True)
    log=(out/"metrics.jsonl").open("a") if rank==0 else None
    for epoch in range(epoch_start,a.epochs):
        sampler.epoch=epoch; sampler.start_batch=micro_start if epoch==epoch_start else 0
        started=time.perf_counter(); logs={}; schedule=None
        for micro,sample in enumerate(loader,start=sampler.start_batch):
            group=micro//accum
            count=min(a.global_batch,len(data)-group*a.global_batch)
            boundary=(micro+1)%accum==0 or micro+1==len(sampler)
            if micro%accum==0:
                schedule_name=random.Random(seed+update*65537).choice(config["schedules"])
                schedule=SCHEDULES[schedule_name]
                model.teacher_passes=0; model.student_block_calls=0
                model.student.rt_measurement={"forward":0,"backward":0} if update<20 else None
            noise,anchor_noise,anchor_tau=sample_noise(sample["training_index"].tolist(),epoch,seed)
            sample={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in sample.items()}
            sample.update(action_scale=scale,action_offset=offset)
            sync=runner.no_sync() if world>1 and not boundary else contextlib.nullcontext()
            with sync:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    loss,components=runner(sample,schedule,noise.to(device),anchor_noise.to(device),anchor_tau.to(device))
                if not torch.isfinite(loss): raise FloatingPointError("Nonfinite RT loss")
                (loss*world/count).backward()
            for key,val in components.items(): logs[key]=logs.get(key,0)+val
            if not boundary: continue
            grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            factor=lr_factor(update,total,math.ceil(.03*total))
            for group_ in opt.param_groups: group_["lr"]=group_["base_lr"]*factor
            opt.step(); opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for n,pv in model.named_parameters():
                    if pv.requires_grad: ema[n].lerp_(pv,.001)
            update+=1; windows_seen+=count
            torch.cuda.synchronize()
            reduced=torch.stack(list(logs.values()))
            if world>1: dist.all_reduce(reduced)
            record=dict(update=update,epoch=epoch+1,windows=count,windows_seen=windows_seen,
                schedule=schedule_name,seconds=time.perf_counter()-started,grad_norm=float(grad),
                peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,
                teacher_passes_per_microbatch=model.teacher_passes/max(1,math.ceil(count/(world*a.microbatch))),
                student_forward_blocks_per_microbatch=model.student_block_calls/max(1,math.ceil(count/(world*a.microbatch))),
                measured_action_blocks=model.student.rt_measurement,
                **{k:float(v)/count for k,v in zip(logs,reduced)})
            if rank==0:
                log.write(json.dumps(record)+"\n"); log.flush()
                if update<=20 or update%20==0: print(json.dumps(record),flush=True)
            epoch_done=micro+1==len(sampler)
            stopping=torch.tensor(int(stop_requested[0]),device=device)
            if world>1: dist.all_reduce(stopping,op=dist.ReduceOp.MAX)
            maxed=a.max_updates is not None and update>=a.max_updates
            if epoch_done or update%a.save_every==0 or maxed or stopping.item():
                state=dict(contract=contract,epoch=epoch+1 if epoch_done else epoch,
                    next_micro=0 if epoch_done else micro+1,update=update,windows_seen=windows_seen,
                    status="complete" if epoch_done and epoch+1==a.epochs else "running",
                    scheduler=dict(step=update,total=total,warmup=math.ceil(.03*total)))
                save_recovery(model,opt,ema,out,state,rank,world)
            if update==2000 and rank==0:
                export_epoch(model,out,epoch,config,update,contract,early=True)
                import shutil
                shutil.copyfile(out/"latest.pt",out/"step_00002000_recovery.pt.tmp")
                os.replace(out/"step_00002000_recovery.pt.tmp",out/"step_00002000_recovery.pt")
            if epoch_done and epoch+1 in (8,9,10) and rank==0:
                export_epoch(model,out,epoch+1,config,update,contract)
            if world>1 and epoch_done: dist.barrier()
            if maxed or stopping.item():
                if rank==0: print(json.dumps(dict(event="stopped",update=update,resume_ready=True)),flush=True)
                if world>1: dist.destroy_process_group()
                return
            started=time.perf_counter(); logs={}
        micro_start=0
    if rank==0:
        print(json.dumps(dict(event="complete",updates=update,windows_seen=windows_seen)),flush=True)
    if world>1: dist.destroy_process_group()

if __name__=="__main__": main()
