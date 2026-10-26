#!/usr/bin/env python3
"""Centered-donor full-suite ablations, four GPUs per existing allocation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_kv_campaign import train_command, FULL_CACHE, FULL_BASE, verify_fairness, spec, read, write

GRID = {
    'p0c6s6_k4': ((0,6,6),4),
    'p1c6s5_k4': ((1,6,5),4),
    'p5c6s1_k4': ((5,6,1),4),
    'p6c6s0_k4': ((6,6,0),4),
    'p1c9s2_k3': ((1,9,2),3),
    'p4c3s5_k7': ((4,3,5),7),
}
QUEUES = {'A': ('p0c6s6_k4','p5c6s1_k4','p1c9s2_k3'),
          'B': ('p1c6s5_k4','p6c6s0_k4','p4c3s5_k7')}
REPO = 'anhdao69/LoopWAM_NT'


def hash_sources(root=Path('.')):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in ('src','scripts') for p in sorted((root/d).rglob('*.py'))}


def require_sources(expected, root=Path('.')):
    if hash_sources(root) != expected:
        raise ValueError('Source differs from verified release')


def experiment(label):
    _, loops = GRID[label]
    return spec(label, v=loops, a=loops)


def command(label, config, out, smoke=False):
    cmd = train_command(experiment(label),config,out,FULL_CACHE,10 if smoke else None)
    cmd[cmd.index('--nproc_per_node=2')]='--nproc_per_node=4'
    cmd[cmd.index('--workers')+1]='4'
    cmd += ['--donor-strategy','stage_centered']
    cmd += ['--block-partition', *map(str,GRID[label][0])]
    # Checkpoints every epoch plus every 500 updates bound restart loss.
    cmd[cmd.index('--save-every')+1]='500'
    return cmd


def run_logged(cmd,path):
    write(path.with_suffix('.command.json'),cmd)
    with path.open('x') as log:
        return subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode


def verify(label, train, smoke=False):
    import math
    m=read(train/'manifest.json'); timing=read(train/'timing.json'); state=read(train/'trainer_state.json')
    expected=10 if smoke else 21700
    required=dict(version='v0',action_kv_mode='aligned',donor_strategy='stage_centered',loops=GRID[label][1],action_core_loops=GRID[label][1],global_batch=128,epochs=10,seed=42,world_size=4,train_windows=277713,planned_updates=21700,planned_windows=2777130,policy_parameters=584536135,resume=None,max_updates=10 if smoke else None)
    if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Centered training contract mismatch')
    if m['microbatch']*4*m['gradient_accumulation']!=128:raise ValueError('Global batch mismatch')
    fairness=verify_fairness(m,read(FULL_BASE/'manifest.json'))
    if timing['status']!=('smoke_complete' if smoke else 'complete') or timing['completed_updates']!=expected or state['update']!=expected or timing['windows_seen']!=(1280 if smoke else 2777130):raise ValueError('Incomplete training')
    if not smoke and (state['epoch']!=9 or state['next_micro']!=math.ceil(277713/(4*m['microbatch']))):raise ValueError('Incomplete last epoch')
    rows=[json.loads(line) for line in (train/'metrics.jsonl').read_text().splitlines()]
    updates=[row for row in rows if 'loss_video' in row]
    if len(updates)!=expected or any(not math.isfinite(row[k]) for row in updates for k in ('loss_video','loss_action','grad_norm')):raise ValueError('Invalid training metrics')
    write(train/'fairness.json',fairness)
    if m['block_partition'] != list(GRID[label][0]):
        raise ValueError('Partition manifest mismatch')
    import torch
    from evaluate_loopwam_libero import validate_checkpoint
    from fastwam.models.wan22.loopwam import _validate_checkpoint_depth
    p=torch.load(train/'latest.pt',map_location='cpu',mmap=True,weights_only=False)
    _validate_checkpoint_depth(p)
    if p['block_partition'] != m['block_partition'] or p['inference_loops']!=GRID[label][1]:
        raise ValueError('Checkpoint partition mismatch')
    validate_checkpoint(p,m['data'],m['data']['normalization_sha256'],smoke=smoke)
    from fastwam.models.wan22.loopwam_init import stage_donor_indices
    if p['architecture'].get('donor_strategy')!='stage_centered' or p['architecture']['donor_indices']!=list(stage_donor_indices(GRID[label][0])):
        raise ValueError('Wrong centered donor initialization')
    if m['backend']=='ddp' and not p.get('optimizer',{}).get('state'):
        raise ValueError('Missing resumable optimizer')
    if m['backend']!='ddp':
        native=p['training_state']['native_optimizer_checkpoint']
        if not (Path(native['directory'])/native['tag']).is_dir():raise ValueError('Missing native optimizer')
    return timing


def validate_benchmark(result):
    import math
    if (result.get('global_batch')!=128 or result.get('world_size')!=4 or result.get('train_windows')!=277713
        or result.get('donor_strategy')!='stage_centered' or result.get('precision',{}).get('policy_dtype')!='float32'
        or result['precision'].get('optimizer_moment_dtypes')!=['torch.float32']
        or not math.isfinite(result['steady_seconds']) or result['steady_seconds']<=0
        or len(result['records'])!=8
        or any(not math.isfinite(row[k]) for row in result['records'] for k in ('loss','grad_norm','seconds'))):
        raise ValueError('Benchmark precision/finite/budget gate failed')


def benchmark(root,lane):
    layouts=([('ddp',8,False),('ddp',16,False),('ddp',16,True),('ddp',32,True)] if lane=='A'
             else [(b,m,c) for b in ('zero1','zero2') for m,c in ((8,False),(16,False),(32,True))])
    configs=[dict(backend=b,microbatch=m,checkpoint_blocks=c) for b,m,c in layouts]
    # Both lanes measure the same reference independently on disjoint GPU pairs.
    results=[]
    for config in configs:
        name=f"bench_{lane}_{config['backend']}_mb{config['microbatch']}_cp{int(config['checkpoint_blocks'])}"
        out=root/name
        cmd=['torchrun','--standalone','--nproc_per_node=4','scripts/benchmark_loopwam.py',
             '--version','v0','--video-loops','4','--action-loops','4','--block-partition','3','6','3',
             '--donor-strategy','stage_centered','--backend',config['backend'],'--microbatch',str(config['microbatch']),'--dataset-scope','full_libero',
             '--workers','4','--updates','8','--structured-attention','--latent-cache-dir',str(FULL_CACHE),'--output-dir',str(out)]
        if config['checkpoint_blocks']:cmd+=['--checkpoint-blocks']
        code=run_logged(cmd,root/f'{name}.log')
        if code:
            text=(root/f'{name}.log').read_text()
            if 'CUDA out of memory' not in text:
                raise RuntimeError(f'Non-OOM benchmark failure: {name}')
            results.append(dict(config=config,status='oom'))
        else:
            result=read(out/'result.json')
            validate_benchmark(result)
            results.append(dict(config=config,status='passed',seconds=result['steady_seconds']))
        write(root/f'bench_{lane}.json',results)


def smoke(root,lane):
    config=read(root/'selected.json')['config']
    sources=hash_sources()
    for label in QUEUES[lane]:
        require_sources(sources)
        out=root/'smoke'/label
        out.parent.mkdir(exist_ok=True)
        if run_logged(command(label,config,out,True),root/f'smoke_{label}.log'):
            raise RuntimeError(f'Smoke failed: {label}')
        require_sources(sources)
        timing=verify(label,out,True)
        # Fresh process exercises native factory restore and actual action inference.
        if run_logged([sys.executable,'scripts/check_partition_checkpoint.py','--checkpoint',str(out/'latest.pt'),
                       '--output',str(out/'inference_check.json')],root/f'inference_{label}.log'):
            raise RuntimeError(f'Inference reload failed: {label}')
        write(out/'passed.json',dict(label=label,timing=timing,inference=read(out/'inference_check.json')))
    require_sources(sources)
    write(root/f'smoke_{lane}_complete.json',dict(labels=list(QUEUES[lane]),sources=sources,config=config))


def validate_export(export, source_digest, source=None):
    from fastwam.models.wan22.loopwam_init import sha256_file
    if not (export/'provenance.json').exists():
        # Recover an interrupted first export only after exact weight/metadata comparison.
        if source is None:raise ValueError('Incomplete export requires original checkpoint')
        import torch
        saved=torch.load(export/'policy.pt',map_location='cpu',mmap=True,weights_only=False)
        fields=('format_version','version','step','architecture','block_partition','video_loops','action_loops','action_kv_mode','trained_max_loops','inference_loops','exit_weight_scale')
        if 'optimizer' in saved or any(saved.get(k)!=source.get(k) for k in fields):raise ValueError('Incomplete export metadata mismatch')
        for family in ('mot','proprio_encoder'):
            if saved[family].keys()!=source[family].keys():raise ValueError('Incomplete export tensor coverage mismatch')
            for key,value in saved[family].items():
                other=source[family][key]
                if value.dtype!=other.dtype or not torch.equal(value,other):raise ValueError('Incomplete export tensor mismatch')
        for field in ('contract','epoch','next_micro','update','windows_seen'):
            if (saved.get('training_state') or {}).get(field)!=(source.get('training_state') or {}).get(field):raise ValueError('Incomplete export training state mismatch')
        return
    prior=read(export/'provenance.json')
    if prior['source_checkpoint_sha256']!=source_digest or prior['sha256']!=sha256_file(export/'policy.pt'):
        raise ValueError('Existing export does not match current final checkpoint')


def publish(label,train,root):
    import torch
    from huggingface_hub import HfApi, CommitOperationAdd
    verify(label,train)
    export=root/'exports'/label
    export.mkdir(parents=True,exist_ok=True)
    weights=export/'policy.pt'
    from fastwam.models.wan22.loopwam_init import sha256_file
    source_digest=sha256_file(train/'latest.pt')
    payload=torch.load(train/'latest.pt',map_location='cpu',mmap=True,weights_only=False)
    if weights.exists():
        validate_export(export,source_digest,payload)
    if not weights.exists():
        payload.pop('optimizer',None)
        torch.save(payload,weights.with_suffix('.tmp'))
        weights.with_suffix('.tmp').replace(weights)
    from fastwam.models.wan22.loopwam_init import sha256_file
    digest=sha256_file(weights)
    destination=f'libero-all-suites/partition-grid-centered-20261026/{label}'
    m=read(train/'manifest.json')
    info=dict(repository=REPO,path=destination,sha256=digest,source_checkpoint_sha256=source_digest,
              block_partition=list(GRID[label][0]),loops=GRID[label][1],donor_strategy='stage_centered',donor_indices=payload['architecture']['donor_indices'],training_manifest=m,
              timing=read(train/'timing.json'),evaluation='not run',optimizer_exported=False)
    write(export/'provenance.json',info)
    for name in ('manifest.json','timing.json','fairness.json'):
        shutil.copy2(train/name,export/name)
    for name in ('dataset_stats.json','data_manifest.json'):
        shutil.copy2(train/'data'/name,export/name)
    (export/'README.md').write_text(f'# LoopWAM v0 {label}\n\nFull four-suite LIBERO, 10 epochs, global batch 128, seed 42. '
        f'Partition {GRID[label][0]}, {GRID[label][1]} loops in both experts; aligned KV; 584,536,135 parameters. '
        'Fresh Wan initialization with beginning prelude, centered core, ending coda; lower-index tie for odd core widths. Final checkpoint; no simulator evaluation yet. '
        'Requires partition-aware LoopWAM code; block_partition must not be ignored. '
        'policy.pt contains FP32 inference weights and training provenance, without optimizer states.\n')
    api=HfApi()
    if api.file_exists(REPO,destination+'/policy.pt'):
        remote=api.model_info(REPO,files_metadata=True)
        obj=next(x for x in remote.siblings if x.rfilename==destination+'/policy.pt')
        if not obj.lfs or obj.lfs.sha256!=digest:
            raise ValueError('Refusing to overwrite different released weights')
    result=api.create_commit(repo_id=REPO,operations=[CommitOperationAdd(path_in_repo=destination+'/'+p.name,path_or_fileobj=str(p)) for p in sorted(export.iterdir()) if p.is_file()],commit_message=f'Release full LIBERO partition {label}, 10 epochs')
    commit=result.oid
    remote=api.model_info(REPO,revision=commit,files_metadata=True)
    obj=next(x for x in remote.siblings if x.rfilename==destination+'/policy.pt')
    if not obj.lfs or obj.lfs.sha256!=digest:raise ValueError('Remote weight SHA verification failed')
    from huggingface_hub import hf_hub_download
    for path in export.iterdir():
        if path.is_file() and path.name != 'policy.pt':
            downloaded=hf_hub_download(REPO,destination+'/'+path.name,revision=commit)
            if sha256_file(downloaded)!=sha256_file(path):
                raise ValueError(f'Remote metadata mismatch: {path.name}')
    write(root/f'published_{label}.json',dict(repository=REPO,path=destination,revision=commit,sha256=digest,verified=True,time=time.time()))


def production(root,lane):
    release=read(root/'release.json');require_sources(release['sources'])
    for peer in ('A','B'):
        evidence=read(root/f'smoke_{peer}_complete.json')
        if evidence['sources']!=release['sources']:raise ValueError('Smoke source mismatch')
        if evidence['config']!=release['config']:raise ValueError('Smoke configuration mismatch')
    config=release['config']
    for label in QUEUES[lane]:
        require_sources(release['sources'])
        if shutil.disk_usage(root).free<100*1024**3:raise OSError('Less than 100 GiB free; training stopped before next experiment')
        out=root/'production'/label
        if out.exists():raise FileExistsError(f'Refusing to overwrite/resume {out}')
        out.parent.mkdir(exist_ok=True)
        write(root/f'lane_{lane}_status.json',dict(stage='training',label=label,started_unix=time.time(),gpu_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),config=config))
        if run_logged(command(label,config,out),root/f'train_{label}.log'):raise RuntimeError(f'Training failed: {label}')
        require_sources(release['sources'])
        verify(label,out)
        env=dict(os.environ,HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0')
        success=False
        for attempt in range(3):
            with (root/f'publish_{label}_{attempt}.log').open('x') as log:
                code=subprocess.run([sys.executable,__file__,'--root',str(root),'--phase','publish','--label',label],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            if code==0:success=True;break
            time.sleep(30)
        if not success:
            write(root/f'upload_pending_{label}.json',dict(label=label,checkpoint=str(out/'latest.pt'),reason='Three upload attempts failed; complete training preserved'))
    write(root/f'lane_{lane}_status.json',dict(stage='complete',finished_unix=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--lane',choices=QUEUES);p.add_argument('--label',choices=GRID)
    p.add_argument('--phase',choices=['benchmark','smoke','run','publish'],required=True)
    a=p.parse_args()
    if a.phase!='publish' and a.lane is None:p.error('--lane required')
    try:
        if a.phase=='benchmark':benchmark(a.root,a.lane)
        elif a.phase=='smoke':smoke(a.root,a.lane)
        elif a.phase=='run':production(a.root,a.lane)
        else:publish(a.label,a.root/'production'/a.label,a.root)
    except BaseException as e:
        write(a.root/f'{a.phase}_{a.lane or a.label}_failed.json',dict(error=repr(e),time=time.time()))
        raise

if __name__=='__main__':main()
