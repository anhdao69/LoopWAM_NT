#!/usr/bin/env python3
"""Resume interrupted full-suite LoopWAM runs and finish their training-only queues."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt')
PLANS={
 'concat':[
  dict(label='full_41_resume',version='v0',video=4,action=1,mode='aligned',scope='full_libero',resume=ROOT/'storage_recovery_20261010/input_checkpoints/full_41_step15190.pt',source=ROOT/'kv_campaign_job4770/campaign_3a89bfa/full_41/train/latest.pt',source_step=15190,config_job=4770,config_label='full_41'),
  dict(label='full_33',version='v0',video=3,action=3,mode='aligned',scope='full_libero',config_job=4770,config_label='full_33'),
  dict(label='full_dense12',version='dense_s12',video=1,action=1,mode='aligned',scope='full_libero',config_job=4770,config_label='full_dense12')],
 'mix':[
  dict(label='full_22_resume',version='v0',video=2,action=2,mode='aligned',scope='full_libero',resume=ROOT/'storage_recovery_20261010/input_checkpoints/full_22_step4340.pt',source=ROOT/'training_only_job4796/full_22/train/latest.pt',source_step=4340,config_job=4771,config_label='full_22'),
  dict(label='full_dense30',version='dense_s30',video=1,action=1,mode='aligned',scope='full_libero',config_job=4771,config_label='full_dense30')],
 'kv_mix':[
  dict(label='full_mix',version='v0',video=4,action=1,mode='mix',scope='full_libero',fresh=True,config_job=None,config_label=None)]}


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def source_hashes(root):
 return {str(p.relative_to(root)):sha(p) for base in ['src','scripts'] for p in sorted((root/base).rglob('*.py'))}
def read_updates(path,limit=None):
 rows=[]
 with Path(path).open() as f:
  for line in f:
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   if 'loss_video' in r:
    rows.append(r)
    if limit is not None and len(rows)>=limit:break
 return rows
def check_updates(rows,first,last):
 ids=[int(r['update']) for r in rows]
 if ids!=list(range(first,last+1)):raise ValueError(f'Expected contiguous updates {first}..{last}; got {len(ids)} ending at {ids[-1] if ids else None}')
 if any(not math.isfinite(float(r[k])) for r in rows for k in ('loss_video','loss_action','grad_norm')):
  raise ValueError('Nonfinite loss or gradient in update log')
def metrics_prefix(path,step):
 rows=read_updates(path,limit=step)
 check_updates(rows,1,step)
 return rows[-1]
def fairness(candidate,baseline):
 fields=('asset_sha256','seed','epochs','global_batch','train_windows','planned_updates','planned_windows','policy_dtype','optimizer_state_dtype','compute_dtype')
 for key in fields:
  if candidate.get(key)!=baseline.get(key):raise ValueError('Resume fairness mismatch: '+key)
 clean=lambda d:{k:v for k,v in d.items() if k!='normalization_path'}
 if clean(candidate['data'])!=clean(baseline['data']):raise ValueError('Resume data/split/normalization mismatch')
 return dict(passed=True,compared_fields=list(fields),data_fields=sorted(clean(candidate['data'])),
             accepted_initialization_change=candidate.get('initialization_mode')=='resume_checkpoint')
def validate_final(path,spec,config,first_update,baseline):
 import torch
 from fastwam.training_backends import expected_policy_parameters
 path=Path(path);m=read(path/'manifest.json');t=read(path/'timing.json');st=read(path/'trainer_state.json')
 total=21700;windows=277713;resume=spec.get('resume')
 expected=dict(version=spec['version'],action_kv_mode=spec['mode'],loops=spec['video'],action_core_loops=spec['action'],
  global_batch=128,epochs=10,seed=42,world_size=2,train_windows=windows,planned_updates=total,planned_windows=windows*10,
  policy_parameters=expected_policy_parameters(spec['version'],spec['mode'],spec['video']),max_updates=None)
 if any(m.get(k)!=v for k,v in expected.items()):raise ValueError(f'Final training contract mismatch: {[(k,m.get(k),v) for k,v in expected.items() if m.get(k)!=v]}')
 if m.get('backend')!='ddp' or m.get('microbatch')!=config['microbatch'] or m.get('gradient_accumulation')*2*config['microbatch']!=128:
  raise ValueError('Final DDP/global batch contract mismatch')
 if m.get('resume')!=(str(resume) if resume else None):raise ValueError('Resume provenance mismatch')
 if t.get('status')!='complete' or t.get('completed_updates')!=total or t.get('windows_seen')!=windows*10:
  raise ValueError('Training did not finish all updates and windows')
 if st.get('update')!=total or st.get('epoch')!=9 or st.get('windows_seen')!=windows*10:
  raise ValueError('Final trainer state incomplete')
 rows=read_updates(path/'metrics.jsonl');check_updates(rows,first_update,total)
 audit=fairness(m,baseline);write(path/'fairness_recovery.json',audit)
 ck=path/'latest.pt'
 if not ck.is_file():raise ValueError('Missing final checkpoint')
 payload=torch.load(ck,map_location='cpu',mmap=True,weights_only=False);state=payload['training_state']
 if (payload.get('step')!=total or payload.get('version')!=spec['version'] or payload.get('video_loops')!=spec['video']
  or payload.get('action_loops')!=spec['action'] or payload.get('action_kv_mode','aligned')!=spec['mode']
  or state.get('update')!=total or state.get('windows_seen')!=windows*10 or state.get('epoch')!=9
  or state.get('next_micro')!=math.ceil(windows/(2*config['microbatch'])) or len(state.get('rng',[]))!=2
  or not payload.get('optimizer',{}).get('state')):raise ValueError('Final checkpoint validation failed')
 if spec.get('mode')=='mix':
  for epoch in range(5,11):
   p=path/f'epoch_{epoch:03d}.pt';e=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
   if (e.get('step')!=epoch*2170 or e.get('action_kv_mode')!='mix'
    or e['training_state'].get('epoch')!=epoch-1 or e['training_state'].get('windows_seen')!=epoch*windows
    or len(e['training_state'].get('rng',[]))!=2 or not e.get('optimizer',{}).get('state')):
    raise ValueError(f'Invalid retained epoch-{epoch} checkpoint')
   if epoch==10 and not __import__('os').path.samefile(p,ck):raise ValueError('latest.pt must point to retained epoch 10')
   del e
 del payload
 return dict(status='complete',completed_updates=total,training_seconds=t['elapsed_training_seconds'],
  seconds_per_update=t['measured_mean_update_seconds'],first_resumed_update=first_update,final_checkpoint_sha256=sha(ck),
  fairness=audit,retained_epochs=list(range(5,11)) if spec.get('mode')=='mix' else [])


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',choices=PLANS,required=True);p.add_argument('--recovery-root',type=Path,required=True)
 a=p.parse_args();from run_kv_campaign import spec as make_spec,train_command,FULL_CACHE
 plan=PLANS[a.plan];base=ROOT/'v0_full_libero_job4659_20261006/train';baseline=read(base/'manifest.json')
 recovery=a.recovery_root/f'job{os.environ["SLURM_JOB_ID"]}_{a.plan}'
 recovery.mkdir(parents=True,exist_ok=False)
 # Validate immutable training source and cache before spending GPU time.
 training_source=Path(os.environ['TRAINING_SOURCE']).resolve()
 pin=dict(source=str(training_source),source_revision=subprocess.check_output(['git','-C',str(training_source),'rev-parse','HEAD'],text=True).strip(),source_hashes=source_hashes(training_source))
 write(recovery/'source.json',pin)
 import numpy as np
 cache=FULL_CACHE
 valid=np.memmap(cache/'valid.uint8',mode='r',dtype=np.uint8)
 if len(valid)!=277713 or not np.all(valid==1):raise ValueError('Full-suite latent cache is incomplete')
 del valid
 for item in plan:
  s=make_spec(item['label'],item['version'],item['video'],item['action'],item['mode'],item['scope'])
  config=item.get('config')
  if config is None:
   if item['config_job']:
    prepared=read(ROOT/f'kv_campaign_job{item["config_job"]}/campaign_3a89bfa/prepared.json')
    config=prepared['configs'][item['config_label']]
   else:config=dict(backend='ddp',microbatch=8)
  if config['backend']!='ddp':raise ValueError('Unverified training backend')
  out=recovery/item['label']/'train';out.parent.mkdir(parents=True,exist_ok=True)
  cmd=train_command(s,config,out,cache)
  if item.get('resume'):
   ck=Path(item['resume']);meta=read(a.recovery_root/'checkpoints'/f'{item["label"]}.json')
   if sha(ck)!=meta['sha256']:raise ValueError('Source recovery checkpoint changed')
   if metrics_prefix(item['source'].parent/'metrics.jsonl',item['source_step'])['update']!=item['source_step']:
    raise ValueError('Original training log does not reach recovery checkpoint')
   cmd+=['--resume',str(ck)]
   first=item['source_step']+1
  else:first=1
  if item.get('mode')=='mix':cmd+=['--retain-epochs-from','5']
  write(recovery/'status.json',dict(stage='training_'+item['label'],command=cmd,started_unix=time.time(),source_step=item.get('source_step')))
  with (recovery/(item['label']+'.log')).open('x') as log:
   subprocess.run(cmd,cwd=training_source,stdout=log,stderr=subprocess.STDOUT,check=True)
  report=validate_final(out,s,config,first,baseline);write(recovery/(item['label']+'_complete.json'),report)
 write(recovery/'status.json',dict(stage='complete',plan=a.plan,completed_unix=time.time()))

if __name__=='__main__':main()
