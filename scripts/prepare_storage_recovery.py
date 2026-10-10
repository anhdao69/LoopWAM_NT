#!/usr/bin/env python3
"""Hash and stage immutable DDP recovery checkpoints on the original run filesystem."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

R=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt')
OUT=R/'storage_recovery_20261010'
BASE=R/'v0_full_libero_job4659_20261006/train'
SOURCES={
 'full_41_resume':(R/'kv_campaign_job4770/campaign_3a89bfa/full_41/train',15190,4,1,8),
 'full_22_resume':(R/'training_only_job4796/full_22/train',4340,2,2,16),
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def readprefix(p,step):
 rows=[]
 with Path(p).open() as f:
  for line in f:
   try:r=json.loads(line)
   except json.JSONDecodeError:continue
   if 'loss_video' not in r:continue
   rows.append(r)
   if len(rows)==step:break
 if len(rows)!=step or [int(r['update']) for r in rows]!=list(range(1,step+1)):
  raise ValueError(f'Incomplete update log through checkpoint step {step}: {len(rows)}')
 if any(not math.isfinite(float(row[k])) for row in rows for k in ['loss_video','loss_action','grad_norm']):
  raise ValueError(f'Nonfinite training record through checkpoint step {step}')
 return rows[-1]
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def main():
 import torch
 if OUT.exists():raise FileExistsError(f'Refusing to overwrite recovery root: {OUT}')
 usage=shutil.disk_usage(R)
 minimum=500*1024**3
 if usage.free<minimum:raise OSError(f'/mnt/data needs at least 500 GiB free; has {usage.free/1024**3:.1f} GiB')
 baseline=read(BASE/'manifest.json')
 # The common full-suite reference was trained on four GPUs; resume sources
 # are the two-GPU experiments. Validate the shared data/optimization budget,
 # and validate each resume source's world size independently below.
 if not all(baseline[k]==v for k,v in dict(dataset_scope='full_libero',global_batch=128,epochs=10,seed=42,train_windows=277713,planned_updates=21700).items()):
  raise ValueError('Baseline is not the expected full-suite training contract')
 cache=R/'v0_full_libero_job4659_20261006/latents'
 valid=__import__('numpy').memmap(cache/'valid.uint8',mode='r',dtype='uint8')
 if len(valid)!=277713 or not bool(valid.all()):raise ValueError('Frozen full-suite latent cache is incomplete')
 del valid
 OUT.mkdir()
 inputs=OUT/'input_checkpoints';inputs.mkdir()
 records={}
 for label,(train,step,video,action,micro) in SOURCES.items():
  manifest=read(train/'manifest.json');timing=read(train/'timing.json');state=read(train/'trainer_state.json')
  completed_epochs=step//2170
  expected_windows=completed_epochs*277713
  if step%2170:raise ValueError(f'{label}: recovery checkpoint must be at an epoch boundary')
  expected=dict(version='v0',action_kv_mode='aligned',loops=video,action_core_loops=action,microbatch=micro,
    global_batch=128,seed=42,epochs=10,world_size=2,train_windows=277713,planned_updates=21700,backend='ddp',policy_dtype='float32',optimizer_state_dtype='float32',compute_dtype='bfloat16')
  if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError(f'{label}: source manifest mismatch')
  if state.get('update')!=step or state.get('windows_seen')!=expected_windows or state.get('epoch')!=completed_epochs-1 or state.get('next_micro')!=math.ceil(277713/(2*micro)) or state.get('backend')!='ddp':
   raise ValueError(f'{label}: checkpoint trainer state mismatch')
  if not readprefix(train/'metrics.jsonl',step):raise ValueError(f'{label}: no metrics prefix')
  source=train/'latest.pt';dest=inputs/f'{label}.pt';src_sha=sha(source)
  if dest.exists():raise FileExistsError(dest)
  shutil.copyfile(source,dest)
  cp_sha=sha(dest)
  if cp_sha!=src_sha:raise IOError(f'{label}: checkpoint copy hash mismatch')
  payload=torch.load(dest,map_location='cpu',mmap=True,weights_only=False);saved=payload['training_state']
  if (payload.get('step')!=step or payload.get('version')!='v0' or payload.get('video_loops')!=video
   or payload.get('action_loops')!=action or payload.get('action_kv_mode','aligned')!='aligned'
   or saved.get('epoch')!=state['epoch'] or saved.get('next_micro')!=state['next_micro']
   or saved.get('windows_seen')!=expected_windows or saved.get('epoch')!=completed_epochs-1 or saved.get('next_micro')!=math.ceil(277713/(2*micro)) or len(saved.get('rng',[]))!=2 or not payload.get('optimizer',{}).get('state')):
   raise ValueError(f'{label}: checkpoint payload invalid')
  records[label]=dict(original_train=str(train),source_sha256=src_sha,staged_checkpoint=str(dest),staged_sha256=cp_sha,
   checkpoint_bytes=dest.stat().st_size,step=step,epoch=state['epoch'],next_micro=state['next_micro'],
   windows_seen=state['windows_seen'],video_loops=video,action_loops=action,microbatch=micro,
   optimizer_parameter_states=len(payload['optimizer']['state']),metrics_through_checkpoint=step)
  del payload
 write(OUT/'recovery_inputs.json',dict(created_unix=time.time(),baseline_manifest=str(BASE/'manifest.json'),
  baseline_normalization_sha256=baseline['data']['normalization_sha256'],baseline_assets=baseline['asset_sha256'],
  full_cache=str(cache),full_cache_entries=277713,checkpoints=records))
 free=shutil.disk_usage(R).free
 if free<minimum:raise OSError(f'Checkpoint staging used space; /mnt/data now has {free/1024**3:.1f} GiB free')
 print(json.dumps(dict(output=str(OUT),free_bytes=free,checkpoints=records),indent=2))
if __name__=='__main__':main()
