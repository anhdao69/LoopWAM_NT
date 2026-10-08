#!/usr/bin/env python3
"""Gated two-H100 queue: Long KV experiment, two full loops, one full dense control."""
from __future__ import annotations
import argparse,json,math,os,subprocess,time
from pathlib import Path
from run_full_libero_v0_job import source_hashes,SUITES
from evaluate_loopwam_pool import write,make_jobs,verify_results
from fastwam.training_backends import expected_policy_parameters
ROOT=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt')
LONG_BASE=ROOT/'v0_v4a1_job4728_20261007/train'
FULL_BASE=ROOT/'v0_full_libero_job4659_20261006/train'
LONG_CACHE=ROOT/'dense_s30_job4719_20261006/production_latents'
FULL_CACHE=FULL_BASE.parent/'latents'
VAE='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
TORCHRUN=['torchrun','--standalone','--nproc_per_node=2']

def spec(label,version='v0',v=4,a=1,mode='aligned',scope='full_libero'):
 return dict(label=label,version=version,video=v,action=a,mode=mode,scope=scope)
QUEUES={
 'concat':[spec('concat_long',mode='concat',scope='long_split'),spec('full_41'),spec('full_33',v=3,a=3),spec('full_dense12','dense_s12',1,1)],
 'mix':[spec('mix_long',mode='mix',scope='long_split'),spec('full_14',v=1,a=4),spec('full_22',v=2,a=2),spec('full_dense30','dense_s30',1,1)]}

def read(p):return json.loads(Path(p).read_text())
def windows(s):return 92678 if s['scope']=='long_split' else 277713
def updates(s):return math.ceil(windows(s)/128)*10
def cache_for(s):return LONG_CACHE if s['scope']=='long_split' else FULL_CACHE

def train_command(s,c,out,cache,max_updates=None):
 command=TORCHRUN+['scripts/train_loopwam.py','--version',s['version'],'--video-loops',str(s['video']),'--action-loops',str(s['action']),'--action-kv-mode',s['mode'],'--backend',c['backend'],'--microbatch',str(c['microbatch']),'--global-batch','128','--epochs','10','--seed','42','--workers','4' if s['scope']=='long_split' else '8','--fused-optimizer','--bucket-views','--structured-attention','--smoke','--dataset-scope',s['scope'],'--dataset-dir','data/lerobot_v30/libero_10_no_noops_lerobot' if s['scope']=='long_split' else 'data/lerobot_v30','--output-dir',str(out),'--latent-cache-dir',str(cache),'--save-every',str(updates(s)//10),'--validation-samples','8' if s['scope']=='long_split' and max_updates is None else '0']
 if max_updates is not None:command+=['--max-updates',str(max_updates)]
 if c.get('checkpoint_blocks'):command+=['--checkpoint-blocks']
 return command

def eval_command(s,train,out,workers,seed,smoke=False,episodes=10):
 train=Path(train)
 command=['python','scripts/evaluate_loopwam_pool.py','--checkpoint',str(train/'latest.pt'),'--stats',str(train/'data/dataset_stats.json'),'--vae-path',VAE,'--output-dir',str(out),'--workers-per-gpu',str(workers),'--seeds',str(seed),'--suites',*(['libero_10'] if s['scope']=='long_split' else SUITES),'--episodes-per-task',str(1 if smoke else episodes)]
 if smoke:command+=['--smoke','--tasks','0','1','--max-steps','100']
 return command

def verify_fairness(candidate,baseline):
 fields=('asset_sha256','seed','epochs','global_batch','train_windows','planned_updates','planned_windows','policy_dtype','optimizer_state_dtype','compute_dtype','initialization_mode')
 for key in fields:
  if candidate.get(key)!=baseline.get(key):raise ValueError(f'Fairness mismatch: {key}')
 # Output-local normalization path is the only data-manifest exception.
 clean=lambda d:{k:v for k,v in d.items() if k!='normalization_path'}
 if clean(candidate['data'])!=clean(baseline['data']):raise ValueError('Fairness mismatch: data/split/content/normalization/text assets')
 return dict(passed=True,compared_fields=list(fields),data_fields=sorted(clean(candidate['data'])),ignored_data_fields=['normalization_path'])

def select_candidate(trials,match_baseline=False):
 if not trials:raise ValueError('No valid benchmark candidates')
 if match_baseline:
  matched=[r for r in trials if r['backend']=='ddp' and r['microbatch']==8]
  if matched:return matched[0]
 return min(trials,key=lambda r:r['steady_seconds'])

def verify_train(path,s,smoke=False):
 path=Path(path);m,t,state=(read(path/(n+'.json')) for n in ('manifest','timing','trainer_state'))
 expected=10 if smoke else updates(s);n=windows(s)
 required=dict(version=s['version'],action_kv_mode=s['mode'],loops=s['video'],action_core_loops=s['action'],global_batch=128,epochs=10,seed=42,world_size=2,train_windows=n,planned_updates=updates(s),planned_windows=n*10,policy_parameters=expected_policy_parameters(s['version'],s['mode'],s['video']),resume=None,max_updates=10 if smoke else None)
 if any(m.get(k)!=v for k,v in required.items()):raise ValueError('Training manifest contract mismatch')
 if m['microbatch']*2*m['gradient_accumulation']!=128:raise ValueError('Global batch mismatch')
 fairness=verify_fairness(m,read((LONG_BASE if s['scope']=='long_split' else FULL_BASE)/'manifest.json'))
 if t['status']!=('smoke_complete' if smoke else 'complete') or t['completed_updates']!=expected or state['update']!=expected or t['windows_seen']!=(1280 if smoke else n*10):raise ValueError('Incomplete training')
 if not smoke and (state['epoch']!=9 or state['next_micro']!=math.ceil(n/(2*m['microbatch']))):raise ValueError('Incomplete last epoch')
 if not (path/'latest.pt').is_file():raise ValueError('Missing checkpoint')
 rows=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
 update_rows=[r for r in rows if 'loss_video' in r]
 if len(update_rows)!=expected or any(not math.isfinite(r[k]) for r in update_rows for k in ('loss_video','loss_action','grad_norm')):raise ValueError('Nonfinite/missing training updates')
 write(path/'fairness.json',fairness)
 return t

def verify_eval(path,s,seed,smoke=False,episodes=10):
 path=Path(path);summary=read(path/'summary.json');manifest=read(path/'manifest.json')
 suites=['libero_10'] if s['scope']=='long_split' else SUITES
 grid=make_jobs(suites,[seed],range(2) if smoke else range(10),1 if smoke else episodes)
 verify_results(grid,summary['episodes'],manifest['checkpoint_sha256'],'smoke' if smoke else 'final_rollout')
 if any(summary.get(k)!=v for k,v in dict(version=s['version'],video_loops=s['video'],action_loops=s['action'],action_kv_mode=s['mode'],checkpoint_step=10 if smoke else updates(s)).items()):raise ValueError('Evaluation model contract mismatch')
 return summary

class Campaign:
 def __init__(self,args):
  self.a=args;self.out=Path(args.output_root);self.peer=Path(args.peer_root);self.queue=QUEUES[args.queue]
  self.hashes=source_hashes();self.revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
  self.started=time.time();self.out.mkdir(parents=True,exist_ok=True)
 def status(self,stage,**kw):
  state=dict(stage=stage,queue=self.a.queue,job=os.getenv('SLURM_JOB_ID'),source_revision=self.revision,updated_unix=time.time(),elapsed_seconds=time.time()-self.started,**kw)
  write(self.out/'status.json',state);print(json.dumps(state),flush=True)
 def run(self,label,command,oom_trial=False,env=None):
  if source_hashes()!=self.hashes:raise ValueError('Source changed after pipeline start')
  self.status(label,command=command)
  log=self.out/(label+'.log')
  with log.open('x') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
  if result.returncode:
   if oom_trial and 'out of memory' in log.read_text(errors='replace').lower():return False
   raise RuntimeError(f'{label} failed; inspect {log}')
  return True
 def prepare(self):
  if (self.out/'prepared.json').exists():raise ValueError('Refusing to overwrite preflight')
  import numpy as np
  for s in self.queue:
   valid=np.memmap(cache_for(s)/'valid.uint8',mode='r',dtype='uint8')
   if len(valid)!=windows(s) or not np.all(valid==1):raise ValueError('Production latent cache incomplete')
   del valid
  configs={}
  for s in self.queue:
   label=s['label'];trials=[]
   candidates=[('ddp',8),('ddp',16),('zero1',16),('zero2',16)]
   if s['version']=='dense_s30':candidates=[('ddp',8),('zero1',8),('zero2',8)]
   if s['version']=='dense_s12':candidates=[('ddp',16),('ddp',32),('zero1',32),('zero2',32)]
   for backend,mb in candidates:
    dest=self.out/f'speed_{label}_{backend}_{mb}'
    c=TORCHRUN+['scripts/benchmark_loopwam.py','--version',s['version'],'--video-loops',str(s['video']),'--action-loops',str(s['action']),'--action-kv-mode',s['mode'],'--dataset-scope',s['scope'],'--backend',backend,'--microbatch',str(mb),'--workers','4' if s['scope']=='long_split' else '8','--updates','8','--structured-attention','--latent-cache-dir',str(cache_for(s)),'--output-dir',str(dest)]
    if self.run(dest.name,c,oom_trial=True):
     result=read(dest/'result.json')
     if (result['global_batch']!=128 or result['world_size']!=2 or result['train_windows']!=windows(s) or result['precision'].get('policy_dtype')!='float32' or result['precision'].get('optimizer_moment_dtypes')!=['torch.float32'] or not math.isfinite(result['steady_seconds']) or any(not math.isfinite(r[k]) for r in result['records'] for k in ('loss','grad_norm'))):raise ValueError('Benchmark precision/finite gate failed')
     trials.append(result)
   selected=select_candidate(trials,s['mode']!='aligned')
   c={k:selected[k] for k in ('backend','microbatch')};c.update(workers_per_gpu=8,steady_seconds=selected['steady_seconds'])
   write(self.out/f'{label}_benchmarks.json',dict(trials=trials,selected=c,selection_reason='Match baseline DDP8 when feasible' if s['mode']!='aligned' else 'Lowest measured steady update time'))
   # Fresh cold and warm native runs use the identical 1280-window prefix.
   phases=('cold','warm') if s['mode']!='aligned' else ('warm',)
   for phase in phases:
    dest=self.out/f'{label}_smoke_{phase}'
    cache=self.out/f'{label}_smoke_cache' if s['mode']!='aligned' else cache_for(s)
    self.run(dest.name,train_command(s,c,dest,cache,10))
    timing=verify_train(dest,s,True)
   c['native_seconds']=timing['measured_mean_update_seconds'];c['training_hours']=c['native_seconds']*updates(s)/3600
   # Diagnostic smoke uses the native ten-update checkpoint without updating it.
   if s['mode']!='aligned':
    self.run(label+'_diagnostics',['python','scripts/smoke_kv_diagnostics.py','--checkpoint',str(dest/'latest.pt'),'--output',str(self.out/f'{label}_diagnostics.json')])
   eval_out=self.out/f'{label}_smoke_eval'
   self.run(eval_out.name,eval_command(s,dest,eval_out,c['workers_per_gpu'],42,True),oom_trial=False)
   summary=verify_eval(eval_out,s,42,True)
   c['eval_smoke_seconds']=summary['elapsed_seconds'];c['eval_peak_worker_gib']=summary['max_worker_gpu_gib']
   configs[label]=c;write(self.out/'configs_partial.json',configs)
  prepared=dict(queue=self.a.queue,configs=configs,source_revision=self.revision,source_hashes=self.hashes,prepared_unix=time.time())
  write(self.out/'prepared.json',prepared);self.status('prepared_awaiting_release',configs=configs)
 def wait_file(self,path,timeout):
  deadline=time.time()+timeout
  while not path.exists():
   if time.time()>deadline:raise TimeoutError(f'Waiting for {path}')
   peer_status=self.peer/'status.json'
   if peer_status.exists() and read(peer_status)['stage']=='failed':raise RuntimeError('Peer campaign failed')
   time.sleep(10)
  return read(path)
 def production(self):
  prepared=read(self.out/'prepared.json');release=read(self.out/'release.json')
  if prepared['source_hashes']!=self.hashes or prepared['source_revision']!=self.revision or release.get('source_revision')!=self.revision or release.get('queue')!=self.a.queue:raise ValueError('Source/release mismatch')
  for index,s in enumerate(self.queue):
   c=prepared['configs'][s['label']];run=self.out/s['label'];train=run/'train'
   if run.exists():raise ValueError('Refusing production resume/overwrite')
   self.run('train_'+s['label'],train_command(s,c,train,cache_for(s)))
   timing=verify_train(train,s)
   summaries=[]
   for seed in (42,43,44):
    dest=run/f'eval_seed{seed}';self.run(f'eval_{s["label"]}_{seed}',eval_command(s,train,dest,c['workers_per_gpu'],seed))
    summary=verify_eval(dest,s,seed);summaries.append({k:v for k,v in summary.items() if k!='episodes'})
   write(run/'summary.json',dict(training=timing,evaluations=summaries))
   if index==0:
    # Latency belongs to the experiment; one allocated GPU, no simulator competition.
    env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=env['CUDA_VISIBLE_DEVICES'].split(',')[0]
    self.run(s['label']+'_latency_eager',['python','scripts/measure_policy_latency.py','--loop-checkpoint',str(train/'latest.pt'),'--output',str(run/'latency_eager.json'),'--trials','2','--samples','50','--warmup','5'],env=env)
    if release.get('native_compiled_verified'):
     self.run(s['label']+'_latency_compiled',['python','scripts/measure_policy_latency.py','--loop-checkpoint',str(train/'latest.pt'),'--output',str(run/'latency_compiled.json'),'--trials','2','--samples','50','--warmup','5','--compiled'],env=env)
    write(self.out/'long_complete.json',dict(source_revision=self.revision,run=str(run)))
    self.status('waiting_for_both_long_experiments')
    peer=self.wait_file(self.peer/'long_complete.json',18*3600)
    if peer['source_revision']!=self.revision:raise ValueError('Peer source mismatch')
    if self.a.queue=='concat':
     self.run('long_analysis',['python','scripts/report_kv_campaign.py','--concat-root',str(self.out),'--mix-root',str(self.peer),'--output',str(self.out/'long_analysis')])
     result=read(self.out/'long_analysis/evidence.json')
     if result['decision']=='inconclusive':
      for label,training in [('concat',train),('aligned',LONG_BASE)]:
       dest=self.out/f'expanded_{label}_seed42'
       self.run(dest.name,eval_command(dict(s,mode='concat' if label=='concat' else 'aligned'),training,dest,c['workers_per_gpu'],42,episodes=50))
       verify_eval(dest,dict(s,mode='concat' if label=='concat' else 'aligned'),42,episodes=50)
      self.run('expanded_analysis',['python','scripts/report_kv_followup.py','--concat-root',str(self.out),'--expanded'])
     write(self.out/'long_analysis_complete.json',dict(source_revision=self.revision))
    else:self.wait_file(self.peer/'long_analysis_complete.json',18*3600)
  write(self.out/'queue_complete.json',dict(source_revision=self.revision))
  if self.a.queue=='concat':
   self.status('waiting_for_full_suite_peer')
   peer=self.wait_file(self.peer/'queue_complete.json',36*3600)
   if peer['source_revision']!=self.revision:raise ValueError('Peer final source mismatch')
   self.run('full_analysis',['python','scripts/report_kv_followup.py','--concat-root',str(self.out),'--mix-root',str(self.peer)])
  self.status('complete')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--queue',choices=QUEUES,required=True);p.add_argument('--output-root',required=True);p.add_argument('--peer-root',required=True);p.add_argument('--phase',choices=['prepare','run'],required=True);p.add_argument('--action-kv-mode',choices=['concat','mix'])
 args=p.parse_args()
 if args.action_kv_mode is not None and args.action_kv_mode!=args.queue:p.error('action-kv-mode must match the first queue experiment')
 campaign=Campaign(args)
 try:campaign.prepare() if args.phase=='prepare' else campaign.production()
 except BaseException as error:campaign.status('failed',error=repr(error));raise
if __name__=='__main__':main()
