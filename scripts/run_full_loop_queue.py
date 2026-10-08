#!/usr/bin/env python3
"""Two-GPU full LIBERO queue: benchmark, gated fresh train/eval/train/eval."""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_full_libero_v0_job import source_hashes,SUITES
from evaluate_loopwam_pool import write,make_jobs,verify_results
ROOT=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM')
BASE=ROOT/'runs/loopwam_nt/v0_full_libero_job4659_20261006'
CACHE=BASE/'latents'
VAE='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
TORCHRUN=['torchrun','--standalone','--nproc_per_node=2']

def read(p):return json.loads(Path(p).read_text())
def train_command(pair,backend,mb,out,updates=None):
 c=TORCHRUN+['scripts/train_loopwam.py','--version','v0','--video-loops',str(pair[0]),'--action-loops',str(pair[1]),'--backend',backend,'--microbatch',str(mb),'--global-batch','128','--epochs','10','--workers','8','--seed','42','--fused-optimizer','--bucket-views','--structured-attention','--smoke','--dataset-scope','full_libero','--dataset-dir','data/lerobot_v30','--validation-samples','0','--output-dir',str(out),'--latent-cache-dir',str(CACHE),'--save-every','2170']
 if updates is not None:c+=['--max-updates',str(updates)]
 return c

def eval_command(train,out,workers,smoke=False,benchmark=False):
 c=['python','scripts/evaluate_loopwam_pool.py','--checkpoint',str(train/'latest.pt'),'--stats',str(train/'data/dataset_stats.json'),'--vae-path',VAE,'--output-dir',str(out),'--workers-per-gpu',str(workers),'--seeds','42','42','42']
 if smoke:
  c=c[:-4]+['--seeds','42','--smoke','--tasks','0','1','--episodes-per-task','2' if benchmark else '1','--max-steps','100' if benchmark else '30']
  if benchmark:c+=['--trace-actions']
 return c

def select_training(trials):
 valid=[]
 for r in trials:
  if (r.get('global_batch')!=128 or r.get('world_size')!=2 or r.get('train_windows')!=277713 or r.get('dataset_scope')!='full_libero'
      or r.get('precision',{}).get('policy_dtype')!='float32' or r.get('precision',{}).get('optimizer_moment_dtypes')!=['torch.float32']
      or not math.isfinite(r.get('steady_seconds',float('nan'))) or r['steady_seconds']<=0
      or not r.get('records') or any(not math.isfinite(x[k]) for x in r['records'] for k in ('loss','grad_norm'))):continue
  valid.append(r)
 if not valid:raise ValueError('No valid finite full-data FP32 training trial')
 return min(valid,key=lambda r:r['steady_seconds'])

def verify_train(directory,pair,smoke=False):
 m,t,s=(read(directory/(n+'.json')) for n in ('manifest','timing','trainer_state'))
 expected=10 if smoke else 21700
 if (m['resume'] is not None or m['initialization_mode']!='canonical_wan_artifact_fresh_optimizer'
     or m['version']!='v0' or m['loops']!=pair[0] or m['action_core_loops']!=pair[1]
     or m['epochs']!=10 or m['global_batch']!=128 or m['world_size']!=2
     or m['microbatch']*2*m['gradient_accumulation']!=128 or m['policy_parameters']!=584536135
     or m['train_windows']!=277713 or m['val_windows']!=0 or m['planned_updates']!=21700
     or m['planned_windows']!=2777130 or m['dataset_scope']!='full_libero'
     or m['data']['available_episodes']!=1712 or len(m['data']['task_counts'])!=40
     or m['data']['split']!='all_train' or m['data']['suites']!=list(SUITES)
     or m['max_updates']!=(10 if smoke else None)):
  raise ValueError('Fresh full LIBERO training contract mismatch')
 if (t['status']!=('smoke_complete' if smoke else 'complete') or t['completed_updates']!=expected
     or t['windows_seen']!=(1280 if smoke else 2777130) or s['update']!=expected):raise ValueError('Training incomplete')
 if not smoke and (s['epoch']!=9 or s['next_micro']!=math.ceil(277713/(2*m['microbatch']))):raise ValueError('Epoch coverage incomplete')
 if not (directory/'latest.pt').is_file():raise ValueError('Checkpoint missing')
 return t

def compare_traces(reference,candidate):
 import numpy as np
 a={r['id']:r for r in read(reference/'summary.json')['episodes']};b={r['id']:r for r in read(candidate/'summary.json')['episodes']}
 if set(a)!=set(b):raise ValueError('Concurrency benchmark episode coverage changed')
 maximum=0.
 for key,x in a.items():
  y=b[key]
  if any(x[k]!=y[k] for k in ('seed','success','steps','replans','checkpoint_sha256')):raise ValueError('Concurrency changed episode outcome')
  u=np.load(x['trace'])['actions'];v=np.load(y['trace'])['actions']
  if u.shape!=v.shape or not np.allclose(u,v,rtol=0,atol=1e-5):raise ValueError('Concurrency changed actions')
  if u.size:maximum=max(maximum,float(np.max(np.abs(u-v))))
 return maximum

class Queue:
 def __init__(self,args):
  self.a=args;self.out=Path(args.output_root).resolve();self.out.mkdir(parents=True,exist_ok=False)
  self.started=time.time();self.hashes=source_hashes();self.state=dict(started_unix=self.started,job_id=os.getenv('SLURM_JOB_ID'),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),pairs=args.pairs)
 def status(self,stage,**kw):
  self.state.update(stage=stage,updated_unix=time.time(),elapsed_seconds=time.time()-self.started,**kw);write(self.out/'status.json',self.state);print(json.dumps(self.state),flush=True)
 def stage(self,name,command,trial=False):
  if source_hashes()!=self.hashes:raise ValueError('Pinned source changed')
  self.status(name,command=command)
  with (self.out/(name+'.log')).open('x') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
  if result.returncode:
   log=(self.out/(name+'.log')).read_text(errors='replace')
   if trial and ('out of memory' in log.lower()):
    self.status(name+'_oom');return False
   raise RuntimeError(f'{name} failed ({result.returncode}); see log')
  return True
 def prepare(self):
  import numpy as np
  valid=np.memmap(CACHE/'valid.uint8',mode='r',dtype='uint8')
  if len(valid)!=277713 or not bool(np.all(valid==1)):raise ValueError('Full latent cache not complete')
  del valid
  evaluations=[];reference=None
  for workers in (1,2,4,8):
   dest=self.out/f'eval_speed_{workers}'
   if not self.stage(f'eval_speed_{workers}',eval_command(BASE/'train',dest,workers,True,True),trial=True):continue
   result=read(dest/'summary.json')
   if result['total_episodes']!=16:raise ValueError('Benchmark episode count mismatch')
   if reference is None:reference=dest
   result['max_action_difference']=compare_traces(reference,dest)
   result['projected_evaluation_seconds']=result['model_startup_seconds']+result['episode_phase_seconds']*1200/16
   evaluations.append({k:v for k,v in result.items() if k not in ('episodes','rounds')})
   write(self.out/'evaluation_benchmarks.json',evaluations)
  best_eval=min(evaluations,key=lambda r:r['projected_evaluation_seconds'])
  configs={}
  for pair in self.a.pairs:
   label=f'{pair[0]}{pair[1]}';trials=[]
   for backend,mb in [('ddp',8),('ddp',16),('zero1',16),('zero2',16)]:
    dest=self.out/f'speed_{label}_{backend}_{mb}'
    command=TORCHRUN+['scripts/benchmark_loopwam.py','--dataset-scope','full_libero','--version','v0','--video-loops',str(pair[0]),'--action-loops',str(pair[1]),'--backend',backend,'--microbatch',str(mb),'--workers','8','--updates','8','--structured-attention','--latent-cache-dir',str(CACHE),'--output-dir',str(dest)]
    if self.stage(dest.name,command,trial=True):trials.append(read(dest/'result.json'))
   selected=select_training(trials);dest=self.out/f'smoke_{label}'
   self.stage(f'train_smoke_{label}',train_command(pair,selected['backend'],selected['microbatch'],dest,10))
   timing=verify_train(dest,pair,True)
   eval_out=self.out/f'smoke_eval_{label}'
   self.stage(f'eval_smoke_{label}',eval_command(dest,eval_out,best_eval['workers_per_gpu'],True))
   smoke=read(eval_out/'summary.json')
   if smoke['total_episodes']!=8 or smoke['video_loops']!=pair[0] or smoke['action_loops']!=pair[1]:raise ValueError('Model native inference smoke failed')
   configs[label]=dict(pair=pair,backend=selected['backend'],microbatch=selected['microbatch'],steady_seconds=selected['steady_seconds'],native_seconds=timing['measured_mean_update_seconds'],estimated_training_hours=timing['measured_mean_update_seconds']*21700/3600,workers_per_gpu=best_eval['workers_per_gpu'])
   write(self.out/'training_benchmarks.json',configs)
  prepared=dict(source_hashes=self.hashes,configs=configs,evaluation=best_eval,prepared_unix=time.time())
  write(self.out/'prepared.json',prepared);self.status('prepared_awaiting_release',configs=configs,evaluation=best_eval)
  return prepared
 def production(self,prepared):
  # Root audits all GPU results before releasing both jobs. Timeout prevents abandoned reservations.
  deadline=time.time()+7200
  while not (self.out/'release.json').exists():
   if time.time()>deadline:raise TimeoutError('Preflight release timeout')
   time.sleep(5)
  release=read(self.out/'release.json')
  if release.get('source_revision')!=self.state['source_revision'] or release.get('approved_pairs')!=self.a.pairs:raise ValueError('Release/source mismatch')
  for pair in self.a.pairs:
   label=f'{pair[0]}{pair[1]}';c=prepared['configs'][label];train=self.out/f'v0_{label}'/'train';evaluation=train.parent/'evaluation'
   if train.exists() or evaluation.exists():raise ValueError('Refusing checkpoint resume or overwrite')
   self.stage(f'train_{label}',train_command(pair,c['backend'],c['microbatch'],train))
   timing=verify_train(train,pair)
   self.stage(f'evaluate_{label}',eval_command(train,evaluation,c['workers_per_gpu']))
   result=read(evaluation/'summary.json');manifest=read(evaluation/'manifest.json')
   verify_results(make_jobs(SUITES,[42]*3,range(10),10),result['episodes'],manifest['checkpoint_sha256'],'final_rollout')
   if result['checkpoint_step']!=21700 or result['video_loops']!=pair[0] or result['action_loops']!=pair[1] or result['total_episodes']!=1200:raise ValueError('Final evaluation mismatch')
   write(train.parent/'summary.json',dict(training=timing,evaluation={k:v for k,v in result.items() if k!='episodes'}))
  self.status('complete')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--pairs',nargs=2,required=True,choices=['22','41','14','33']);a=p.parse_args();a.pairs=[[int(x[0]),int(x[1])] for x in a.pairs]
 q=Queue(a)
 try:q.production(q.prepare())
 except BaseException as e:q.status('failed',error=repr(e));raise
if __name__=='__main__':main()
