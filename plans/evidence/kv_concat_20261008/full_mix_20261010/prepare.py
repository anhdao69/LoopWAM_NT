import json, math, os, subprocess, sys, time, hashlib
from pathlib import Path
from run_kv_campaign import spec,train_command,FULL_CACHE,verify_train
ROOT=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/mix_full_20261010')
s=spec('full_mix',mode='mix')
def write(p,v):p.write_text(json.dumps(v,indent=2))
def run(label,cmd,oom=False):
 write(ROOT/'status.json',dict(stage=label,time=time.time()))
 with (ROOT/(label+'.log')).open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:
  if oom and 'out of memory' in (ROOT/(label+'.log')).read_text().lower():return False
  raise RuntimeError(label+' failed')
 return True
try:
 run('checkpoint_tests',[sys.executable,'-m','pytest','tests/test_loopwam_zero_training.py','-q'])
 trials=[]
 for backend in ['ddp','zero1','zero2']:
  for mb in [8,16,32]:
   label=f'benchmark_{backend}_{mb}'
   cmd=['torchrun','--standalone','--nproc_per_node=2','scripts/benchmark_loopwam.py','--version','v0','--video-loops','4','--action-loops','1','--action-kv-mode','mix','--dataset-scope','full_libero','--backend',backend,'--microbatch',str(mb),'--workers','8','--updates','10','--structured-attention','--latent-cache-dir',str(FULL_CACHE),'--output-dir',str(ROOT/label)]
   if not run(label,cmd,oom=True):break
   t=json.loads((ROOT/label/'result.json').read_text())
   assert t['global_batch']==128 and t['train_windows']==277713 and t['world_size']==2
   assert t['precision']['policy_dtype']=='float32' and t['precision']['optimizer_moment_dtypes']==['torch.float32']
   assert all(math.isfinite(r[k]) for r in t['records'] for k in ['loss','grad_norm','seconds'])
   trials.append(t);write(ROOT/'benchmarks.json',trials)
 best=min(trials,key=lambda t:t['steady_seconds'])
 config={k:best[k] for k in ['backend','microbatch']}
 out=ROOT/'native_smoke'
 run('native_smoke',train_command(s,config,out,FULL_CACHE,10)+['--retain-epochs-from','5'])
 timing=verify_train(out,s,True)
 write(ROOT/'prepared.json',dict(spec=s,config=config,benchmark_seconds=best['steady_seconds'],native_timing=timing,training_hours=timing['measured_mean_update_seconds']*21700/3600))
 write(ROOT/'status.json',dict(stage='prepared',time=time.time()))
except Exception as e:
 write(ROOT/'failed.json',dict(error=repr(e),time=time.time()));raise
