import json, math, subprocess, time
from pathlib import Path
from run_kv_campaign import train_command,FULL_CACHE,verify_train
r=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/mix_full_20261010')
def write(p,x):p.write_text(json.dumps(x,indent=2))
original=json.loads((r/'prepared.json').read_text())
trials=json.loads((r/'benchmarks.json').read_text())
for mb in [16,32,64]:
 label=f'benchmark_ddp_checkpoint_{mb}'
 write(r/'status.json',dict(stage=label,time=time.time()))
 cmd=['torchrun','--standalone','--nproc_per_node=2','scripts/benchmark_loopwam.py','--version','v0','--video-loops','4','--action-loops','1','--action-kv-mode','mix','--dataset-scope','full_libero','--backend','ddp','--microbatch',str(mb),'--workers','8','--updates','10','--structured-attention','--checkpoint-blocks','--latent-cache-dir',str(FULL_CACHE),'--output-dir',str(r/label)]
 with (r/(label+'.log')).open('x') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:
  if 'out of memory' in (r/(label+'.log')).read_text().lower():break
  raise RuntimeError(label+' failed')
 t=json.loads((r/label/'result.json').read_text())
 assert t['global_batch']==128 and t['train_windows']==277713 and t['world_size']==2
 assert t['precision']['policy_dtype']=='float32' and t['precision']['optimizer_moment_dtypes']==['torch.float32']
 assert all(math.isfinite(row[k]) for row in t['records'] for k in ['loss','grad_norm','seconds'])
 trials.append(t)
write(r/'all_benchmarks.json',trials)
best=min(trials,key=lambda t:t['steady_seconds'])
if best['checkpoint_blocks']:
 config={k:best[k] for k in ['backend','microbatch','checkpoint_blocks']}
 out=r/'native_smoke_checkpoint'
 write(r/'status.json',dict(stage='native_smoke_checkpoint',time=time.time()))
 with (r/'native_smoke_checkpoint.log').open('x') as f:subprocess.run(train_command(original['spec'],config,out,FULL_CACHE,10)+['--retain-epochs-from','5'],stdout=f,stderr=subprocess.STDOUT,check=True)
 timing=verify_train(out,original['spec'],True)
 (r/'prepared.json').rename(r/'prepared_without_checkpoint_trials.json')
 write(r/'prepared.json',dict(spec=original['spec'],config=config,benchmark_seconds=best['steady_seconds'],native_timing=timing,training_hours=timing['measured_mean_update_seconds']*21700/3600))
write(r/'status.json',dict(stage='prepared_all_trials',time=time.time()))
