"""Measured fresh eight-GPU batch sweep; no training checkpoint is resumed."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root)
shared='/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin50_20261025/shared/robotwin50'
base=['torchrun','--standalone','--nproc_per_node=8','scripts/train_loopwam_robotwin_full.py','--dataset','robotwin50','--run-name','LoopWAM-v0-robotwin-full',
 '--init-artifact','checkpoints/LoopWAM/wan21_compact_donors.pt','--shared-dir',shared,'--epochs','5','--global-batch','128',
 '--seed','42','--lr-backbone','0.0001','--lr-robot-io','0.0001','--warmup-fraction','0.05','--save-epochs','3,4,5','--text-threads','4']
results=[]
def trial(name,micro,cp,compile):
 out=r/name
 if out.exists():raise FileExistsError(out)
 command=base+['--output-dir',str(out),'--microbatch',str(micro),'--checkpoint-blocks' if cp else '--no-checkpoint-blocks','--compile' if compile else '--no-compile','--max-updates','45' if compile else '30','--no-save']
 (r/(name+'.command.json')).write_text(json.dumps(command,indent=2))
 with (r/(name+'.log')).open('x') as log:code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
 entry=dict(name=name,micro=micro,checkpoint_blocks=cp,compile=compile)
 if code:
  if 'out of memory' not in (r/(name+'.log')).read_text().lower():raise RuntimeError('Non-OOM benchmark failure '+name)
  entry['status']='oom'
 else:
  import math
  m=json.loads((out/'manifest.json').read_text());t=json.loads((out/'timing.json').read_text())
  assert m['world_size']==8 and m['memory_parameters']==0 and m['policy_parameters']==584567886 and m['resumed_from_update'] is None
  assert t['status']=='max_updates'
  rows=[json.loads(x) for x in (out/'metrics.jsonl').read_text().splitlines()];rows=[x for x in rows if 'loss' in x]
  assert len(rows)==(45 if compile else 30) and all(math.isfinite(x[k]) for x in rows for k in ('loss','grad_norm','seconds'))
  steady=rows[15:];entry.update(status='passed',seconds=sum(x['seconds'] for x in steady)/len(steady),windows_per_second=sum(x['valid_windows'] for x in steady)/sum(x['seconds'] for x in steady),peak_gb=max(x['peak_allocated_gb'] for x in rows),planned_updates=t['planned_updates'])
 results.append(entry);(r/'bench_results.json').write_text(json.dumps(results,indent=2));print(json.dumps(entry),flush=True)
for micro,cp in [(4,False),(8,False),(16,True)]:trial(f'bench_m{micro}_cp{int(cp)}_eager',micro,cp,False)
best=min((x for x in results if x['status']=='passed'),key=lambda x:x['seconds'])
trial('bench_compiled',best['micro'],best['checkpoint_blocks'],True)
best=min((x for x in results if x['status']=='passed'),key=lambda x:x['seconds'])
(r/'selected.json').write_text(json.dumps(best,indent=2));print('SELECTED',best,flush=True)
