"""Selected-layout native gates and original-v0 production launcher."""
import argparse,json,math,subprocess,sys,time,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',choices=['smoke','production'],required=True);a=p.parse_args();r=Path(a.root)
read=lambda p:json.loads(p.read_text())
def source_hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in ('src','scripts') for p in sorted(Path(root).rglob('*.py'))}
sources=source_hashes()
selected=read(r/'exact_batch/selected.json')
shared='/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin50_20261025/shared/robotwin50'
base=['torchrun','--standalone','--nproc_per_node=8','scripts/train_loopwam_robotwin_full.py','--dataset','robotwin50','--run-name','LoopWAM-v0-full-RoboTwin','--init-artifact','checkpoints/LoopWAM/wan21_compact_donors.pt','--shared-dir',shared,'--epochs','5','--global-batch','128','--seed','42','--lr-backbone','0.0001','--lr-robot-io','0.0001','--warmup-fraction','0.05','--save-epochs','3,4,5','--text-threads','4','--microbatch',str(selected['micro']),'--checkpoint-blocks' if selected['checkpoint_blocks'] else '--no-checkpoint-blocks','--compile' if selected['compile'] else '--no-compile']
def run(name,cmd):
 (r/(name+'.command.json')).write_text(json.dumps(cmd,indent=2))
 with (r/(name+'.log')).open('x') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
 if code:raise RuntimeError(name+' failed; see log')
def metrics(path):return [json.loads(l) for l in (path/'metrics.jsonl').read_text().splitlines() if l.strip()]
if a.phase=='smoke':
 run('native_smoke',base+['--output-dir',str(r/'smoke'),'--max-updates','10'])
 run('native_reload',[sys.executable,'scripts/check_loopwam_robotwin_checkpoint.py','--checkpoint',str(r/'smoke/latest.pt'),'--shared',shared,'--output',str(r/'reload_check.json')])
 run('native_resume',base+['--output-dir',str(r/'smoke'),'--resume','--max-updates','12'])
 rows=metrics(r/'smoke');assert [x['update'] for x in rows]==list(range(1,13)) and all(x['valid_windows']==128 for x in rows)
 reference=metrics(r/'exact_batch'/selected['name'])
 drift=[]
 for x,y in zip(rows,reference):
  for k in ('loss_video','loss_action'):
   delta=abs(x[k]-y[k]);drift.append(dict(update=x['update'],metric=k,absolute=delta,relative=delta/max(abs(y[k]),1e-12)))
   assert delta<=1e-4+1e-3*abs(y[k]),(x['update'],k,x[k],y[k])
 (r/'independent_run_drift.json').write_text(json.dumps(drift,indent=2))
 assert read(r/'smoke/resume_audit_u10.json')['optimizer_exact']
 assert read(r/'smoke/timing.json')['windows_seen']==1536
 run('native_tail',base+['--output-dir',str(r/'tail_smoke'),'--smoke-tail','--max-updates','1','--no-save'])
 tail=metrics(r/'tail_smoke');assert len(tail)==1 and tail[0]['valid_windows']==95 and tail[0]['epochs_complete']==1
 assert read(r/'reload_check.json')['passed']
 assert source_hashes()==sources,'Source changed during smoke'
 (r/'gates_passed.json').write_text(json.dumps(dict(passed=True,sources=sources,selected=selected,updates=12,optimizer_restoration_exact=True,resume_cursor_verified=True,independent_run_numerically_consistent=True,tail_real_windows=95,time=time.time()),indent=2))
else:
 assert read(r/'gates_passed.json')['sources']==sources
 assert '328 passed, 17 skipped' in (r/'gpu_final_tests.log').read_text()
 assert '321 passed, 24 skipped' in (r/'cpu_exact_tests.log').read_text()
 assert read(r/'gates_passed.json')['passed'] and read(r/'gates_passed.json')['selected']==selected
 assert read(r/'exact_audit/audit.json')['passed']
 assert not (r/'production').exists()
 run('production',base+['--output-dir',str(r/'production')])
 t=read(r/'production/timing.json');assert t['status']=='complete' and t['update']==237310 and t['windows_seen']==30375515
 for epoch in (3,4,5):assert (r/'production/checkpoints'/f'epoch_{epoch:02d}.pt').is_file()
 (r/'verified_complete.json').write_text(json.dumps(dict(passed=True,timing=t,completed_unix=time.time()),indent=2))
