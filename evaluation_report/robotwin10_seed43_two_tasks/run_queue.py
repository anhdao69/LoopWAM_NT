"""Run the two requested tasks with seed43, preserving the frozen scoring protocol."""
import concurrent.futures,csv,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
Q=Path('/data14t/anhdh35/loopwam_robotwin10_seed43_two_tasks_20261103')
R=Path('/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_seed43_two_tasks')
W=Path('/storage/anhdh35/LoopWAM-robotwin10-dense-eval')
CONFIG=json.loads((Q/'queue_config.json').read_text())
THRESHOLDS={'v0_4x4':{1:30000,2:42000,4:60000},'dense_s30':{1:36000,2:48000,3:60000}}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def write(path,data):
 p=Path(path);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)

def status(stage,**values):
 d=dict(stage=stage,queue_pid=os.getpid(),updated_unix=time.time(),**values)
 write(Q/'queue_status.json',d);write(R/'queue_status.json',d);print(json.dumps(d),flush=True)

def gpu():
 raw=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 a,b,c,d=map(int,raw.strip().split(','));return dict(total_mib=a,used_mib=b,free_mib=c,utilization=d)

def wait_memory(required,model):
 stable=0
 while stable<3:
  g=gpu();stable=stable+1 if g['free_mib']>=required else 0
  status('waiting_for_memory',model=model,required_free_mib=required,stable_samples=stable,gpu=g)
  if stable<3:time.sleep(10)

def verify_and_bind():
 checked={};reports=[]
 def checked_hash(path):
  if path not in checked:checked[path]=sha(path)
  return checked[path]
 for m in CONFIG['models']:
  b=Path(m['bulk']);old=Path(m['prior_bulk']);r=Path(m['report'])
  assert sha(b/'checkpoint/policy.pt')==m['checkpoint_sha256']
  assert json.loads((b/'cpu_validation.json').read_text())['strict_load']
  gate=json.loads((b/'graph_gate.json').read_text());assert gate['graph_passed'] and gate['max_abs_difference']==0 and gate['checkpoint_sha256']==m['checkpoint_sha256']
  assert "42+i%3" in (old/'scripts/graph_gate.py').read_text(),'Graph validation must cover seed43'
  assert json.loads((Path(m['prior_report'])/'final_audit.json').read_text())['status']=='passed'
  runtime=json.loads((old/'runtime_manifest.json').read_text())
  for path,info in runtime['files'].items():assert checked_hash(path)==info['sha256'],path
  prior=json.loads((old/'results/manifest.json').read_text())
  for path,expected in prior['runtime_hashes'].items():assert checked_hash(path)==expected,path
  protocol=json.loads((b/'planned_protocol.json').read_text());assert protocol['seed']==43 and protocol['tasks']==CONFIG['tasks'] and protocol['episodes']==100
  compare={}
  for key in ['settings','episodes','instruction_type','num_inference_steps','action_horizon','replan_steps','noise_seed','image_preprocessing','instruction_rng','cuda_graph','compile_action_infer','renderer_shader','stats_sha256','vae_sha256']:
   compare[key]=dict(baseline=prior[key],rerun=protocol[key],equal=prior[key]==protocol[key]);assert compare[key]['equal'],key
  write(r/'protocol_comparison.json',compare)
  files=dict(runtime['files'])
  additional=list((b/'scripts').glob('*.py'))+[b/'environment.sh',b/'launch.sh',b/'finalize_results.py',b/'planned_protocol.json',Q/'run_queue.py',Q/'queue_config.json',Q/'start_queue.sh']
  for path in additional:files[str(path)]=dict(sha256=sha(path),size=path.stat().st_size)
  target=b/'runtime_manifest.json'
  if target.exists():assert json.loads(target.read_text())['files']==files,'Runtime changed during resume'
  else:write(target,dict(created_unix=time.time(),files=files,model=m['variant'],seed=43,source_runtime_manifest=str(old/'runtime_manifest.json'),inherited_runtime=True))
  evidence=dict(checked_unix=time.time(),checkpoint_sha256=m['checkpoint_sha256'],seed=43,graph_evidence_reused=True,graph_gate_seeds=[42,43,44],strict_load_evidence_reused=True,source_runtime_verified=True,files=len(files),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip())
  write(b/'preflight.json',evidence);write(r/'preflight.json',evidence);reports.append(evidence)
 write(R/'preflight.json',dict(checked_unix=time.time(),models=reports,unique_hashed_runtime_paths=len(checked)))

def command(m,out,workers,episodes,attempts):
 b=Path(m['bulk'])
 return [str(b/'env/bin/python'),str(W/'scripts/evaluate_loopwam_robotwin.py'),'--checkpoint',str(b/'checkpoint/policy.pt'),'--output-dir',str(out),'--robotwin-root',str(W/'third_party/RoboTwin'),'--data-dir',str(b/'checkpoint'),'--gpus','0','--per-gpu',str(workers),'--episodes',str(episodes),'--seed','43','--expected-step','45445','--tasks',','.join(CONFIG['tasks']),'--max-attempts',str(attempts),'--no-progress-timeout','1800']

def smoke(m,workers):
 b=Path(m['bulk']);r=Path(m['report']);name=f'calibration_w{workers}';output=b/name;profile=b/(name+'_profile.json')
 if profile.exists():return json.loads(profile.read_text())
 env={**os.environ,'LOOPWAM_CUDA_GRAPH':'1','LOOPWAM_RUNTIME_MANIFEST':str(b/'runtime_manifest.json')}
 before=gpu();peak=before['used_mib'];min_free=before['free_mib'];start=time.time();proc=None
 status('calibrating',model=m['key'],workers=workers)
 with (b/(name+'.log')).open('a') as log,(b/(name+'_gpu.jsonl')).open('a') as telemetry:
  try:
   proc=subprocess.Popen(command(m,output,workers,2,1),cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while proc.poll() is None:
    g=gpu();peak=max(peak,g['used_mib']);min_free=min(min_free,g['free_mib']);telemetry.write(json.dumps(dict(unix=time.time(),**g))+'\n');telemetry.flush();time.sleep(3)
  finally:
   if proc is not None and proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(45)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 result=dict(name=name,workers=workers,backend='cuda-graph',seconds=time.time()-start,passed=False,exit_code=proc.returncode,episodes=0,peak_gpu_mib=peak,peak_added_mib=max(0,peak-before['used_mib']),minimum_free_mib=min_free)
 if proc.returncode==0 and (output/'COMPLETE.json').exists():
  from fastwam.robotwin_pipeline import validate_eval_summary
  c=json.loads((output/'COMPLETE.json').read_text());validate_eval_summary(c,m['checkpoint_sha256'],CONFIG['tasks'],2)
  result.update(passed=True,episodes=8,successes=sum(x['successes'] for x in c['finished'].values()));shutil.copy2(output/'COMPLETE.json',r/(name+'_complete.json'))
 write(profile,result);return result

def calibrate(m):
 b=Path(m['bulk']);r=Path(m['report'])
 if (b/'calibration.json').exists():return json.loads((b/'calibration.json').read_text())
 thresholds=THRESHOLDS[m['key']];wait_memory(thresholds[1],m['key'])
 workers=[2,m['max_workers']] if gpu()['free_mib']>=thresholds[2] else [1]
 profiles=[]
 for n in workers:
  if gpu()['free_mib']<thresholds[n]:continue
  p=smoke(m,n);profiles.append(p);write(b/'calibration_progress.json',dict(profiles=profiles))
  if not p['passed'] or p['minimum_free_mib']<8000:break
 valid=[p for p in profiles if p['passed'] and p['minimum_free_mib']>=8000]
 if not valid:raise RuntimeError(f'No safe calibration profile for {m["key"]}')
 choice=min(valid,key=lambda p:p['seconds']/p['episodes'])
 result=dict(profiles=profiles,selected=choice,finished_unix=time.time(),basis='Fastest valid same-model/seed43/two-task 8-episode graph profile, >=8000MiB minimum free GPU memory; calibration excluded from scored results')
 write(b/'calibration.json',result);write(r/'calibration.json',result);return result

def audit(m):
 b=Path(m['bulk']);r=Path(m['report']);status('auditing',model=m['key'])
 with (b/'finalizer.log').open('a') as log:subprocess.run([str(b/'env/bin/python'),str(b/'finalize_results.py')],cwd=W,stdout=log,stderr=subprocess.STDOUT,check=True)
 actual=json.loads((b/'results/manifest.json').read_text());assert actual['seed']==43 and actual['tasks']==CONFIG['tasks'] and actual['episodes']==100 and actual['num_inference_steps']==10
 shutil.copy2(b/'results/manifest.json',r/'protocol.json')


def final_comparison():
 rows=[]
 for m in CONFIG['models']:
  current=json.loads((Path(m['report'])/'final_results.json').read_text())
  prior=json.loads((Path(m['prior_report'])/'final_results.json').read_text())
  for task in CONFIG['tasks']:
   for setting in CONFIG['settings']:
    a=next(x for x in prior['tasks'] if x['task']==task and x['setting']==setting)
    z=next(x for x in current['tasks'] if x['task']==task and x['setting']==setting)
    rows.append(dict(model=m['key'],task=task,setting=setting,seed42_successes=a['successes'],seed43_successes=z['successes'],episodes_per_seed=100))
 write(R/'comparison.json',dict(rows=rows,seed43_total_episodes=800,finished_unix=time.time()))
 with (R/'comparison.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 lines=['# RoboTwin-10 two-task seed comparison','','Both models completed 400 seed-43 episodes each; all scored artifacts passed verification. Seed-42 results are the matching subset of the prior full evaluation.','','| Task | v0 seed42 | v0 seed43 | Dense-s30 seed42 | Dense-s30 seed43 |','| --- | ---: | ---: | ---: | ---: |']
 for task in CONFIG['tasks']+['Combined']:
  values=[]
  for key in CONFIG['order']:
   r=[x for x in rows if x['model']==key and (task=='Combined' or x['task']==task)]
   for seed in [42,43]:values.append(100*sum(x[f'seed{seed}_successes'] for x in r)/(len(r)*100))
  lines.append('| '+task+' | '+' | '.join(f'{v:.2f}%' for v in values)+' |')
 lines+=['','All rates combine clean and randomized equally. The selected two-task results are not full-ten-task benchmark scores. The original instruction sampler is not explicitly Python-seeded; instructions and accepted scene seeds are logged, but exact instruction pairing is not assumed. The new seed also changes the fixed-per-query CPU action noise.','', '[v0 detailed report](v0_4x4/final_report.md) · [Dense-s30 detailed report](dense_s30/final_report.md) · [Per-setting comparison CSV](comparison.csv) · [Setup](setup_report.md)']
 (R/'final_report.md').write_text('\n'.join(lines)+'\n')

def main():
 lock=(Q/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status('verifying');verify_and_bind()
 for m in CONFIG['models']:
  b=Path(m['bulk']);r=Path(m['report'])
  if not (b/'results/COMPLETE.json').exists():
   c=calibrate(m);required=max(THRESHOLDS[m['key']][c['selected']['workers']],c['selected']['peak_added_mib']+10000)
   wait_memory(required,m['key']);status('scoring',model=m['key'],workers=c['selected']['workers'],planned_episodes=400)
   with (b/'controller.log').open('a') as log:subprocess.run(['bash',str(b/'launch.sh')],cwd=W,stdout=log,stderr=subprocess.STDOUT,check=True)
  audit(m)
 final_comparison();status('complete',finished_unix=time.time(),scored_episodes=800)

if __name__=='__main__':
 try:main()
 except BaseException:
  status('error',error=traceback.format_exc());raise
