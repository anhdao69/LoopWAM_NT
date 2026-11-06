"""Ordered, persistent RoboTwin-10 v0 evaluation using the frozen evaluator."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import requests
B=Path('/data14t/anhdh35/loopwam_robotwin10_dense_s12_20261105')
D=Path('/data14t/anhdh35/loopwam_robotwin10_dense_s30_20261102')
W=Path('/storage/anhdh35/LoopWAM-robotwin10-dense-eval')
R=Path('/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_dense_s12_20261105')
PY=str(B/'env/bin/python')
THRESHOLDS={1:30000,2:42000,4:60000}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def write(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def status(stage,**data):
 value=dict(stage=stage,updated_unix=time.time(),queue_pid=os.getpid(),**data)
 write(B/'queue_status.json',value);write(R/'queue_status.json',value)
 print(json.dumps(value),flush=True)

def same_process(identity):
 if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=identity['boot_id']:return False
 try:
  fields=Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(')',1)[1].split()
  return fields[0]!='Z' and fields[19]==identity['start_ticks']
 except FileNotFoundError:return False

def gpu():
 line=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
 total,used,free,util=map(int,line.split(','));return dict(total_mib=total,used_mib=used,free_mib=free,utilization=util)

def wait_previous():
 config=json.loads((B/'wait_for.json').read_text())
 while True:
  active=[p['pid'] for p in config['processes'] if same_process(p)]
  complete=Path(config['dense_complete']).exists()
  audited=Path(config['dense_final_audit']).exists() and json.loads(Path(config['dense_final_audit']).read_text())['status']=='passed'
  if not active and complete and audited:
   write(B/'predecessor_completion.json',dict(checked_unix=time.time(),processes=config['processes'],dense_complete_sha256=sha(config['dense_complete']),dense_audit_sha256=sha(config['dense_final_audit'])));return
  status('waiting_for_previous_run',active_predecessor_pids=active,dense_complete=complete,dense_audited=audited,download_started=False,gpu=gpu());time.sleep(30)

def wait_memory(required,stage):
 stable=0
 while stable<3:
  g=gpu();stable=stable+1 if g['free_mib']>=required else 0
  status(stage,required_free_mib=required,stable_samples=stable,gpu=g)
  if stable<3:time.sleep(15)
 return g

def verify_file(path,item):
 if not path.is_file() or path.stat().st_size!=item['size']:return False
 if item.get('lfs'):return sha(path)==item['lfs']['sha256']
 content=path.read_bytes();return hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==item['blobId']

def download():
 m=json.loads((B/'hf_manifest.json').read_text());receipts=[]
 for item in m['files']:
  name=Path(item['rfilename']).name;target=B/'checkpoint'/name
  if not verify_file(target,item):
   error=None
   for attempt in range(1,6):
    status('downloading',file=name,attempt=attempt,revision=m['revision'])
    try:
     url=f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{item['rfilename']}"
     with requests.get(url,stream=True,timeout=(30,120)) as response:
      response.raise_for_status()
      with target.with_suffix(target.suffix+'.part').open('wb') as output:
       for chunk in response.iter_content(8*1024*1024):output.write(chunk)
     temp=target.with_suffix(target.suffix+'.part')
     if not verify_file(temp,item):raise ValueError(f'Integrity mismatch: {name}')
     temp.replace(target);error=None;break
    except Exception as exc:
     error=exc
     status('download_retry',file=name,attempt=attempt,error=repr(exc))
     if attempt<5:time.sleep(30)
   if error:raise RuntimeError(f'Cannot download verified {name}') from error
  receipts.append(dict(name=name,bytes=target.stat().st_size,sha256=sha(target)))
 for line in (B/'checkpoint/SHA256SUMS').read_text().splitlines():
  if not line.strip():continue
  expected,name=line.split(maxsplit=1);name=name.lstrip('*')
  assert Path(name).name==name and sha(B/'checkpoint'/name)==expected,name
 write(B/'download_receipt.json',dict(revision=m['revision'],verified_unix=time.time(),files=receipts));shutil.copy2(B/'download_receipt.json',R/'download_receipt.json')
 config=json.loads((B/'checkpoint/config.json').read_text());assert config==json.loads((B/'release_config_preview.json').read_text())
 for name,expected in config['training_code_sha256'].items():assert sha(W/name)==expected,name
 assert config['variant']=='dense-s12' and config['epoch']==5 and config['optimizer_update']==45445
 dense_config=json.loads((D/'checkpoint/config.json').read_text())
 for key in ['dataset','tasks','episodes','epochs','seed','global_batch','shuffle','noise','windows_per_epoch','planned_windows','planned_updates','normalization_sha256','data_manifest_sha256','vae_sha256','precision']:
  assert config['training_contract'][key]==dense_config['training_contract'][key],f'Training recipe differs: {key}'
 assert json.loads((B/'checkpoint/dataset_stats.json').read_text())==json.loads((D/'checkpoint/dataset_stats.json').read_text())
 protocol=json.loads((B/'planned_protocol.json').read_text());assert protocol['tasks']==config['training_contract']['tasks']
 write(B/'checkpoint/evaluation_protocol.json',protocol)


def run_logged(name,command):
 status(name,status='running')
 with (B/(name+'.log')).open('a') as log:
  subprocess.run(command,cwd=W,stdout=log,stderr=subprocess.STDOUT,check=True)
 if (B/(name+'.json')).exists():shutil.copy2(B/(name+'.json'),R/(name+'.json'))

def bind_runtime():
 old=json.loads((D/'runtime_manifest.json').read_text());files={p:info for p,info in old['files'].items() if not p.startswith(str(D)+'/')}
 for p,info in files.items():assert sha(p)==info['sha256'],f'Runtime drift: {p}'
 dense=json.loads((D/'results/manifest.json').read_text())
 for p,expected in dense['runtime_hashes'].items():
  if not p.startswith(str(D)+'/'):assert sha(p)==expected,f'Frozen evaluator drift: {p}'
 paths=list((B/'scripts').glob('*.py'))+list((B/'checkpoint').glob('*.json'))+[B/'environment.sh',B/'launch.sh',B/'finalize_results.py',B/'planned_protocol.json',B/'hf_manifest.json',B/'tests.json',B/'calibration_plan.json']
 for p in paths:files[str(p)]=dict(sha256=sha(p),size=p.stat().st_size)
 old.update(files=files,created_unix=time.time(),variant='dense-s12',hf_revision=json.loads((B/'hf_manifest.json').read_text())['revision'],shared_asset_manifest=str(D/'runtime_manifest.json'))
 target=B/'runtime_manifest.json'
 if target.exists():
  previous=json.loads(target.read_text());assert previous['files']==files,'Frozen runtime changed on resume'
 else:write(target,old)


def smoke(name,workers,graph):
 status(name,status='running',workers=workers,backend='cuda-graph' if graph else 'native')
 out=B/name
 plan=json.loads((B/'calibration_plan.json').read_text());tasks=plan['tasks'];episodes=plan['episodes_per_cell']
 cmd=[PY,str(W/'scripts/evaluate_loopwam_robotwin.py'),'--checkpoint',str(B/'checkpoint/policy.pt'),'--output-dir',str(out),'--robotwin-root',str(W/'third_party/RoboTwin'),'--data-dir',str(B/'checkpoint'),'--gpus','0','--per-gpu',str(workers),'--episodes',str(episodes),'--seed','42','--expected-step','45445','--tasks',','.join(tasks),'--max-attempts','1','--no-progress-timeout','600']
 start=time.time();before=gpu();peak=before['used_mib'];proc=None
 with (B/(name+'.log')).open('a') as log,(B/(name+'_gpu.jsonl')).open('a') as telemetry:
  try:
   proc=subprocess.Popen(cmd,cwd=W,env={**os.environ,'LOOPWAM_CUDA_GRAPH':str(int(graph)),'LOOPWAM_RUNTIME_MANIFEST':str(B/'runtime_manifest.json')},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while proc.poll() is None:
    g=gpu();peak=max(peak,g['used_mib']);telemetry.write(json.dumps(dict(unix=time.time(),**g))+'\n');telemetry.flush();time.sleep(3)
  finally:
   if proc is not None and proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(60)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 result=dict(name=name,workers=workers,backend='cuda-graph' if graph else 'native',seconds=time.time()-start,peak_gpu_mib=peak,baseline_gpu_mib=before['used_mib'],peak_added_mib=max(0,peak-before['used_mib']),exit_code=proc.returncode,passed=False,episodes=0)
 if proc.returncode==0 and (out/'COMPLETE.json').exists():
  from fastwam.robotwin_pipeline import validate_eval_summary
  c=json.loads((out/'COMPLETE.json').read_text());validate_eval_summary(c,sha(B/'checkpoint/policy.pt'),tasks,episodes)
  result.update(passed=True,episodes=len(tasks)*2*episodes,successes=sum(x['successes'] for x in c['finished'].values()))
  shutil.copy2(out/'COMPLETE.json',R/(name+'_complete.json'))
 result.update(tasks=tasks,episodes_per_cell=episodes);result['memory_margin_mib']=before['total_mib']-peak;write(B/(name+'_profile.json'),result)
 return result


def calibrate():
 if (B/'calibration.json').exists():return json.loads((B/'calibration.json').read_text())
 from speed_selection import required_memory,select_profile,assert_fresh_sweep
 assert_fresh_sweep(B)
 profiles=[];skipped=[]
 for name,workers,graph in [('speed_native_w4',4,False),('speed_graph_w4',4,True),('speed_graph_w8',8,True),('speed_graph_w12',12,True),('speed_graph_w10',10,True)]:
  try:required=required_memory(workers,profiles)
  except ValueError as exc:
   skipped.append(dict(name=name,reason=str(exc)));continue
  if workers==4:wait_memory(required,'waiting_for_calibration_memory')
  elif gpu()['free_mib']<required:
   skipped.append(dict(name=name,reason='Insufficient free GPU memory for measured projection plus reserve',required_mib=required,gpu=gpu()));continue
  result=smoke(name,workers,graph);profiles.append(result)
  write(B/'calibration_progress.json',dict(profiles=profiles,skipped=skipped))
  if workers==4 and not result['passed']:raise RuntimeError(f'Required simulator gate failed: {name}')
  if workers==8 and (not result['passed'] or result['memory_margin_mib']<8000):break
 selected=select_profile(profiles,available_mib=gpu()['free_mib'])
 calibration=dict(profiles=profiles,skipped=skipped,selected=selected,finished_unix=time.time(),basis='Fastest successful CUDA-graph wall time per episode across all ten tasks, both settings, one unscored episode per cell; native4 and graph4/8/12/10; higher counts require memory projections with reserve. Minimum selected-profile headroom 12000MiB; final admission must fit current free memory. Short profiles include startup and are not full-run time predictions.')
 write(B/'calibration.json',calibration);shutil.copy2(B/'calibration.json',R/'calibration.json');return calibration


def main():
 lock=(B/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not (B/'calibration.json').exists():
  from speed_selection import assert_fresh_sweep
  assert_fresh_sweep(B)
 wait_previous();download()
 run_logged('cpu_validation',[PY,str(B/'scripts/cpu_validate.py')])
 tests=json.loads((B/'tests.json').read_text());assert tests['passed'] and tests['count']==38
 bind_runtime()
 if not (B/'graph_gate.json').exists():
  wait_memory(18000,'waiting_for_graph_gate_memory');run_logged('graph_gate',[PY,str(B/'scripts/graph_gate.py')])
 gate=json.loads((B/'graph_gate.json').read_text());assert gate['graph_passed'] and gate['max_abs_difference']==0 and gate['checkpoint_sha256']==sha(B/'checkpoint/policy.pt')
 calibration=calibrate();cpu=json.loads((B/'cpu_validation.json').read_text());assert cpu['strict_load']
 preflight=dict(passed=True,tests_passed=38,source_hashes_match=9,strict_load=True,graph_bit_exact=True,smoke_episodes=sum(p['episodes'] for p in calibration['profiles']),smoke_excluded=True,workers=calibration['selected']['workers'],planned_episodes=2000,seed=42,nfe=10,variant='dense-s12',step=45445,verified_unix=time.time())
 write(B/'preflight.json',preflight);shutil.copy2(B/'preflight.json',R/'preflight.json')
 if not (B/'results/COMPLETE.json').exists():
  selected=calibration['selected'];from speed_selection import required_memory;required=required_memory(selected['workers'],calibration['profiles'])
  wait_memory(required,'waiting_for_full_evaluation_memory');status('full_evaluation',status='running',workers=selected['workers'])
  with (B/'controller.log').open('a') as log:subprocess.run(['bash',str(B/'launch.sh')],cwd=W,stdout=log,stderr=subprocess.STDOUT,check=True)
 status('final_audit',status='running');run_logged('finalizer',[PY,str(B/'finalize_results.py')])
 status('complete',status='complete',finished_unix=time.time(),results=json.loads((R/'final_results.json').read_text())['totals'])

if __name__=='__main__':
 try:main()
 except BaseException:
  status('error',status='error',error=traceback.format_exc());raise
