"""Persistent gated epoch10 evaluation; owned subprocesses stop with supervisor."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
B=Path(__file__).resolve().parent.parent;R=B/'report';P=Path('/data14t/anhdh35/loopwam_robotwin10_dense_s12_20261105');W=Path('/storage/anhdh35/LoopWAM-robotwin10-dense-eval')
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def status(stage,**extra):
 d=dict(stage=stage,updated_unix=time.time(),queue_pid=os.getpid(),**extra);write(B/'queue_status.json',d);write(R/'queue_status.json',d);print(json.dumps(d),flush=True)
def process(pid):
 try:
  a=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return dict(state=a[0],start_ticks=a[19])
 except FileNotFoundError:return None
def gpu():
 line=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip()
 total,used,free=map(int,line.split(','));return dict(total_mib=total,used_mib=used,free_mib=free)
def run(stage,cmd,reserve_mib=0):
 status(stage)
 started=time.time();before=gpu() if reserve_mib else None;peak=before['used_mib'] if before else 0
 with (B/(stage+'.log')).open('a') as f:
  p=None;record=dict(stage=stage,command=cmd,started_unix=time.time(),status='starting')
  try:
   p=subprocess.Popen(cmd,cwd=W,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   record.update(pid=p.pid,start_ticks=process(p.pid)['start_ticks'],status='running');write(B/'active_process.json',record)
   while p.poll() is None:
    if reserve_mib:
     g=gpu();peak=max(peak,g['used_mib'])
     if g['free_mib']<reserve_mib:raise RuntimeError(f'GPU memory guard: {g["free_mib"]}MiB free below {reserve_mib}')
    time.sleep(2)
   if p.returncode:raise RuntimeError(f'{stage} exited {p.returncode}; see {stage}.log')
  finally:
   if p is not None and p.poll() is None:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(240)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   record.update(status='exited',exit_code=p.returncode if p is not None else None,finished_unix=time.time());write(B/'active_process.json',record)
 return dict(seconds=time.time()-started,peak_gpu_mib=peak,peak_added_mib=peak-before['used_mib'],memory_margin_mib=before['total_mib']-peak) if before else {}
def mirror():
 for name in ['planned_protocol.json','hf_manifest.json','download_receipt.json','cpu_validation.json','graph_gate.json','calibration.json','preflight.json','tests.log','runtime_manifest.json']:
  if (B/name).exists():shutil.copy2(B/name,R/name)
def main():
 lock=(B/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (B/'active_process.json').exists():
  old=json.loads((B/'active_process.json').read_text());actual=process(old['pid'])
  if actual and actual['state']!='Z' and actual['start_ticks']==old['start_ticks']:raise RuntimeError('Previous child still active; refusing duplicate')
 def stop(*_):raise KeyboardInterrupt('Stopped by user')
 signal.signal(signal.SIGTERM,stop)
 run('runtime_verification',[sys.executable,str(B/'scripts/verify_runtime.py')])
 tests=[W/'tests'/n for n in ['test_robotwin_pipeline.py','test_loopwam_robot_eval.py','test_loopwam_robotwin10.py','test_loopwam_robot_full.py']]+[W/'tests/test_robotwin_local_input.py']
 receipt=B/'prior_checks.json'
 if receipt.exists():
  d=json.loads(receipt.read_text());assert d['tests_passed']==38
  for name,expected in d['sha256'].items():assert hashlib.sha256((B/name).read_bytes()).hexdigest()==expected,name
 else:run('tests',[sys.executable,'-m','pytest','-q',*[str(p) for p in tests]])
 run('renderer_guard_tests',[sys.executable,str(B/'scripts/test_renderer_guard.py'),str(B/'finalize_results.py')])
 if not receipt.exists():run('cpu_validation',[sys.executable,str(B/'scripts/cpu_validate.py')])
 cpu=json.loads((B/'cpu_validation.json').read_text());assert cpu['strict_load'] and cpu['finite_weights'] and cpu['step']==90890 and cpu['epoch']==10
 while True:
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  if free>=60000:break
  status('waiting_for_gpu',free_mib=free,required_mib=60000);time.sleep(30)
 os.environ['LOOPWAM_CUDA_GRAPH']='1';os.environ['LOOPWAM_RUNTIME_MANIFEST']=str(B/'runtime_manifest.json')
 run('graph_gate',[sys.executable,str(B/'scripts/graph_gate.py')])
 gate=json.loads((B/'graph_gate.json').read_text());assert gate['graph_passed'] and gate['max_abs_difference']==0 and gate['checkpoint_sha256']=='a222704c10688bf42a531a4a4175fe45a1b172c437cf9b9efa435c4b4fae1ac8'
 from calibrate import calibrate
 calibration=calibrate(B,W,sys.executable,run,status,gpu,write,gate['checkpoint_sha256'])
 selected=calibration['selected'];required=max(60000,selected['peak_added_mib']+8000)
 while gpu()['free_mib']<required:status('waiting_for_scoring_memory',required_mib=required);time.sleep(30)
 write(B/'preflight.json',dict(passed=True,epoch=10,step=90890,variant='dense-s12',video_loops=1,action_loops=1,planned_episodes=2000,seed=42,nfe=10,graph_bit_exact=True,smoke_episodes=sum(p['episodes'] for p in calibration['profiles']),smoke_excluded=True,workers=selected['workers']));mirror()
 from fastwam.robotwin_pipeline import validate_eval_summary
 if not (B/'results/COMPLETE.json').exists():run('controller',[sys.executable,str(B/'scripts/robotwin_local_queue.py')])
 else:
  c=json.loads((B/'results/COMPLETE.json').read_text());p=json.loads((B/'results/manifest.json').read_text());assert p['checkpoint_sha256']==gate['checkpoint_sha256'] and p['expected_step']==90890
  validate_eval_summary(c,gate['checkpoint_sha256'],json.loads((B/'planned_protocol.json').read_text())['tasks'],100)
  l=json.loads((B/'launch.json').read_text());l.update(status='complete',finished_unix=(B/'results/COMPLETE.json').stat().st_mtime);write(B/'launch.json',l)
 run('finalizer',[sys.executable,str(B/'finalize_results.py')]);status('complete',final_audit_passed=True);mirror()
if __name__=='__main__':
 try:main()
 except BaseException as e:status('stopped_by_user' if isinstance(e,KeyboardInterrupt) else 'error',error=traceback.format_exc());raise
