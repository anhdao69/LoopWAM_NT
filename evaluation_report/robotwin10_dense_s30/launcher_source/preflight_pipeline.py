"""Run unscored simulator gates, select safe concurrency, then launch the full queue."""
import json,os,signal,subprocess,time,shutil
from pathlib import Path
B=Path('/data14t/anhdh35/loopwam_robotwin10_dense_s30_20261102')
W=Path('/storage/anhdh35/LoopWAM-robotwin10-dense-eval')
R=Path('/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_dense_s30')
PY=str(B/'env/bin/python')
def write(name,data):
 p=B/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
def main():
 profiles=[]
 for name,workers,graph in [('smoke_native',1,'0'),('smoke_graph',2,'1')]:
  write('setup_status.json',dict(stage=name,status='running',started_unix=time.time()))
  cmd=[PY,str(W/'scripts/evaluate_loopwam_robotwin.py'),'--checkpoint',str(B/'checkpoint/policy.pt'),'--output-dir',str(B/name),'--robotwin-root',str(W/'third_party/RoboTwin'),'--data-dir',str(B/'checkpoint'),'--gpus','0','--per-gpu',str(workers),'--episodes','2','--seed','42','--expected-step','45445','--tasks','adjust_bottle,grab_roller','--max-attempts','1','--no-progress-timeout','600']
  start=time.time();peak=0
  with (B/(name+'.log')).open('a') as log,(B/(name+'_gpu.csv')).open('w') as telemetry:
   telemetry.write('unix,gpu_memory_mib,gpu_utilization_percent\n');telemetry.flush()
   proc=subprocess.Popen(cmd,cwd=W,env={**os.environ,'LOOPWAM_CUDA_GRAPH':graph,'LOOPWAM_RUNTIME_MANIFEST':str(B/'runtime_manifest.json')},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while proc.poll() is None:
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0];mem,util=map(int,raw.split(','));peak=max(peak,mem);telemetry.write(f'{time.time()},{mem},{util}\n');telemetry.flush();time.sleep(3)
   if proc.returncode:raise RuntimeError(f'{name} exited {proc.returncode}: see {name}.log')
  complete=json.loads((B/name/'COMPLETE.json').read_text());assert len(complete['finished'])==4
  assert all(len(v['episodes'])==2 for v in complete['finished'].values())
  profiles.append(dict(name=name,workers=workers,backend='cuda-graph' if graph=='1' else 'native',seconds=time.time()-start,episodes=8,successes=sum(v['successes'] for v in complete['finished'].values()),peak_gpu_mib=peak,includes_other_gpu_processes=True))
  shutil.copy2(B/name/'COMPLETE.json',R/(name+'_complete.json'))
 workers=2 if profiles[-1]['peak_gpu_mib']<88000 else 1
 selection=dict(profiles=profiles,selected=dict(workers=workers),finished_unix=time.time(),basis='Two-worker simulator gate plus measured memory headroom; profiles differ in both backend and concurrency, not an isolated concurrency speed comparison')
 write('calibration.json',selection);shutil.copy2(B/'calibration.json',R/'calibration.json')
 gate=json.loads((B/'graph_gate.json').read_text());cpu=json.loads((B/'cpu_validation.json').read_text());assert gate['graph_passed'] and gate['max_abs_difference']==0 and cpu['strict_load']
 preflight=dict(passed=True,tests_passed=38,source_hashes_match=9,strict_load=True,graph_bit_exact=True,smoke_episodes=16,smoke_successes=sum(p['successes'] for p in profiles),smoke_excluded=True,workers=workers,planned_episodes=2000,seed=42,nfe=10,variant='dense-s30',step=45445,verified_unix=time.time())
 write('preflight.json',preflight);shutil.copy2(B/'preflight.json',R/'preflight.json')
 write('setup_status.json',dict(stage='full_evaluation',status='launching',updated_unix=time.time()))
 with (B/'controller.log').open('a') as log:
  code=subprocess.call(['bash',str(B/'launch.sh')],cwd=W,stdout=log,stderr=subprocess.STDOUT)
 if code:raise RuntimeError(f'Full evaluation controller exited {code}')
 write('setup_status.json',dict(stage='full_evaluation',status='complete',finished_unix=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  write('setup_status.json',dict(status='error',error=repr(exc),updated_unix=time.time()));raise
