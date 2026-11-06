"""Launch the calibrated full benchmark and keep the report current."""
import fcntl,hashlib,json,os,signal,subprocess,time,traceback
from pathlib import Path
from robotwin_local_report import update
B=Path('/data14t/anhdh35/loopwam_robotwin10_v0_4x1_local_20261105');ROOT=Path('/storage/anhdh35/LoopWAM-robotwin10-dense-eval')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 lock=(B/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 calibration=json.loads((B/'calibration.json').read_text());assert calibration.get('finished_unix')
 gate=json.loads((B/'graph_gate.json').read_text());assert gate['graph_passed'] and gate['max_abs_difference']==0
 cpu=json.loads((B/'cpu_validation.json').read_text());assert cpu['strict_load'] and cpu['step']==45445
 assert sha(B/'checkpoint/policy.pt')==gate['checkpoint_sha256'],'Checkpoint changed after gate'
 for path,info in json.loads((B/'runtime_manifest.json').read_text())['files'].items():
  assert sha(path)==info['sha256'],f'Runtime asset drift: {path}'
 from fastwam.robotwin_pipeline import validate_eval_summary
 for directory in [x['name'] for x in calibration['profiles'] if x['passed']]:
  smoke=json.loads((B/directory/'COMPLETE.json').read_text());validate_eval_summary(smoke,gate['checkpoint_sha256'],['adjust_bottle','grab_roller'],2)
 workers=calibration['selected']['workers']
 if (B/'launch.json').exists():
  old=json.loads((B/'launch.json').read_text());assert old['workers']==workers,'Resume must keep selected workers'
 env={**os.environ,'LOOPWAM_CUDA_GRAPH':'1','LOOPWAM_RUNTIME_MANIFEST':str(B/'runtime_manifest.json')};env.pop('LOOPWAM_CAPTURE_INPUTS',None)
 cmd=[str(B/'env/bin/python'),str(ROOT/'scripts/evaluate_loopwam_robotwin.py'),'--checkpoint',str(B/'checkpoint/policy.pt'),
  '--output-dir',str(B/'results'),'--robotwin-root',str(ROOT/'third_party/RoboTwin'),'--data-dir',str(B/'checkpoint'),
  '--gpus','0','--per-gpu',str(workers),'--episodes','100','--seed','42','--expected-step','45445','--tasks',','.join(cpu['tasks']),'--max-attempts','3','--no-progress-timeout','1800']
 state=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),status='running',started_unix=time.time(),workers=workers,backend='cuda-graph',planned_episodes=2000,command=cmd)
 def save():
  tmp=B/'launch.json.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(B/'launch.json');update()
 def stop(*args):raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,stop)
 with (B/'evaluation.log').open('a') as log:proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state.update(pid=proc.pid,variant='v0-video4-action1',epoch=5,expected_step=45445,hf_revision=json.loads((B/'hf_manifest.json').read_text())['revision']);save()
 try:
  while proc.poll() is None:save();time.sleep(10)
  if proc.returncode:raise RuntimeError(f'Evaluation exit {proc.returncode}; see evaluation.log')
  assert (B/'results/COMPLETE.json').exists()
  state.update(status='complete',finished_unix=time.time());save()
 except BaseException:
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(60)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  state.update(status='error',error=traceback.format_exc(),stopped_unix=time.time());save();raise

if __name__=='__main__':main()
