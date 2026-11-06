"""Persistent gated evaluation followed by strict artifact audit."""
import fcntl,importlib.util,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from queue_safety import reuse_completed
B=Path(__file__).resolve().parent.parent
S=json.loads((B/'run_settings.json').read_text());W=Path(S['repo']);R=Path(S['report']);R.mkdir(parents=True,exist_ok=True)
def status(stage,**extra):
 d=dict(stage=stage,updated_unix=time.time(),queue_pid=os.getpid(),variant=S['variant'],**extra)
 for p in [B/'queue_status.json',R/'queue_status.json']:
  tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
 print(json.dumps(d),flush=True)
def run(stage,args):
 status(stage)
 with (B/(stage+'.log')).open('a') as out:subprocess.run(args,cwd=W,stdout=out,stderr=subprocess.STDOUT,check=True)
def main():
 lock=(B/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 while not (B/'download_receipt.json').exists():status('waiting_for_verified_download');time.sleep(10)
 for name in ['tmp','cache']: (B/name).mkdir(exist_ok=True)
 run('runtime_verification',[sys.executable,str(B/'scripts/verify_runtime.py')])
 from fastwam.robotwin_pipeline import validate_eval_summary
 if reuse_completed(B,validate_eval_summary):
  run('finalizer',[sys.executable,str(B/'finalize_results.py')])
  status('complete',final_audit_passed=True,reused_complete=True);return
 local_test=W/'tests/test_robotwin_local_input.py'
 if S['host']!='local':local_test=Path('/media/newdisk/loopwam_robotwin10_v0_4x1_20261105/portable_tests/test_robotwin_local_input.py')
 tests=[str(W/'tests'/n) for n in ['test_robotwin_pipeline.py','test_loopwam_robot_eval.py','test_loopwam_robotwin10.py','test_loopwam_robot_full.py']]+[str(local_test)]
 run('tests',[sys.executable,'-m','pytest','-q',*tests])
 run('renderer_guard_tests',[sys.executable,str(B/'scripts/test_renderer_guard.py'),str(B/'finalize_results.py')])
 run('cpu_validation',[sys.executable,str(B/'scripts/cpu_validate.py')])
 minimum=60000 if S['workers']==4 else 37000
 while True:
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
  if free>=minimum:break
  status('waiting_for_gpu_memory',free_mib=free,required_mib=minimum);time.sleep(30)
 os.environ['LOOPWAM_CUDA_GRAPH']='1';os.environ['LOOPWAM_RUNTIME_MANIFEST']=str(B/'runtime_manifest.json')
 run('graph_gate',[sys.executable,str(B/'scripts/graph_gate.py')])
 gate=json.loads((B/'graph_gate.json').read_text());assert gate['graph_passed'] and gate['max_abs_difference']==0
 run('smoke_graph',[sys.executable,str(W/'scripts/evaluate_loopwam_robotwin.py'),'--checkpoint',str(B/'checkpoint/policy.pt'),'--output-dir',str(B/'smoke_graph'),'--robotwin-root',str(W/'third_party/RoboTwin'),'--data-dir',str(B/'checkpoint'),'--gpus','0','--per-gpu',str(S['workers']),'--episodes','1','--seed','42','--expected-step','45445','--tasks','adjust_bottle,open_microwave','--max-attempts','2','--no-progress-timeout','600'])
 from fastwam.robotwin_pipeline import validate_eval_summary
 smoke=json.loads((B/'smoke_graph/COMPLETE.json').read_text());validate_eval_summary(smoke,gate['checkpoint_sha256'],['adjust_bottle','open_microwave'],1)
 spec=importlib.util.spec_from_file_location('audit',B/'finalize_results.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
 for cell in smoke['finished'].values():audit.assert_clean_render_log((Path(cell['directory'])/'eval.log').read_text(errors='replace'))
 record=dict(passed=True,epoch=5,step=45445,variant=S['variant'],workers=S['workers'],seed=42,nfe=10,planned_episodes=2000,smoke_episodes=4,smoke_successes=sum(c['successes'] for c in smoke['finished'].values()),smoke_excluded=True,smoke_renderer_logs_clean=True,verified_unix=time.time())
 (B/'preflight.json').write_text(json.dumps(record,indent=2))
 for name in ['preflight.json','download_receipt.json','hf_manifest.json','cpu_validation.json','graph_gate.json','runtime_manifest.json','tests.log','renderer_guard_tests.log','planned_protocol.json','calibration.json']:shutil.copy2(B/name,R/name)
 shutil.copy2(B/'smoke_graph/COMPLETE.json',R/'smoke_complete.json')
 run('controller',[sys.executable,str(B/'scripts/robotwin_local_queue.py')])
 run('finalizer',[sys.executable,str(B/'finalize_results.py')])
 status('complete',final_audit_passed=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  status('error',error=traceback.format_exc());raise
