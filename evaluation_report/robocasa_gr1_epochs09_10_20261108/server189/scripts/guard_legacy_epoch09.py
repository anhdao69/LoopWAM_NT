#!/usr/bin/env python3
"""Guard the already-running epoch9 wrapper; hand off to reviewed runner only if it fails/retries."""
import pathlib,json,os,signal,time,subprocess,sys
root=pathlib.Path(__file__).resolve().parents[1];out=root/'results/official_/epoch_09';original_pid=1080416
proc=pathlib.Path(f'/proc/{original_pid}')
# Bind to /proc starttime as well as command path, preventing PID-reuse cleanup.
start=(proc/'stat').read_text().split()[21]
def original_alive():
 try:return (proc/'stat').read_text().split()[21]==start and str(root/'scripts/run.py').encode() in (proc/'cmdline').read_bytes()
 except FileNotFoundError:return False
def clean_driver(d):
 pid=d.get('driver_pid')
 if not pid:return
 owned=False
 for p in pathlib.Path('/proc').glob('[0-9]*'):
  try:
   if int((p/'stat').read_text().rsplit(')',1)[1].split()[2])==pid and str(root/'code/experiments/robocasa_gr1/').encode() in (p/'cmdline').read_bytes():owned=True;break
  except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError):pass
 if not owned:return
 try:os.killpg(pid,signal.SIGTERM)
 except ProcessLookupError:return
 time.sleep(5)
 try:os.killpg(pid,signal.SIGKILL)
 except ProcessLookupError:pass
last_driver={}
while True:
 s=json.loads((out/'STATUS.json').read_text());phase=s['phase']
 if s.get('driver_pid'):last_driver=s
 if phase=='complete':print('Legacy supervisor completed and validated normally',flush=True);break
 if phase=='retry_wait' and original_alive():
  print('Intercept retry and hand off to reviewed runner',flush=True);os.kill(original_pid,signal.SIGTERM)
  for _ in range(20):
   if not original_alive():break
   time.sleep(1)
 if not original_alive():
  clean_driver(last_driver)
  if (out/'STOP').exists() or (phase=='stopped'):
   print('Explicit stop respected; no restart',flush=True);break
  print('Legacy supervisor exited unexpectedly; launch reviewed runner preserving completed task cells',flush=True)
  # Old supervisor may have crashed before quitting its private MPS daemon.
  env=os.environ.copy();env['CUDA_MPS_PIPE_DIRECTORY']=str(root/'tmp/mps_09/pipe');env['CUDA_MPS_LOG_DIRECTORY']=str(root/'tmp/mps_09/log')
  subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  with (root/'logs/epoch09_supervisor.log').open('a') as log:
   new=subprocess.Popen(['bash',str(root/'scripts/launch.sh'),'9','24','6','7600'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  (out/'reviewed_handoff.json').write_text(json.dumps(dict(legacy_pid=original_pid,new_launcher_pid=new.pid,utc_seconds=time.time()),indent=2));break
 time.sleep(5)
