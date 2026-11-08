#!/usr/bin/env python3
"""Persistent, checkpoint-bound official evaluation. Resume preserves validated task cells."""
import argparse,datetime,json,os,pathlib,signal,subprocess,sys,time,shutil,hashlib
from control_helpers import verified_inputs, cleanup_group
p=argparse.ArgumentParser();p.add_argument('epoch',type=int,choices=(9,10));p.add_argument('--parallel',type=int,required=True);p.add_argument('--servers',type=int,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--port',type=int,required=True);a=p.parse_args()
root=pathlib.Path(os.environ['RUN_ROOT']);out=root/'results'/('smoke_' if a.smoke else 'official_')/f'epoch_{a.epoch:02d}';out.mkdir(parents=True,exist_ok=True)
status_path=out/'STATUS.json'; child=None; stop=False; config={}

def status(phase,**extra):
 d=dict(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supervisor_pid=os.getpid(),host=os.uname().nodename,epoch=a.epoch,checkpoint_sha256=config.get('checkpoint_sha256'),parallel=a.parallel,servers=a.servers,**extra)
 temp=status_path.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2));temp.replace(status_path)
def halt(sig=None,frame=None):
 global stop
 stop=True
 if child and child.poll() is None:
  try: os.killpg(child.pid,signal.SIGTERM)
  except ProcessLookupError:pass
signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
status('verifying')
ck=root/'models/Robocasa-GR1'/f'epoch_{a.epoch:02d}.pt';inputs=verified_inputs(root,a.epoch,os.environ['FASTWAM_PY'])
config=dict(epoch=a.epoch,checkpoint=str(ck),checkpoint_sha256=inputs['checkpoint_sha256'],code_revision=inputs['code_revision'],expected_step=a.epoch*23516,episodes=1 if a.smoke else 50,n_envs=1,n_action_steps=12,max_episode_steps=720,env_seed=0,noise_seed=42,nfe=10,cfg=1.0,render='egl',video=True,tasks=inputs['tasks'][:1] if a.smoke else inputs['tasks'])
binding=out/'binding.json'
if binding.exists(): assert json.loads(binding.read_text())==config,'Output binding differs: refuse reuse'
else: binding.write_text(json.dumps(config,indent=2))
if stop or (out/'STOP').exists():
 status('stopped');sys.exit(143)


def validate_smoke():
 f=out/'n12'/f"{config['tasks'][0].split('/')[-1]}.json";d=json.loads(f.read_text());m=d['server_metadata']
 for k,v in dict(n_episodes=1,n_envs=1,n_action_steps=12,max_episode_steps=720,seed=0).items(): assert d[k]==v,(k,d[k],v)
 for k,v in dict(step=config['expected_step'],normalization_sha256=inputs['normalization_sha256'],num_inference_steps=10,cfg_scale=1.0,noise_seed=42,action_chunk_size=32).items():assert m[k]==v,(k,m[k],v)
 assert pathlib.Path(m['checkpoint']).resolve()==ck.resolve();assert len(d['successes'])==1;assert 60<=d['client']['queries']<=61;assert len(list((out/'videos/n12').glob('*/*_success[01].mp4')))==1
 (out/'VALIDATED.json').write_text(json.dumps(dict(valid=True,checkpoint_sha256=config['checkpoint_sha256'],queries=d['client']['queries'],successes=d['successes_first_n'],wall_seconds=d['wall_seconds']),indent=2))

if not a.smoke:
 smoke=root/'results/smoke_'/f'epoch_{a.epoch:02d}'/'VALIDATED.json';assert json.loads(smoke.read_text())['checkpoint_sha256']==config['checkpoint_sha256'],'Need successful smoke for exact checkpoint'
# The unmodified driver skips completed task cells. Check their complete bindings before trusting them.
for f in (out/'n12').glob('*.json'):
 d=json.loads(f.read_text());m=d['server_metadata'];assert d['env_name'] in config['tasks'];assert pathlib.Path(m['checkpoint']).resolve()==ck.resolve();assert m['step']==config['expected_step'];assert d['n_episodes']==config['episodes'];assert len(d['successes'])==config['episodes'];assert d['seed']==0
# Isolate MPS to this authorized run. The prior validated workstation evaluator uses MPS too.
mps=root/'tmp'/f'mps_{a.epoch:02d}';(mps/'pipe').mkdir(parents=True,exist_ok=True);(mps/'log').mkdir(exist_ok=True)
os.environ['CUDA_MPS_PIPE_DIRECTORY']=str(mps/'pipe');os.environ['CUDA_MPS_LOG_DIRECTORY']=str(mps/'log')
mps_log=open(out/'mps.log','a');mps_started=subprocess.run(['nvidia-cuda-mps-control','-d'],stdout=mps_log,stderr=subprocess.STDOUT).returncode==0
if not mps_started:
 os.environ.pop('CUDA_MPS_PIPE_DIRECTORY',None);os.environ.pop('CUDA_MPS_LOG_DIRECTORY',None)
try:
 for attempt in range(1,4):
  if stop or (out/'STOP').exists():break
  # A failed cell must restart from episode zero to preserve its seed sequence.
  stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  for d in (out/'videos/n12').glob('*'):
   if d.is_dir() and not (out/'n12'/f'{d.name}.json').exists():
    dest=out/'partial_attempts'/stamp;dest.mkdir(parents=True,exist_ok=True);shutil.move(str(d),str(dest/d.name))
  if (out/'logs').exists() and any((out/'logs').iterdir()):
   archive=out/'attempt_logs'/stamp;archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(out/'logs'),str(archive))
  cmd=[os.environ['FASTWAM_PY'],str(root/'code/experiments/robocasa_gr1/run_eval.py'),'--checkpoint',str(ck),'--stats',str(root/'models/Robocasa-GR1/dataset_stats.json'),'--out',str(out),'--steps','12','--episodes',str(config['episodes']),'--n-envs','1','--max-episode-steps','720','--seed','0','--gpus','0','--servers-per-gpu',str(a.servers),'--base-port',str(a.port),'--parallel',str(a.parallel),'--server-threads','4','--sim-threads','2','--video','--render','egl','--vae-path',str(root/'models/vae/Wan2.1_VAE.pth'),'--expected-step',str(config['expected_step']),'--no-t5-fallback']
  if a.smoke:cmd+=['--tasks',config['tasks'][0]]
  (out/f'command_attempt{attempt}.json').write_text(json.dumps(cmd,indent=2))
  with (out/'driver.log').open('a') as log:
   child=subprocess.Popen(cmd,cwd=root/'code',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while child.poll() is None:
    status('running',driver_pid=child.pid,driver_pgid=child.pid,attempt=attempt,finished_tasks=len(list((out/'n12').glob('*.json'))),finished_videos=len(list((out/'videos/n12').glob('*/*_success[01].mp4'))),mps=mps_started)
    if (out/'STOP').exists():halt()
    time.sleep(10)
  rc=child.returncode
  # Terminate descendants still present after an exceptional driver exit.
  cleanup_group(child)
  if stop:break
  if rc==0:
   if a.smoke:validate_smoke();status('complete',validated=True);sys.exit(0)
   with (out/'validation.txt').open('w') as log:
    valid=subprocess.run([sys.executable,str(root/'scripts/gr1_validate.py'),str(out),str(config['expected_step']),'--json',str(out/'validation.json')],stdout=log,stderr=subprocess.STDOUT).returncode==0
   if valid:
    (out/'COMPLETE.json').write_text(json.dumps(dict(valid=True,checkpoint_sha256=config['checkpoint_sha256'],completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2));status('complete',validated=True);sys.exit(0)
  status('retry_wait',attempt=attempt,returncode=rc)
  for _ in range(6):
   if stop:break
   time.sleep(10)
 status('stopped' if stop or (out/'STOP').exists() else 'failed');sys.exit(143 if stop else 1)
finally:
 cleanup_group(child)
 if mps_started:subprocess.run(['nvidia-cuda-mps-control'],input='quit\n',text=True,stdout=mps_log,stderr=subprocess.STDOUT)
