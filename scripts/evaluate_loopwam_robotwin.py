#!/usr/bin/env python3
"""Full original-v0 RoboTwin evaluation; only verified complete cells are resumable."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fastwam.robotwin_pipeline import validate_eval_summary,fingerprint_files
EPISODE=re.compile(r'Success rate: (?:\x1b\[\d+m)?(\d+)/(\d+).*?current seed: (?:\x1b\[\d+m)?(\d+)')

def write(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def completed_records(directory,config,expected,verify=True):
    triples=EPISODE.findall((directory/'eval.log').read_text(errors='replace'))
    if len(triples)!=expected:raise ValueError(f'Expected{expected}episodes, found{len(triples)}')
    out=[];previous=0
    for index,(successes,count,seed) in enumerate(triples):
        successes,count,seed=map(int,(successes,count,seed));success=successes-previous
        if count!=index+1 or success not in (0,1):raise ValueError('Invalid cumulative episode counters')
        video=directory/f'episode{index}_randomized-{str(config=="demo_randomized").lower()}_success-{str(bool(success)).lower()}.mp4'
        if not video.is_file() or video.stat().st_size<1024:raise ValueError(f'Missing/nonempty video:{video}')
        if verify:
            p=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,duration','-of','json',str(video)],capture_output=True,text=True,timeout=30)
            if p.returncode or not json.loads(p.stdout).get('streams'):raise ValueError(f'Invalid video:{video}')
        out.append(dict(seed=seed,success=bool(success),video=str(video),video_verified=True));previous=successes
    if len({x['seed'] for x in out})!=expected:raise ValueError('Repeated scene seed')
    return out

def terminate(proc):
    if proc.poll() is not None:return
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(20)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()

def main():
    p=argparse.ArgumentParser()
    for name in ['checkpoint','output-dir','robotwin-root','data-dir']:p.add_argument('--'+name,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--per-gpu',type=int,default=2)
    p.add_argument('--episodes',type=int,default=100);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tasks');p.add_argument('--expected-step',type=int,default=237310)
    p.add_argument('--compile-action-infer',action='store_true');p.add_argument('--max-attempts',type=int,default=3)
    p.add_argument('--no-progress-timeout',type=int,default=1800);a=p.parse_args()
    from fastwam.datasets.chronoloop_robot import ROBOTWIN50_TASKS
    tasks=list(ROBOTWIN50_TASKS) if not a.tasks else a.tasks.split(',')
    if len(set(tasks))!=len(tasks) or not set(tasks)<=set(ROBOTWIN50_TASKS):raise ValueError('Invalid tasks')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'runner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sha=digest(a.checkpoint)
    robotwin=Path(a.robotwin_root);policy=(robotwin/'policy/loopwam_policy').resolve()
    if policy!=(ROOT/'experiments/robotwin/loopwam_policy').resolve():raise ValueError('Wrong deployed policy import target')
    bound=list((robotwin/'envs').rglob('*.py'))+list((robotwin/'task_config').rglob('*.yml'))+list(policy.glob('*.py'))+list(policy.glob('*.yml'))+list((ROOT/'src/fastwam/models/wan22').rglob('*.py'))
    protocol=dict(checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=sha,expected_step=a.expected_step,
        tasks=tasks,settings=['demo_clean','demo_randomized'],episodes=a.episodes,seed=a.seed,instruction_type='unseen',
        num_inference_steps=10,action_horizon=32,replan_steps=24,noise_seed='fixed seed for every query, FastWAM default',
        compile_action_infer=a.compile_action_infer,renderer_shader=os.environ.get('LOOPWAM_ROBOTWIN_SHADER','rt'),harness_sha256=digest(Path(a.robotwin_root)/'script/eval_policy.py'),
        policy_sha256=digest(ROOT/'experiments/robotwin/loopwam_policy/deploy_policy.py'),
        stats_sha256=digest(Path(a.data_dir)/'dataset_stats.json'),runtime_hashes=fingerprint_files(bound),
        vae_sha256=digest(ROOT/'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'))
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:raise ValueError('Cannot reuse different evaluation contract')
    write(manifest,protocol)
    finished={};pending=[];attempts={}
    for task in tasks:
        for config in protocol['settings']:
            key=f'{task}/{config}';cell=out/f'{task}__{config}';cell.mkdir(exist_ok=True)
            result=cell/'result.json'
            if result.exists():
                old=json.loads(result.read_text());d=Path(old['directory'])
                if old['checkpoint_sha256']!=sha:raise ValueError('Mismatched cell checkpoint')
                records=completed_records(d,config,a.episodes)
                if records!=old['episodes']:raise ValueError('Cell records changed')
                finished[key]=old;continue
            attempts[key]=len(list(cell.glob('attempt_*')));pending.append((task,config))
    summary=dict(checkpoint_sha256=sha,finished=finished,running={},updated=time.time())
    if not pending:
        validate_eval_summary(summary,sha,tasks,a.episodes);write(out/'COMPLETE.json',summary);return
    env=dict(os.environ);env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    gpus=a.gpus.split(',');slots=[g for _ in range(a.per_gpu) for g in gpus]
    sock=f'/tmp/loopwam_t5_{os.getpid()}.sock'
    server_log=(out/f't5_{time.time_ns()}.log').open('w')
    server=subprocess.Popen([sys.executable,str(ROOT/'experiments/robotwin/chronoloop_policy/t5_server.py'),'--address',sock],env={**env,'CUDA_VISIBLE_DEVICES':gpus[0]},stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
    running={};failed=[]
    def stop_signal(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop_signal)
    try:
        start=time.time()
        while not Path(sock).exists():
            if server.poll() is not None or time.time()-start>600:raise RuntimeError('T5 startup failed/timed out')
            time.sleep(2)
        while pending or running:
            if server.poll() is not None:raise RuntimeError('Shared T5 server exited')
            for slot,gpu in enumerate(slots):
                if slot in running or not pending:continue
                task,config=pending.pop(0);key=f'{task}/{config}';attempts[key]+=1
                if attempts[key]>a.max_attempts:failed.append(key);continue
                d=out/f'{task}__{config}'/f'attempt_{attempts[key]:02d}';d.mkdir()
                cmd=[sys.executable,'script/eval_policy.py','--config','policy/loopwam_policy/deploy_policy.yml','--overrides',
                    '--task_name',task,'--task_config',config,'--ckpt_setting',str(Path(a.checkpoint).resolve()),
                    '--seed',str(a.seed),'--eval_num_episodes',str(a.episodes),'--instruction_pool',str(a.episodes),
                    '--eval_output_dir',str(d),'--data_dir',a.data_dir,'--t5_server',sock,
                    '--expected_step',str(a.expected_step),'--compile_action_infer',str(a.compile_action_infer)]
                write(d/'command.json',cmd);log=(d/'eval.log').open('w')
                proc=subprocess.Popen(cmd,cwd=a.robotwin_root,env={**env,'CUDA_VISIBLE_DEVICES':gpu},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[slot]=dict(proc=proc,task=task,config=config,d=d,log=log,started=time.time(),progress=time.time(),episodes=0)
            for slot,s in list(running.items()):
                text=(s['d']/'eval.log').read_text(errors='replace');count=len(EPISODE.findall(text))
                if count>s['episodes']:s['episodes']=count;s['progress']=time.time()
                if time.time()-s['progress']>a.no_progress_timeout:terminate(s['proc'])
                rc=s['proc'].poll()
                if rc is None:continue
                key=f'{s["task"]}/{s["config"]}';s['log'].close()
                try:
                    if rc!=0:raise RuntimeError(f'Worker exit{rc}')
                    records=completed_records(s['d'],s['config'],a.episodes)
                    cell=dict(complete=True,exit_code=0,checkpoint_sha256=sha,task=s['task'],config=s['config'],episodes=records,
                        successes=sum(x['success'] for x in records),directory=str(s['d']),seconds=time.time()-s['started'])
                    write(s['d'].parent/'result.json',cell);finished[key]=cell
                except Exception as e:
                    write(s['d']/'failure.json',dict(error=str(e),exit_code=rc,episodes=count));pending.append((s['task'],s['config']))
                del running[slot]
            summary=dict(checkpoint_sha256=sha,finished=finished,running={f'{s["task"]}/{s["config"]}':dict(episodes=s['episodes'],started=s['started']) for s in running.values()},failed=failed,pending=len(pending),updated=time.time())
            write(out/'summary.json',summary)
            time.sleep(5)
        if failed:raise RuntimeError(f'Failed evaluation cells:{failed}')
        validate_eval_summary(summary,sha,tasks,a.episodes)
        summary['success_rates']={c:sum(v['successes'] for v in finished.values() if v['config']==c)/(len(tasks)*a.episodes) for c in protocol['settings']}
        write(out/'COMPLETE.json',summary)
    finally:
        for s in running.values():terminate(s['proc']);s['log'].close()
        terminate(server);server_log.close()

if __name__=='__main__':main()
