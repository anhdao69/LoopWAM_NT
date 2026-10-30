#!/usr/bin/env python3
"""Persistent allocation controller: verified epoch5 -> full eval -> epochs6-10."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fastwam.robotwin_pipeline import validate_eval_summary

def write(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def read(path):return json.loads(Path(path).read_text())

def active_steps(job,name=None):
    p=subprocess.run(['squeue','--steps','--jobs',str(job),'--noheader','--format=%i|%j'],capture_output=True,text=True,check=True)
    pairs=[line.strip().split('|',1) for line in p.stdout.splitlines() if '|' in line]
    return [i for i,n in pairs if name is None or n==name]

def shell(env,argv):return ['bash','-c',f'source {shlex.quote(env)}; exec '+shlex.join(argv)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=read(a.config)
    out=Path(cfg['queue_dir']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job=os.environ.get('SLURM_JOB_ID',str(cfg['job_id']))
    def heartbeat(stage,**extra):write(out/'status.json',dict(stage=stage,job_id=job,controller_pid=os.getpid(),updated=time.time(),**extra))
    def stage(name,command,gpus=8,attempts=1):
        status=out/f'{name}.status.json';command_file=out/f'{name}.command.json';jobname=f'rwq_{name}'
        old=read(status) if status.exists() else None
        if old and old.get('state')=='finished' and old.get('exit_code')==0:
            if old.get('command')!=command:raise ValueError(f'Completed stage command changed:{name}')
            return
        for attempt in range(attempts):
            steps=active_steps(job,jobname)
            proc=None
            if not steps:
                write(command_file,command)
                log=(out/f'{name}.{time.time_ns()}.log').open('w')
                argv=['srun','--jobid',str(job),'--overlap','--ntasks=1','--cpus-per-task='+str(112 if gpus else 4),
                      '--gres='+('gpu:8' if gpus else 'none'),'--kill-on-bad-exit=1','--job-name',jobname,
                      sys.executable,str(ROOT/'scripts/robotwin_queue_stage.py'),'--command',str(command_file),'--status',str(status)]
                write(out/f'{name}.launch.json',dict(command=argv,created=time.time(),attempt=attempt))
                proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
                log.close()
            else:heartbeat(name,adopted_steps=steps)
            started=time.time()
            while True:
                current=read(status) if status.exists() else {}
                live=active_steps(job,jobname)
                if current.get('state')=='finished' and not live:
                    if proc is not None and proc.poll() is None:time.sleep(1);continue
                    if current['exit_code']==0:return
                    break
                if proc is not None and proc.poll() is not None and not live:break
                if proc is None and not live and time.time()-started>30:break
                heartbeat(name,child=current,steps=live);time.sleep(20)
            if attempt+1==attempts:raise RuntimeError(f'Stage{name}failed; inspect stage logs')
    try:
        heartbeat('waiting_for_epoch5')
        parent=Path(cfg['parent_dir']);release=parent.parent
        while not (release/'verified_complete.json').exists() or f'{job}.10' in active_steps(job):
            heartbeat('waiting_for_epoch5',parent_timing=read(parent/'timing.json'))
            time.sleep(30)
        while not Path(cfg['preflight']).exists():
            heartbeat('waiting_for_preflight');time.sleep(30)
        preflight=read(cfg['preflight'])
        if not preflight.get('passed') or not preflight.get('simulator_clean_and_randomized'):
            raise ValueError('Simulator preflight did not pass')
        t=read(parent/'COMPLETE')
        if (t['status'],t['update'],t['windows_seen'])!=('complete',237310,30375515):raise ValueError('Parent did not complete exact epoch5 budget')
        for path,sha in cfg['source_hashes'].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:raise ValueError(f'Queue source changed:{path}')
        weights=parent/'checkpoints/epoch_05.pt';recovery=parent/'latest.pt'
        stage('handoff',shell(cfg['train_env'],[sys.executable,str(ROOT/'scripts/verify_robotwin_handoff.py'),'--weights',str(weights),'--recovery',str(recovery),'--output',str(out/'handoff.json')]),gpus=0)
        handoff=read(out/'handoff.json')
        if not handoff['passed']:raise ValueError('Handoff gate failed')
        stage('eval5',shell(cfg['eval_env'],[sys.executable,str(ROOT/'scripts/evaluate_loopwam_robotwin.py'),'--checkpoint',str(weights),'--output-dir',cfg['eval_dir'],
            '--robotwin-root',cfg['robotwin_root'],'--data-dir',cfg['shared_dir']+'/data','--per-gpu',str(cfg['eval_per_gpu']),'--seed','42']))
        from fastwam.datasets.chronoloop_robot import ROBOTWIN50_TASKS
        validate_eval_summary(read(Path(cfg['eval_dir'])/'COMPLETE.json'),handoff['weights_sha256'],ROBOTWIN50_TASKS)
        base=['torchrun','--standalone','--nproc_per_node=8',str(ROOT/'scripts/train_loopwam_robotwin_full.py'),
            '--shared-dir',cfg['shared_dir'],'--init-artifact',cfg['init_artifact'],'--epochs','10','--seed','42','--global-batch','128',
            '--microbatch','8','--compile','--no-checkpoint-blocks','--lr-backbone','3e-5','--lr-robot-io','3e-5','--warmup-fraction','0',
            '--save-epochs','6,7,8,9,10','--continuation-from',str(recovery)]
        smoke=out/'continuation_smoke'
        stage('smoke6',shell(cfg['train_env'],base+['--output-dir',str(smoke),'--max-updates','237312']))
        stage('resume6',shell(cfg['train_env'],base+['--output-dir',str(smoke),'--resume','--max-updates','237314']))
        audit=read(smoke/'resume_audit_u237312.json');st=read(smoke/'timing.json')
        if not audit['optimizer_exact'] or st['update']!=237314 or st['windows_seen']!=30375515+4*128:raise ValueError('Continuation smoke/resume failed')
        dest=Path(cfg['continuation_dir']);cmd=base+['--output-dir',str(dest)]
        if (dest/'latest.pt').exists() and not (dest/'COMPLETE').exists():cmd+=['--resume']
        stage('train10',shell(cfg['train_env'],cmd))
        final=read(dest/'COMPLETE')
        if (final['status'],final['update'],final['windows_seen'])!=('complete',474620,60751030):raise ValueError('Continuation budget incomplete')
        for e in range(6,11):
            if not (dest/'checkpoints'/f'epoch_{e:02d}.pt').is_file():raise ValueError('Missing epoch snapshot')
        write(out/'COMPLETE.json',dict(passed=True,final=final,eval_dir=cfg['eval_dir'],finished=time.time()));heartbeat('complete')
    except BaseException as e:
        heartbeat('failed',error=repr(e));raise

if __name__=='__main__':main()
