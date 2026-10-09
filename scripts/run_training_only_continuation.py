#!/usr/bin/env python3
"""Finish live training, then hand off to two training-only stages without inference."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PIN='3a89bfac5cfccad66bfebbe00014f7f739a78adc'
EXPECTED={'concat':['full_41','full_33','full_dense12'],
          'mix':['full_14','full_22','full_dense30']}


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def remaining_specs(queue,queues):
    stages=queues[queue][1:]
    if [s['label'] for s in stages]!=EXPECTED[queue]:raise ValueError('Unexpected queue ordering')
    return stages[1:]


def require_ddp_configs(specs,configs):
    if any(configs[s['label']]['backend']!='ddp' for s in specs):
        raise ValueError('This continuation supports only the verified DDP configurations')


def validate_controller(argv,env,uid,expected_uid,job,queue,root):
    def arg(flag):
        try:return argv[argv.index(flag)+1]
        except (ValueError,IndexError):return None
    if (uid!=expected_uid or env.get('SLURM_JOB_ID')!=str(job)
        or 'scripts/run_kv_campaign.py' not in argv or arg('--queue')!=queue
        or arg('--output-root')!=str(root) or arg('--phase')!='run'):
        raise ValueError('Controller ownership/job/command mismatch')


def validate_checkpoint(payload,manifest):
    state=payload.get('training_state') or {}
    expected=dict(update=manifest['planned_updates'],windows_seen=manifest['planned_windows'],
                  epoch=manifest['epochs']-1,next_micro=math.ceil(manifest['train_windows']/(manifest['world_size']*manifest['microbatch'])),backend='ddp')
    if (payload.get('step')!=manifest['planned_updates'] or not payload.get('optimizer',{}).get('state')
        or any(state.get(k)!=v for k,v in expected.items())
        or len(state.get('rng',[]))!=manifest['world_size']
        or payload.get('version')!=manifest['version']
        or payload.get('video_loops')!=manifest['loops']
        or payload.get('action_loops')!=manifest['action_core_loops']
        or payload.get('action_kv_mode','aligned')!=manifest['action_kv_mode']):
        raise ValueError('Incomplete/mismatched final checkpoint or missing optimizer/RNG state')


def validate_handoff(record,job,queue,revision,root,checkpoint_hash):
    expected=dict(status='complete',job=job,queue=queue,source_revision=revision,
                  predecessor_root=str(root),checkpoint_sha256=checkpoint_hash)
    if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('Handoff incomplete or identity/hash mismatch')


def proc_info(pid):
    p=Path('/proc')/str(pid);raw=(p/'stat').read_text().split(') ',1)[1].split()
    return dict(pid=int(pid),uid=p.stat().st_uid,state=raw[0],ppid=int(raw[1]),start=raw[19],
                argv=(p/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                env=dict(s.split('=',1) for s in (p/'environ').read_bytes().decode().split('\0') if '=' in s))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['plan','guard','run'],required=True)
    p.add_argument('--queue',choices=EXPECTED,required=True)
    p.add_argument('--predecessor-job',type=int,required=True)
    p.add_argument('--predecessor-root',type=Path,required=True)
    p.add_argument('--handoff-dir',type=Path,required=True)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--runner-sha256',required=True)
    a=p.parse_args()
    # Import only from the unchanged, verified source checkout supplied in PYTHONPATH.
    from run_kv_campaign import QUEUES,train_command,cache_for,verify_train
    from run_full_libero_v0_job import source_hashes
    prepared=read(a.predecessor_root/'prepared.json')
    def verify_source():
        if (sha(__file__)!=a.runner_sha256 or subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=PIN
            or prepared['source_revision']!=PIN or prepared['source_hashes']!=source_hashes()):
            raise ValueError('Immutable training source changed')
    verify_source();specs=remaining_specs(a.queue,QUEUES);current=QUEUES[a.queue][1]
    require_ddp_configs([current,*specs],prepared['configs'])
    if prepared['queue']!=a.queue:raise ValueError('Prepared queue mismatch')
    predecessor=a.predecessor_root/current['label']/'train'
    handoff=a.handoff_dir/f'job{a.predecessor_job}'
    def completed(path,spec):
        # Original completion/fairness gate; writes its usual fairness evidence.
        timing=verify_train(path,spec)
        import torch
        payload=torch.load(path/'latest.pt',map_location='cpu',mmap=True,weights_only=False)
        validate_checkpoint(payload,read(path/'manifest.json'))
        del payload
        return timing
    if a.phase=='plan':
        out=a.output_root or Path('TRAINING_ONLY_OUTPUT')
        print(json.dumps(dict(predecessor=current['label'],stages=[dict(spec=s,command=train_command(s,prepared['configs'][s['label']],out/s['label']/'train',cache_for(s))) for s in specs],inference=False),indent=2));return
    if a.phase=='guard':
        if os.environ.get('SLURM_JOB_ID')!=str(a.predecessor_job):raise ValueError('Guard must run inside its own predecessor allocation')
        if read(a.predecessor_root/'status.json')['stage']!='train_'+current['label']:raise ValueError('Predecessor no longer in expected training stage')
        found=[]
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:
                info=proc_info(path.name)
                validate_controller(info['argv'],info['env'],info['uid'],os.getuid(),a.predecessor_job,a.queue,a.predecessor_root)
                found.append(info)
            except (ValueError,FileNotFoundError,PermissionError,ProcessLookupError):pass
        if len(found)!=1:raise ValueError('Expected exactly one owned campaign controller')
        controller=found[0];pid=controller['pid'];children=[]
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:
                info=proc_info(path.name)
                if info['ppid']==pid and 'scripts/train_loopwam.py' in info['argv']:children.append(info['pid'])
            except (FileNotFoundError,PermissionError,ProcessLookupError,ValueError):pass
        if len(children)!=1:raise ValueError('Expected exactly one live torchrun child')
        handoff.mkdir(parents=True,exist_ok=False)
        stopped=False
        try:
            os.kill(pid,signal.SIGSTOP);stopped=True
            for _ in range(20):
                if proc_info(pid)['state']=='T':break
                time.sleep(.1)
            if proc_info(pid)['state']!='T':raise ValueError('Controller did not stop')
            write(handoff/'armed.json',dict(job=a.predecessor_job,queue=a.queue,pid=pid,start=controller['start'],training_child=children[0],armed_unix=time.time(),controller_only_stopped=True))
            print('Guard armed; controller stopped; training child continues',flush=True)
            while True:
                verify=proc_info(pid)
                if verify['start']!=controller['start'] or verify['state']!='T':raise ValueError('Controller identity/state changed')
                try:timing=read(predecessor/'timing.json')
                except json.JSONDecodeError:time.sleep(2);continue
                if timing['status']=='complete':break
                try:child=proc_info(children[0])
                except FileNotFoundError:raise RuntimeError('Training child exited before completion')
                if child['state']=='Z':raise RuntimeError('Training child failed before completion')
                time.sleep(10)
            verify_source();timing=completed(predecessor,current)
            record=dict(status='complete',job=a.predecessor_job,queue=a.queue,source_revision=PIN,
                        predecessor_root=str(a.predecessor_root),checkpoint_sha256=sha(predecessor/'latest.pt'),
                        checkpoint=str(predecessor/'latest.pt'),training_timing=timing,completed_unix=time.time())
            write(handoff/'complete.json',record)
            print('Final checkpoint verified; releasing old allocation without evaluation',flush=True)
            subprocess.run(['scancel',str(a.predecessor_job)],check=True)
        except BaseException as exc:
            write(handoff/'failed.json',dict(error=repr(exc),unix=time.time(),controller_stopped=stopped))
            # Keep live training intact; the stopped controller cannot start inference.
            # A failed monitor leaves a failure record and requires operator recovery.
            raise
    else:
        if a.output_root is None:raise ValueError('Output root required')
        if not (handoff/'complete.json').exists() or (handoff/'failed.json').exists():raise ValueError('Successful predecessor handoff required')
        record=read(handoff/'complete.json')
        validate_handoff(record,a.predecessor_job,a.queue,PIN,a.predecessor_root,sha(predecessor/'latest.pt'))
        completed(predecessor,current)
        a.output_root.mkdir(parents=True,exist_ok=False)
        write(a.output_root/'manifest.json',dict(source_revision=PIN,queue=a.queue,job=os.environ.get('SLURM_JOB_ID'),predecessor=record,prepared_sha256=sha(a.predecessor_root/'prepared.json'),runner_sha256=sha(__file__),inference=False,stages=specs,configs={s['label']:prepared['configs'][s['label']] for s in specs}))
        try:
            for spec in specs:
                verify_source();label=spec['label'];dest=a.output_root/label/'train'
                command=train_command(spec,prepared['configs'][label],dest,cache_for(spec))
                if any('evaluate' in arg or 'latency' in arg for arg in command) or '--resume' in command:raise ValueError('Only fresh next-stage training is allowed')
                write(a.output_root/'status.json',dict(stage='train_'+label,command=command,unix=time.time()))
                with (a.output_root/f'{label}.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                timing=completed(dest,spec);write(a.output_root/f'{label}_complete.json',timing)
            write(a.output_root/'status.json',dict(stage='complete',inference=False,unix=time.time()))
        except BaseException as exc:
            write(a.output_root/'status.json',dict(stage='failed',error=repr(exc),unix=time.time()));raise


if __name__=='__main__':main()
