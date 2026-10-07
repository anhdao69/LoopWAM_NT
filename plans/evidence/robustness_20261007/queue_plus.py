#!/usr/bin/env python3
"""Queue Plus behind Pro in the existing two-GPU allocation; fail closed."""
import json, subprocess, time
from pathlib import Path
ROOT=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM')
OUT=ROOT/'runs/loopwam_nt/libero_plus_job4728_after_pro_20261007'
PRO=ROOT/'runs/loopwam_nt/libero_pro_job4728_20261007/pipeline_status.json'
OUT.mkdir(parents=True,exist_ok=True)
def status(stage,**extra):
    target=OUT/'queue_status.json'
    tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(stage=stage,updated_unix=time.time(),allocation='4728',gpus=2,**extra),indent=2))
    tmp.replace(target)
def allocation():
    result=subprocess.check_output(['squeue','-h','-j','4728','-o','%T|%u|%N'],text=True).strip()
    if result!='RUNNING|anhdh35|worker-1':
        raise RuntimeError('Expected allocation unavailable: '+result)
try:
    status('waiting_for_pro')
    while True:
        allocation()
        state=json.loads(PRO.read_text())
        if state['stage']=='failed':
            raise RuntimeError('Pro failed: '+str(state.get('error')))
        steps=subprocess.check_output(['squeue','--steps','-h','-j','4728','-o','%i'],text=True).split()
        if state['stage']=='complete' and '4728.9' not in steps:
            if any(x not in ('4728.0','4728.extern','4728.batch') for x in steps):
                raise RuntimeError('Unexpected active step: '+str(steps))
            break
        time.sleep(30)
    status('launching_plus')
    with (OUT/'launcher.log').open('a') as log:
        result=subprocess.run(['srun','--jobid=4728','--job-name=test_training','--overlap','--exact','-N1','-n1','-c24','--gres=gpu:2','bash',str(OUT/'run.sh')],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Plus launcher exit '+str(result.returncode))
    if json.loads((OUT/'pipeline_status.json').read_text())['stage']!='complete':
        raise RuntimeError('Plus pipeline did not complete')
    status('complete')
except BaseException as error:
    status('failed',error=repr(error))
    raise
