"""One-shot release of two verified four-GPU training queues; never auto-restarts."""
import json, subprocess, time
from pathlib import Path
from run_centered_partition_grid import GRID, QUEUES, hash_sources, require_sources, verify, read, write
R=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_centered_20261026')
sources=hash_sources()
revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert revision.startswith('96dc3f9')
assert '473 passed, 16 skipped' in (R/'cpu_final_tests.log').read_text()
assert '489 passed' in (R/'gpu_final_tests.log').read_text()
audit=read(R/'native_donor_audit.json')
assert audit['passed'] and len(audit['runs'])==6 and all(x['exact_equal'] and x['policy_parameters']==584536135 for x in audit['runs'])
config=read(R/'selected.json')['config']
trials=read(R/'bench_A.json')+read(R/'clean_benchmark/bench_B.json')
assert len(trials)==10
assert not list(R.glob('*_failed.json'))
assert not list((R/'clean_benchmark').glob('*_failed.json'))
assert min((x for x in trials if x['status']=='passed'),key=lambda x:x['seconds'])['config']==config
projection={}
for lane in QUEUES:
    accepted=read(R/f'smoke_{lane}_complete.json')
    assert accepted['sources']==sources and accepted['config']==config and accepted['labels']==list(QUEUES[lane])
    for label in QUEUES[lane]:
        result=verify(label,R/'smoke'/label,True)
        passed=read(R/'smoke'/label/'passed.json')
        assert passed['inference']['passed'] and passed['inference']['repeat_bit_identical']
        projection[label]=result
require_sources(sources)
assert not (R/'release.json').exists()
write(R/'release.json',dict(sources=sources,source_revision=revision,config=config,queues=QUEUES,allocations={'A':5248,'B':4925},gpus_per_run=4,smoke_timing=projection,released_unix=time.time(),native_audit=audit))
processes=[]
for lane,job in [('A',5248),('B',4925)]:
    command=['srun',f'--jobid={job}','--overlap','--ntasks=1','--cpus-per-task=48','--gres=gpu:4','--job-name=test_training','bash',str(R/'lane.sh'),lane,'run']
    write(R/f'production_{lane}_launch.json',dict(command=command,time=time.time()))
    log=(R/f'production_{lane}_driver.log').open('x')
    processes.append((lane,subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)))
for lane,p in processes:
    code=p.wait()
    write(R/f'production_{lane}_exit.json',dict(returncode=code,time=time.time()))
