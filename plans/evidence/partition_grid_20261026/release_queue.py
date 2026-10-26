"""Release only fully checked native smokes, then execute the authorized queue."""
import hashlib,json,subprocess,time
from pathlib import Path
from run_partition_grid import GRID,QUEUES,hash_sources,require_sources,verify,read,write
parent=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_grid_20261026')
root=parent/'preflight_r2'
expected=hash_sources()
for name,count in [('cpu_helper_fix_tests.log','449 passed, 16 skipped'),('gpu_helper_fix_tests.log','465 passed')]:
    text=(parent/name).read_text()
    if count not in text or 'FAILED ' in text:raise ValueError(f'Test gate failed: {name}')
while not all((root/f'smoke_{lane}_complete.json').is_file() for lane in QUEUES):
    if list(root.glob('*_failed.json')):raise RuntimeError('Native preflight failed; no production launch')
    require_sources(expected)
    time.sleep(10)
config=read(root/'selected.json')['config']
for lane in QUEUES:
    evidence=read(root/f'smoke_{lane}_complete.json')
    if evidence['sources']!=expected or evidence['config']!=config or evidence['labels']!=list(QUEUES[lane]):
        raise ValueError('Native source/config/coverage mismatch')
timings={}
for label,(partition,loops) in GRID.items():
    d=root/'smoke'/label
    timings[label]=verify(label,d,True)
    infer=read(d/'inference_check.json')
    if (not infer['passed'] or not infer['finite'] or not infer['repeat_bit_identical'] or infer['partition']!=list(partition)
        or infer['loops']!=loops or infer['effective_depth']!=30 or infer['parameters']!=584536135):
        raise ValueError(f'Native inference gate failed: {label}')
require_sources(expected)
if (root/'release.json').exists():raise FileExistsError('Already released')
now=time.time()
hours={lane:sum(timings[label]['measured_mean_update_seconds']*21700/3600 for label in labels) for lane,labels in QUEUES.items()}
release=dict(sources=expected,config=config,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             released_unix=now,queues=QUEUES,native_timings=timings,projected_training_hours=hours,
             tests={name:hashlib.sha256((parent/name).read_bytes()).hexdigest() for name in ['cpu_helper_fix_tests.log','gpu_helper_fix_tests.log']},
             note='Training-time projection excludes setup/checkpoint/upload overhead; no final SR claimed')
write(root/'release.json',release)
with (root/'production_launch.log').open('x') as log:
    result=subprocess.run(['srun','--jobid=5248','--overlap','--ntasks=1','--cpus-per-task=48','--gres=gpu:4','bash',str(parent/'pair_phase.sh'),'run',str(root)],stdout=log,stderr=subprocess.STDOUT)
if result.returncode:raise RuntimeError('Production step failed; inspect lane status and failure files')
write(root/'queue_complete.json',dict(completed_unix=time.time(),source_revision=release['source_revision']))
