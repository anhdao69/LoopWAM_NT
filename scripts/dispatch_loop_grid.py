#!/usr/bin/env python3
"""Run independent two-GPU loop experiments within one Slurm step."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',required=True,help='Comma separated video/action pairs, e.g. 2/2,1/4')
    p.add_argument('--output-root',required=True)
    p.add_argument('--production-cache',required=True)
    a=p.parse_args()
    pairs=[tuple(map(int,x.split('/'))) for x in a.pairs.split(',')]
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(visible)!=2*len(pairs) or len(set(visible))!=len(visible) or '' in visible:
        raise ValueError(f'Need exactly two distinct allocated GPUs per experiment: {visible}')
    if not os.getenv('SLURM_JOB_ID'):raise RuntimeError('Must run within the authorized Slurm allocation')
    root=Path(a.output_root);root.mkdir(parents=True,exist_ok=False)
    cpus=sorted(os.sched_getaffinity(0));children=[]
    if len(cpus)<16*len(pairs):raise RuntimeError('Insufficient allocated CPUs')
    resources=[]
    for i,(video,action) in enumerate(pairs):
        gpu=visible[2*i:2*i+2];cpu=cpus[i*len(cpus)//len(pairs):(i+1)*len(cpus)//len(pairs)]
        out=root/f'v{video}a{action}'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(gpu),OMP_NUM_THREADS='4',LP_NUM_THREADS='4')
        cmd=[sys.executable,'scripts/autotune_loop_grid.py','--video-loops',str(video),'--action-loops',str(action),
            '--output-root',str(out),'--production-cache',a.production_cache]
        log=(root/f'v{video}a{action}.log').open('x')
        process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
            preexec_fn=lambda cpu=cpu:os.sched_setaffinity(0,cpu))
        children.append((process,log))
        resources.append(dict(video_loops=video,action_loops=action,gpus=gpu,cpus=cpu,pid=process.pid))
    (root/'dispatch.json').write_text(json.dumps(dict(job=os.getenv('SLURM_JOB_ID'),step=os.getenv('SLURM_STEP_ID'),
        started_unix=time.time(),resources=resources),indent=2))
    codes=[]
    for child,log in children:codes.append(child.wait());log.close()
    (root/'dispatch_result.json').write_text(json.dumps(dict(exit_codes=codes,finished_unix=time.time())))
    if any(codes):raise SystemExit(1)

if __name__=='__main__':main()
