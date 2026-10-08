#!/usr/bin/env python3
"""Measure two-GPU configurations, validate native training/rollouts, start fresh run."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import run_loop_grid_job as job
from scripts.run_dense_v2_job import verify_benchmark_sources


def validate_result(r, video, action):
    if (r['version']!='v0' or r['world_size']!=2 or r['loops']!=video or r['action_loops']!=action
        or r['global_batch']!=128 or r['microbatch']*2*r['accumulation']!=128
        or r['precision']['policy_dtype']!='float32' or r['precision']['optimizer_moment_dtypes']!=['torch.float32']
        or not r['fused'] or not r['structured_attention'] or len(r['records'])<5
        or not math.isfinite(r['steady_seconds']) or r['steady_seconds']<=0
        or any(not math.isfinite(row[k]) for row in r['records'] for k in ('loss','grad_norm','seconds'))):
        raise ValueError('Invalid benchmark contract or nonfinite result')
    verify_benchmark_sources(r,job.source_hashes())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video-loops',type=int,required=True)
    p.add_argument('--action-loops',type=int,required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--production-cache',required=True)
    a=p.parse_args()
    job.VIDEO_LOOPS,job.ACTION_LOOPS=a.video_loops,a.action_loops
    pipeline=job.Pipeline(a.output_root)
    root=Path(a.output_root)
    if (root/'resource_manifest.json').exists():raise ValueError('Output already used')
    job.write_json(root/'resource_manifest.json',dict(video_loops=a.video_loops,action_loops=a.action_loops,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cpu_affinity=sorted(os.sched_getaffinity(0)),
        gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used','--format=csv,noheader'],text=True)))
    results=[]
    def bench(backend,mb,updates=8,suffix=''):
        name=f'benchmark_{backend}_mb{mb}{suffix}'
        cmd=job.TORCHRUN+['scripts/benchmark_loopwam.py','--version','v0','--video-loops',str(a.video_loops),
            '--action-loops',str(a.action_loops),'--backend',backend,'--microbatch',str(mb),
            '--workers','8','--structured-attention','--updates',str(updates),'--latent-cache-dir',a.production_cache,
            '--output-dir',str(root/name)]
        if pipeline.run(name,cmd,allow_oom=True):
            r=job.read_json(root/name/'result.json');validate_result(r,a.video_loops,a.action_loops)
            results.append(r);return r
        return None
    try:
        for mb in (8,16,32):
            if bench('ddp',mb) is None:break
        if not results:raise RuntimeError('No DDP configuration fit')
        best=min(results,key=lambda r:r['steady_seconds'])
        for backend in ('zero1','zero2'):bench(backend,best['microbatch'])
        fastest=min(results,key=lambda r:r['steady_seconds'])
        confirmed=bench(fastest['backend'],fastest['microbatch'],20,'_confirm')
        if confirmed is None:raise RuntimeError('Selected configuration failed confirmation')
        candidate={k:confirmed[k] for k in ('backend','microbatch','workers','checkpoint_blocks')}
        job.write_json(root/'benchmark_selection.json',dict(selected=candidate,results=results))
        pipeline.prepare(candidate,a.production_cache)
        baseline=job.read_json('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_scratch_fast_bs128_20261005/manifest.json')
        manifest=job.read_json(root/'warm_train/manifest.json')
        for key in ('seed','global_batch','epochs','train_windows','planned_updates','planned_windows','asset_sha256'):
            if baseline[key]!=manifest[key]:raise ValueError(f'Fairness contract mismatch: {key}')
        for key in ('train_episodes','validation_episodes','normalization_sha256','camera_order','video_offsets','action_offsets','content_files'):
            if baseline['data'][key]!=manifest['data'][key]:raise ValueError(f'Fairness data mismatch: {key}')
        prepared=job.read_json(root/'prepared.json')
        prepared['estimated_training_hours']=(prepared['warm_seconds']*7250+prepared['estimated_checkpoint_seconds'])/3600
        prepared['note']='Full frozen cache already populated: 7250 warm updates plus ten checkpoint saves; excludes final rollouts and setup.'
        job.write_json(root/'prepared.json',prepared)
        job.write_json(root/'fairness_verified.json',dict(reference='v0_scratch_fast_bs128_20261005',verified=True))
        pipeline.production()
    except BaseException as exc:
        pipeline.status('failed',error=repr(exc));raise

if __name__=='__main__':main()
