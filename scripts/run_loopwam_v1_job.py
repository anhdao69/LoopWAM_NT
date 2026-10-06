#!/usr/bin/env python3
"""Benchmark -> v1 smoke/inference preflight -> fresh ten epochs -> rollouts."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def select_microbatch(results):
    eligible=[]
    for result in results:
        if result.get('version')!='v1' or result.get('backend')!='ddp':
            raise ValueError('Selection requires actual v1 DDP measurements')
        if result['global_batch']!=128 or result['microbatch']*2*result['accumulation']!=128:
            raise ValueError('Benchmark changed the global batch')
        seconds=result['steady_seconds']
        if not math.isfinite(seconds) or seconds<=0 or len(result['records'])<5:
            raise ValueError('Incomplete or invalid benchmark')
        if any(not math.isfinite(r['loss']) or not math.isfinite(r['grad_norm']) for r in result['records']):
            raise ValueError('Nonfinite benchmark')
        eligible.append(result)
    if not eligible: raise ValueError('No successful v1 microbatch candidates')
    return min(eligible,key=lambda r:r['steady_seconds'])


def verify_finished_training(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    timing=json.loads((directory/'timing.json').read_text())
    state=json.loads((directory/'trainer_state.json').read_text())
    if manifest['version']!='v1' or manifest['resume'] is not None or manifest['epochs']!=10 or manifest['global_batch']!=128:
        raise ValueError('Production training was not fresh ten-epoch global128 v1')
    if manifest['planned_updates']!=7250 or manifest['planned_windows']!=926780:
        raise ValueError('Unexpected LIBERO-Long budget')
    if timing['status']!='complete' or timing['completed_updates']!=7250 or state['update']!=7250 or state['windows_seen']!=926780:
        raise ValueError('Training incomplete; refusing final inference')
    if not (directory/'latest.pt').is_file(): raise FileNotFoundError('Final checkpoint missing')
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-root',required=True)
    p.add_argument('--episodes-per-task',type=int,default=10)
    a=p.parse_args()
    if a.episodes_per_task<1: p.error('episodes-per-task must be positive')
    out=Path(a.output_root).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'job_status.json').exists(): raise ValueError('Use a fresh output root; production never resumes benchmark weights')
    started=time.time()
    state=dict(slurm_job_id=os.getenv('SLURM_JOB_ID'),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),started_unix=started)
    def status(stage,**kwargs):
        state.update(stage=stage,elapsed_seconds=time.time()-started,**kwargs)
        temporary=out/'job_status.json.tmp'; temporary.write_text(json.dumps(state,indent=2)); temporary.replace(out/'job_status.json')
        print(json.dumps(state),flush=True)
    def run(stage,command,allow_oom=False):
        status(stage,command=command)
        logfile=out/(stage+'.log')
        with logfile.open('x') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            text=logfile.read_text()
            if allow_oom and ('torch.OutOfMemoryError: CUDA out of memory' in text or 'CUDA out of memory.' in text):
                status(stage+'_oom',exit_code=result.returncode)
                return False
            raise RuntimeError(f'{stage} failed with exit {result.returncode}; inspect {logfile}')
        return True
    torchrun=['torchrun','--standalone','--nproc_per_node=2']
    try:
        cache=str(out/'benchmark_latents')
        base=torchrun+['scripts/benchmark_loopwam.py','--version','v1','--backend','ddp','--structured-attention','--latent-cache-dir',cache,'--updates','5']
        run('benchmark_fill4',base+['--microbatch','4','--output-dir',str(out/'benchmark_fill4')])
        candidates=[]; failed=[]
        for micro in [4,8,16]:
            name=f'benchmark_warm{micro}'
            if run(name,base+['--microbatch',str(micro),'--output-dir',str(out/name)],allow_oom=True):
                candidates.append(json.loads((out/name/'result.json').read_text()))
            else: failed.append(micro)
        # Validate every candidate before ranking; then verify cold-VAE memory fit.
        select_microbatch(candidates)
        cold_oom=[]
        selected=None
        for candidate in sorted(candidates,key=lambda r:r['steady_seconds']):
            micro=candidate['microbatch']
            train_base=torchrun+['scripts/train_loopwam.py','--config','configs/loopwam_s_v1_fast.yaml','--microbatch',str(micro),'--smoke']
            smoke=out/f'preflight_train_mb{micro}'
            if run(f'training_smoke_mb{micro}',train_base+['--output-dir',str(smoke),'--latent-cache-dir',str(smoke/'latent_cache'),'--max-updates','3','--validation-samples','0'],allow_oom=True):
                selected=candidate
                break
            cold_oom.append(micro)
        if selected is None: raise RuntimeError('Every candidate failed cold training memory fit')
        (out/'selection.json').write_text(json.dumps(dict(selected=selected,successful_candidates=candidates,oom_microbatches=failed,cold_oom_microbatches=cold_oom,selection_scope='DDP microbatch4/8/16, cold-memory validated; fusedFP32 AdamW, BF16compute, structuralattention, exactlatentcache'),indent=2))
        evaluate=torchrun+['scripts/evaluate_loopwam_libero.py','--vae-path','checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth']
        run('inference_smoke',evaluate+['--checkpoint',str(smoke/'latest.pt'),'--stats',str(smoke/'data/dataset_stats.json'),'--output-dir',str(out/'preflight_inference'),'--episodes-per-task','1','--task-ids','0,1','--smoke','--max-steps','20'])
        train=out/'train'
        # Separate directory + no --resume: all benchmark/smoke optimizer states are discarded.
        run('training',train_base+['--output-dir',str(train),'--latent-cache-dir',str(train/'latent_cache')])
        verify_finished_training(train)
        run('inference',evaluate+['--checkpoint',str(train/'latest.pt'),'--stats',str(train/'data/dataset_stats.json'),'--output-dir',str(out/'inference'),'--episodes-per-task',str(a.episodes_per_task)])
        status('complete',microbatch=micro,checkpoint=str(train/'latest.pt'),inference_dir=str(out/'inference'))
    except BaseException as exc:
        status('failed',error=repr(exc))
        raise


if __name__=='__main__': main()
