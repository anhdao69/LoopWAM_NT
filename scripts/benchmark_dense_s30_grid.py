#!/usr/bin/env python3
"""Compare Dense-S30 backends at global128 on an existing two-H100 allocation."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',required=True)
    args=parser.parse_args()
    out=Path(args.output_root).resolve()
    out.mkdir(parents=True,exist_ok=False)
    cache=out/'latents'
    trials=[('ddp',4,False,'cold'),('ddp',4,False,'warm'),
            ('ddp',8,False,'warm'),('zero1',8,False,'warm'),
            ('zero2',8,False,'warm'),('zero1',16,False,'warm'),
            ('zero2',16,False,'warm'),('zero1',16,True,'warm'),
            ('zero1',32,True,'warm')]
    results=[]
    for backend,microbatch,checkpoint,kind in trials:
        name=f'{backend}_mb{microbatch}_cp{int(checkpoint)}_{kind}'
        command=['torchrun','--standalone','--nproc_per_node=2','scripts/benchmark_loopwam.py',
            '--version','dense_s30','--backend',backend,'--microbatch',str(microbatch),
            '--workers','4','--structured-attention','--latent-cache-dir',str(cache),
            '--updates','7','--output-dir',str(out/name)]
        if checkpoint: command+=['--checkpoint-blocks']
        (out/'status.json').write_text(json.dumps(dict(stage=name,command=command,time=time.time())))
        print('START',name,flush=True)
        started=time.time()
        with (out/(name+'.log')).open('x') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            if 'CUDA out of memory' not in (out/(name+'.log')).read_text():
                raise RuntimeError(f'Unexpected benchmark failure: {name}')
            row=dict(name=name,oom=True,exit_code=result.returncode)
        else:
            row=json.loads((out/name/'result.json').read_text())
            if row['policy_parameters']!=1416114247 or row['effective_depth']!=30 or row['physical_depth']!=30:
                raise ValueError('Dense-S30 architecture differs from planned control')
            row.update(name=name,oom=False,exit_code=0)
        row['wall_seconds']=time.time()-started
        results.append(row)
        (out/'results.json').write_text(json.dumps(results,indent=2))
        print('DONE',name,row.get('steady_seconds'),'oom',row['oom'],flush=True)
    candidates=sorted((r for r in results if not r['oom'] and r['name'].endswith('_warm')),
                      key=lambda r:r['steady_seconds'])
    if not candidates: raise RuntimeError('No Dense-S30 benchmark configuration succeeded')
    (out/'selection.json').write_text(json.dumps(dict(selected=candidates[0],candidates=candidates),indent=2))
    (out/'status.json').write_text(json.dumps(dict(stage='complete',time=time.time())))


if __name__=='__main__':
    main()
