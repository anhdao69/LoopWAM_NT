#!/usr/bin/env python3
"""Native four-H100 dense/v2 throughput grid. Use a fresh output directory."""
import argparse, json, os, subprocess, time
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--output-root',required=True)
args=parser.parse_args()
out=Path(args.output_root).resolve()
out.mkdir(exist_ok=True)
cache=out/'benchmark_latents'
trials=[('dense_s12','ddp',4,'fill')]
trials += [('dense_s12',b,m,'warm') for b in ('ddp','zero1','zero2') for m in (8,16,32)]
trials += [('v2',b,m,'warm') for b in ('ddp','zero1','zero2') for m in ((4,8,16) if b=='ddp' else (8,16))]
for v,b,m,kind in trials:
 name=f'{v}_{b}_mb{m}_{kind}'
 cmd=['torchrun','--standalone','--nproc_per_node=4','scripts/benchmark_loopwam.py','--version',v,'--backend',b,'--microbatch',str(m),'--structured-attention','--latent-cache-dir',str(cache),'--updates','5','--output-dir',str(out/name)]
 (out/'benchmark_status.json').write_text(json.dumps(dict(stage=name,command=cmd,start=time.time())))
 print('START',name,flush=True)
 with (out/(name+'.log')).open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:
  log=(out/(name+'.log')).read_text()
  if 'CUDA out of memory' not in log: raise RuntimeError(f'Unexpected trial failure {name}; inspect log')
  (out/(name+'.oom')).write_text(str(r.returncode)); print('OOM',name,flush=True)
 else:print('DONE',name,json.loads((out/name/'result.json').read_text())['steady_seconds'],flush=True)
(out/'benchmark_status.json').write_text(json.dumps(dict(stage='complete',end=time.time())))
