#!/usr/bin/env python3
"""Preflight, full robustness evaluation, and checked aggregate results."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_dense_v2_job import Pipeline, read_json, write_json
from scripts.evaluate_loopwam_robustness import make_tasks, sha256_file


def verify(directory, world, checkpoint_hash, expected=None):
    rows=[]
    for rank in range(world):
        done=read_json(directory/f'done_rank{rank}.json')
        part=[json.loads(x) for x in (directory/f'rank{rank}.jsonl').read_text().splitlines()]
        if done['completed']!=len(part) or done['checkpoint_sha256']!=checkpoint_hash:
            raise ValueError('Rank completion mismatch')
        rows.extend(part)
    keys=[(r['suite'],r['task_id'],r['episode_index']) for r in rows]
    if len(set(keys))!=len(keys) or expected is not None and set(keys)!=expected:
        raise ValueError('Duplicate or missing episodes')
    for row in rows:
        video=Path(row['video'])
        if row['checkpoint_sha256']!=checkpoint_hash or not video.is_file() or not video.stat().st_size:
            raise ValueError('Checkpoint mismatch or missing video')
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--benchmark',choices=['pro','plus'],required=True)
    p.add_argument('--world',type=int,required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    out=Path(a.output).resolve()
    if (out/'pipeline_status.json').exists():
        raise ValueError('Use fresh output; this queue never implicitly resumes')
    pipeline=Pipeline(out)
    try:
        checkpoint_hash=sha256_file(a.checkpoint)
        tasks=make_tasks(a.benchmark)
        if len(tasks)!=(200 if a.benchmark=='pro' else 10030):
            raise ValueError('Official benchmark task coverage changed')
        base=['scripts/evaluate_loopwam_robustness.py','--benchmark',a.benchmark,
              '--checkpoint',a.checkpoint,'--text-cache',str(out/'text_cache')]
        pipeline.run('prepare_text',['python']+base+['--output',str(out/'text_prepare'),'--prepare-text'])
        torchrun=['torchrun','--standalone',f'--nproc_per_node={a.world}']
        preflight=out/'preflight'
        pipeline.run('preflight',torchrun+base+['--output',str(preflight),'--preflight'])
        smoke=verify(preflight,a.world,checkpoint_hash)
        categories={r['category'] for r in smoke}
        if len(categories)!=(5 if a.benchmark=='pro' else 7):
            raise ValueError('Preflight did not execute every perturbation category')
        full=out/'evaluation'
        episodes=10 if a.benchmark=='pro' else 1
        expected={(name,index,ep) for _,name,index,_,_ in tasks for ep in range(episodes)}
        pipeline.status('preflight_verified',planned_episodes=len(expected),categories=sorted(categories))
        pipeline.run('evaluation',torchrun+base+['--output',str(full)])
        rows=verify(full,a.world,checkpoint_hash,expected)
        if sha256_file(a.checkpoint)!=checkpoint_hash:
            raise ValueError('Checkpoint changed during evaluation')
        groups=defaultdict(list)
        for row in rows:
            groups[row['category']].append(row)
        result=dict(benchmark=a.benchmark,checkpoint_sha256=checkpoint_hash,total_episodes=len(rows),
            successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/len(rows),
            categories={key:dict(episodes=len(v),success_rate=sum(r['success'] for r in v)/len(v)) for key,v in groups.items()},
            episodes=rows)
        write_json(out/'summary.json',result)
        pipeline.status('complete',success_rate=result['success_rate'],episodes=len(rows))
    except BaseException as exc:
        pipeline.status('failed',error=repr(exc))
        raise


if __name__=='__main__':main()
