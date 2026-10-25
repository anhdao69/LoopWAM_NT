#!/usr/bin/env python3
"""Success-rate table for ChronoLoop RoboTwin evaluations (scripts/evaluate_chronoloop_robotwin.py outputs).

python scripts/chronoloop_robotwin_eval_report.py CL-0=EVAL_DIR_A CL-A=EVAL_DIR_B [--json OUT.json] [--tasks a,b,...]

A scene seed gives the same scene and instruction for every model, but the set of accepted seeds can differ by a few
between runs: the expert check (CuRobo) does not accept exactly the same borderline seeds every time. Models are
therefore compared on the seeds that all of them finished ("matched"). Split jobs (TASK__CONFIG__fromKK) are merged;
a seed evaluated twice (split overlap) counts once, at its first episode. Each model's own total is listed too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_chronoloop_robotwin import TASKS, episodes, job_dirs  # noqa: E402

CONFIGS = ('demo_clean', 'demo_randomized')


def job_episodes(root, task, config):
    """{seed: success} over all parts of one job, in episode order; a repeated seed keeps its first episode."""
    out = {}
    for d in job_dirs(Path(root), task, config):
        for seed, success in episodes(d / 'eval.log'):
            out.setdefault(seed, success)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', help='NAME=EVAL_DIR')
    p.add_argument('--json', default=None)
    p.add_argument('--tasks', default=','.join(TASKS), help='comma-separated RoboTwin tasks (default: the hard 5)')
    a = p.parse_args()
    runs = dict(r.split('=', 1) for r in a.runs)
    names = list(runs)
    rows, record = [], {}
    tasks = a.tasks.split(',')
    for c in CONFIGS:
        for t in tasks:
            eps = {n: job_episodes(d, t, c) for n, d in runs.items()}
            matched = sorted(set.intersection(*(set(e) for e in eps.values())))
            cell = {n: dict(matched=sum(eps[n][k] for k in matched), k=len(matched),
                            total=sum(eps[n].values()), n=len(eps[n])) for n in names}
            record[f'{t}/{c}'] = dict(cell, success_by_seed={n: dict(sorted(eps[n].items())) for n in names})
            rows.append((c, t, len(matched), cell))
    pct = lambda s, n: f'{s}/{n} ({100 * s / n:.0f}%)' if n else '-'
    print('| Setting | Task | Matched scenes | ' + ' | '.join(f'{n} (matched)' for n in names) + ' | '
          + ' | '.join(f'{n} (all finished)' for n in names) + ' |')
    print('|---|---|---:|' + '---:|' * (2 * len(names)))
    for c, t, k, cell in rows:
        print(f'| {c.removeprefix("demo_")} | {t} | {k} | ' + ' | '.join(pct(cell[n]['matched'], k) for n in names)
              + ' | ' + ' | '.join(pct(cell[n]['total'], cell[n]['n']) for n in names) + ' |')
    mean = lambda xs: f'**{100 * sum(xs) / len(xs):.1f}%**' if xs else '-'
    for c in CONFIGS:                                     # LiLa-WAM's "average success rate": mean of task rates
        sub = [(k, cell) for cc, _, k, cell in rows if cc == c]
        print(f'| {c.removeprefix("demo_")} | **mean of {len(tasks)} tasks** | {sum(k for k, _ in sub)} | '
              + ' | '.join(mean([cell[n]['matched'] / k for k, cell in sub if k]) for n in names) + ' | '
              + ' | '.join(mean([cell[n]['total'] / cell[n]['n'] for _, cell in sub if cell[n]['n']]) for n in names)
              + ' |')
    if a.json:
        Path(a.json).write_text(json.dumps(dict(runs=runs, cells=record), indent=2))


if __name__ == '__main__':
    main()
