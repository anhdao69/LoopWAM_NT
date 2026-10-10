#!/usr/bin/env python3
"""Experiment tracking table for every ChronoLoop run (markdown to stdout, JSON with --json)."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from chronoloop_experiments import EXPERIMENTS

RUNS = Path(os.environ.get('CHRONO_RUNS', '/groups/yshang/an221229/checkpoints/ChronoLoop/runs'))


def slurm_jobs():
    try:
        out = subprocess.check_output(['squeue', '-h', '-u', os.environ.get('USER', ''), '-o', '%i|%j|%T|%M|%R'], text=True)
    except Exception:
        return {}
    jobs = {}
    for line in out.splitlines():
        i, name, state, elapsed, reason = line.split('|')
        jobs.setdefault(name, []).append(f'{i} {state} {elapsed if state == "RUNNING" else reason}')
    return jobs


def row(name, spec, jobs):
    d = RUNS / spec['hf']
    timing = json.loads((d / 'timing.json').read_text()) if (d / 'timing.json').exists() else {}
    manifest = json.loads((d / 'manifest.json').read_text()) if (d / 'manifest.json').exists() else {}
    upload = json.loads((d / 'hf_upload.json').read_text()) if (d / 'hf_upload.json').exists() else {}
    ckpts = sorted(p.name for p in (d / 'checkpoints').glob('*.pt')) if (d / 'checkpoints').exists() else []
    subs = (RUNS / 'submissions.txt').read_text().splitlines() if (RUNS / 'submissions.txt').exists() else []
    submitted = [s.split()[0] for s in subs if len(s.split()) > 1 and s.split()[1] == name]
    conts = (d / 'continuations.txt').read_text().split() if (d / 'continuations.txt').exists() else []
    interactive = sorted(p.name.split('_')[1].split('.')[0] for p in d.glob('interactive_*.log'))
    status = 'complete' if (d / 'COMPLETE').exists() else timing.get('status', 'not started')
    if status == 'running' and time.time() - timing.get('heartbeat_unix', 0) > 1800:
        status = 'stopped (resumable)' if (d / 'latest.pt').exists() else 'stopped'
    return dict(run=name, flags=spec['flags'], hf_folder=spec['hf'], job_name=spec['job'], status=status,
                update=timing.get('update'), planned_updates=timing.get('planned_updates'),
                training_hours=round(timing.get('elapsed_training_seconds', 0) / 3600, 2),
                world_size=manifest.get('world_size'), checkpoints=ckpts,
                slurm=jobs.get(spec['job'], []), submitted_jobs=submitted + conts, interactive_jobs=interactive,
                hf_verified_epochs=upload.get('epochs_verified', []), hf_url=upload.get('url'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--json', action='store_true'); a = p.parse_args()
    jobs = slurm_jobs()
    rows = [row(n, s, jobs) for n, s in EXPERIMENTS.items()]
    if a.json:
        print(json.dumps(rows, indent=2)); return
    print('| Run | Flags (mem/src/write/K_a/hist) | Status | Update | Train h | GPUs | Slurm (live) | Jobs submitted | Interactive | Checkpoints | HF verified |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        f = r['flags']
        flags = f"{f['memory_tokens']}/{f['mem_source']}/{f['mem_write']}/{f['action_loops']}/{f['history_frame']}"
        upd = f"{r['update']}/{r['planned_updates']}" if r['update'] is not None else '-'
        print(f"| {r['run']} | {flags} | {r['status']} | {upd} | {r['training_hours']} | {r['world_size'] or '-'} | "
              f"{'; '.join(r['slurm']) or '-'} | {', '.join(r['submitted_jobs']) or '-'} | {', '.join(r['interactive_jobs']) or '-'} | "
              f"{', '.join(r['checkpoints']) or '-'} | {', '.join(r['hf_verified_epochs']) or '-'} |")


if __name__ == '__main__':
    main()
