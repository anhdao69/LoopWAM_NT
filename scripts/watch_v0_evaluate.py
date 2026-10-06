#!/usr/bin/env python3
"""Wait outside Slurm for v0, evaluate on its two GPUs, then release allocation."""
from __future__ import annotations
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from run_dense_v2_job import read_json, write_json, verify_evaluation


def verify_manifest(train, job):
    m = read_json(Path(train) / 'manifest.json')
    expected = dict(version='v0', resume=None, epochs=10, global_batch=128,
                    world_size=2, microbatch=8, gradient_accumulation=8,
                    policy_parameters=584536135, train_windows=92678,
                    planned_updates=7250, planned_windows=926780)
    if any(m.get(k) != v for k, v in expected.items()) or str(m.get('slurm_job_id')) != job:
        raise ValueError('Training manifest does not match the authorized fresh v0 run')
    return m


def verify_finished_training(train, job):
    train = Path(train)
    m = verify_manifest(train, job)
    t, s = read_json(train/'timing.json'), read_json(train/'trainer_state.json')
    if (t.get('status') != 'complete' or t.get('completed_updates') != 7250
            or t.get('windows_seen') != 926780 or s.get('update') != 7250
            or s.get('windows_seen') != 926780 or s.get('epoch') != 9
            or s.get('micro') != math.ceil(92678/(m['world_size']*m['microbatch'])) - 1):
        raise ValueError('Training incomplete; refusing final evaluation and cancellation')
    if not (train/'latest.pt').is_file() or not (train/'latest.pt').stat().st_size:
        raise ValueError('Final training checkpoint missing')


def verify_identity(current, expected, job):
    if (current.get('JobId') != job or current.get('JobState') != 'RUNNING'
            or not re.fullmatch(r'.+\(' + str(os.getuid()) + r'\)', current.get('UserId',''))
            or current.get('StartTime') in (None, 'Unknown', 'N/A')
            or any(current.get(k) != expected.get(k) for k in ('JobId','UserId','StartTime'))):
        raise ValueError('Allocation identity, owner or running state changed')


def verify_idle_steps(steps, job):
    unexpected = set(steps) - {job+'.0', job+'.extern'}
    if unexpected:
        raise ValueError(f'Additional Slurm steps present; refusing evaluation/cancellation: {sorted(unexpected)}')


def ready_for_evaluation(train, steps, training_step):
    # Wait for all training ranks/processes to exit, including final checkpoint I/O.
    if training_step in steps:
        return False
    if read_json(Path(train)/'timing.json').get('status') != 'complete':
        raise ValueError('Training step exited without successful completion; allocation retained')
    return True


class Slurm:
    def __init__(self, source):
        self.source = Path(source).resolve()

    def describe(self, job):
        text = subprocess.check_output(['scontrol','show','job',job,'-o'], text=True, timeout=30)
        return dict(re.findall(r'(?:^|\s)(JobId|UserId|StartTime|JobState)=(\S+)', text))

    def steps(self, job):
        text = subprocess.check_output(['squeue','--steps','--noheader','--jobs',job,'--format=%i'],
                                       text=True, timeout=30)
        return [line.strip() for line in text.splitlines() if line.strip()]

    def evaluate(self, job, train, output):
        command = ['srun', '--jobid='+job, '--overlap', '--exact', '-N1', '-n1', '-c12',
                   '--gres=gpu:2', '--job-name=test_training_v0_eval', 'bash',
                   str(self.source/'scripts/evaluate_v0_in_allocation.sh'), str(train), str(output)]
        with (output.parent/'evaluation.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    def cancel(self, job):
        subprocess.run(['scancel', job], check=True, timeout=30)


def finish_run(train, output, job, expected, cluster, status):
    verify_finished_training(train, job)
    verify_identity(cluster.describe(job), expected, job)
    verify_idle_steps(cluster.steps(job), job)
    if output.exists():
        raise ValueError('Evaluation output already exists; refusing overwrite or implicit resume')
    status('evaluating', episodes=100, videos=True)
    cluster.evaluate(job, train, output)
    result = verify_evaluation(output, 'v0', train/'latest.pt')
    status('evaluation_verified', episodes=result['total_episodes'], success_rate=result['success_rate'])
    # An evaluation failure, incomplete videos, changed job or new step never reaches scancel.
    verify_identity(cluster.describe(job), expected, job)
    verify_idle_steps(cluster.steps(job), job)
    status('cancelling_allocation', cancellation_target=job)
    cluster.cancel(job)
    status('complete', cancellation_requested=True, cancellation_target=job)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id', required=True)
    p.add_argument('--training-step', required=True)
    p.add_argument('--train-dir', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--poll-seconds', type=int, default=30)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if (not re.fullmatch(r'[0-9]+', a.job_id)
            or not re.fullmatch(re.escape(a.job_id)+r'\.[0-9]+', a.training_step)
            or a.training_step == a.job_id+'.0' or not 5 <= a.poll_seconds <= 60):
        p.error('Require numeric allocation, its training step (not .0), and 5..60s poll interval')
    if os.getenv('SLURM_JOB_ID'):
        p.error('Launch the watcher outside Slurm so it survives allocation cancellation')
    source = Path(__file__).resolve().parents[1]
    train, out = Path(a.train_dir).resolve(), Path(a.output_root).resolve()
    cluster = Slurm(source)
    expected = cluster.describe(a.job_id)
    verify_identity(expected, expected, a.job_id)
    verify_manifest(train, a.job_id)
    steps = cluster.steps(a.job_id)
    verify_idle_steps([s for s in steps if s != a.training_step], a.job_id)
    if a.check_only:
        print(json.dumps(dict(check='passed', job=expected, steps=steps,
            train_dir=str(train), output_root=str(out), would_evaluate_now=ready_for_evaluation(train,steps,a.training_step))))
        return
    out.mkdir(parents=True, exist_ok=True)
    with (out/'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'status.json').exists():
            raise ValueError('Watcher status already exists; use a fresh output root')
        started = time.time()
        state = dict(job_id=a.job_id, training_step=a.training_step, allocation_identity=expected,
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
            source_dir=str(source), train_dir=str(train), output_root=str(out),
            pid=os.getpid(), hostname=socket.gethostname(), started_unix=started)
        def status(stage, **kwargs):
            state.update(stage=stage, updated_unix=time.time(), elapsed_seconds=time.time()-started, **kwargs)
            write_json(out/'status.json', state)
            print(json.dumps(state), flush=True)
        try:
            while True:
                verify_identity(cluster.describe(a.job_id), expected, a.job_id)
                steps = cluster.steps(a.job_id)
                verify_idle_steps([s for s in steps if s != a.training_step], a.job_id)
                if ready_for_evaluation(train, steps, a.training_step):
                    break
                # Trainer rewrites timing.json in place; incomplete JSON is retried next poll.
                try:
                    progress = read_json(train/'timing.json').get('completed_updates')
                except json.JSONDecodeError:
                    progress = None
                status('waiting_for_training', completed_updates=progress, planned_updates=7250)
                time.sleep(a.poll_seconds)
            finish_run(train, out/'inference', a.job_id, expected, cluster, status)
        except BaseException as exc:
            status('failed', error=repr(exc), cancellation_requested=False)
            raise


if __name__ == '__main__':
    main()
