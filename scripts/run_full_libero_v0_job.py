#!/usr/bin/env python3
"""Preflight, fresh ten-epoch v0 training, then 400 four-suite LIBERO rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
TORCHRUN = ['torchrun', '--standalone', '--nproc_per_node=4']
VAE = 'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
EXPECTED_WINDOWS = 277713


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def source_hashes():
    paths = [*Path('src/fastwam').rglob('*.py'), *Path('scripts').glob('*.py'),
             *Path('experiments/libero').rglob('*.py')]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def training_command(backend, microbatch, workers, output, cache, smoke_updates=None):
    command = TORCHRUN + ['scripts/train_loopwam.py', '--version', 'v0', '--backend', backend,
        '--microbatch', str(microbatch), '--global-batch', '128', '--epochs', '10',
        '--workers', str(workers), '--seed', '42', '--fused-optimizer', '--bucket-views',
        '--structured-attention', '--smoke', '--dataset-scope', 'full_libero',
        '--dataset-dir', 'data/lerobot_v30', '--validation-samples', '0',
        '--output-dir', str(output), '--latent-cache-dir', str(cache), '--save-every', '2170']
    if smoke_updates is not None:
        command += ['--max-updates', str(smoke_updates)]
    return command


def evaluation_command(train, output, suite, smoke=False):
    if suite not in SUITES:
        raise ValueError('Unknown LIBERO suite')
    command = TORCHRUN + ['scripts/evaluate_loopwam_libero.py', '--suite', suite,
        '--checkpoint', str(train/'latest.pt'), '--vae-path', VAE,
        '--stats', str(train/'data/dataset_stats.json'), '--output-dir', str(output),
        '--episodes-per-task', '1' if smoke else '10']
    if smoke:
        command += ['--smoke', '--task-ids', '0,1,2,3', '--max-steps', '30']
    return command


def verify_training(directory):
    directory = Path(directory)
    m, t, s = (read_json(directory/(name+'.json')) for name in ('manifest','timing','trainer_state'))
    if (m['version'] != 'v0' or m['resume'] is not None or m['max_updates'] is not None
            or m['epochs'] != 10 or m['global_batch'] != 128 or m['world_size'] != 4
            or m['microbatch']*4*m['gradient_accumulation'] != 128 or m['loops'] != 4
            or m['policy_parameters'] != 584536135 or m['train_windows'] != EXPECTED_WINDOWS
            or m['val_windows'] != 0 or m['validation_samples'] != 0
            or m['planned_updates'] != 21700 or m['planned_windows'] != 2777130
            or m['initialization_mode'] != 'canonical_wan_artifact_fresh_optimizer'
            or m['dataset_scope'] != 'full_libero' or m['data']['dataset_scope'] != 'full_libero'
            or m['data']['split'] != 'all_train' or m['data']['suites'] != list(SUITES)
            or m['data']['train_windows'] != EXPECTED_WINDOWS):
        raise ValueError('Training differs from fresh all-four-suite ten-epoch contract')
    if (t['status'] != 'complete' or t['completed_updates'] != 21700 or t['windows_seen'] != 2777130
            or s['update'] != 21700 or s['windows_seen'] != 2777130 or s['epoch'] != 9
            or s['next_micro'] != math.ceil(EXPECTED_WINDOWS/(4*m['microbatch']))):
        raise ValueError('Training incomplete; refusing final evaluation')
    if not (directory/'latest.pt').is_file() or (directory/'latest.pt').stat().st_size == 0:
        raise ValueError('Final checkpoint missing')
    return m


def verify_evaluation(directory, suite, checkpoint):
    directory = Path(directory)
    m, s = read_json(directory/'manifest.json'), read_json(directory/'summary.json')
    if (m['mode'] != 'final_rollout' or m['version'] != 'v0' or m['loops'] != 4
            or m['suite'] != suite or Path(m['checkpoint']).resolve() != checkpoint.resolve()
            or s['mode'] != 'final_rollout' or s['suite'] != suite or s['version'] != 'v0'
            or s['checkpoint_step'] != 21700 or s['checkpoint_sha256'] != m['checkpoint_sha256']
            or s['total_episodes'] != 100 or set(s['per_task']) != set(map(str,range(10)))
            or any(task['episodes'] != 10 for task in s['per_task'].values())):
        raise ValueError('Evaluation checkpoint, suite or protocol mismatch')
    rows = s['episodes']
    expected = {(task,episode) for task in range(10) for episode in range(10)}
    if len(rows) != 100 or {(r['task_id'],r['episode_index']) for r in rows} != expected:
        raise ValueError('Missing or duplicate evaluation episodes')
    for row in rows:
        if (row['suite'] != suite or row['mode'] != 'final_rollout'
                or row['checkpoint_sha256'] != m['checkpoint_sha256']
                or not Path(row['video']).is_file() or Path(row['video']).stat().st_size == 0):
            raise ValueError('Episode evidence missing or inconsistent')
    if s['successes'] != sum(bool(r['success']) for r in rows):
        raise ValueError('Evaluation success accounting mismatch')
    return s


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output_root).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.durations = {}
        self.state = dict(slurm_job_id=os.getenv('SLURM_JOB_ID'), slurm_step_id=os.getenv('SLURM_STEP_ID'),
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            started_unix=self.started)

    def status(self, stage, **kwargs):
        self.state.update(stage=stage, elapsed_seconds=time.time()-self.started,
                          stage_durations_seconds=self.durations, **kwargs)
        write_json(self.out/'pipeline_status.json',self.state)
        print(json.dumps(self.state),flush=True)

    def run(self, stage, command):
        self.status(stage,command=command)
        started = time.time()
        with (self.out/(stage+'.log')).open('x') as stream:
            result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        self.durations[stage] = time.time()-started
        if result.returncode:
            raise RuntimeError(f'{stage} failed ({result.returncode}); see {stage}.log')
        self.status(stage+'_passed')

    def prepare(self):
        if (self.out/'prepared.json').exists():
            raise ValueError('Already prepared; use --phase run')
        hashes = source_hashes()
        cache = self.out/'latents'
        cold, warm = self.out/'smoke_cold', self.out/'smoke_warm'
        for name, destination in [('cold',cold),('warm',warm)]:
            self.run('smoke_'+name,training_command(self.args.backend,self.args.microbatch,
                self.args.workers,destination,cache,10))
            m, t = read_json(destination/'manifest.json'), read_json(destination/'timing.json')
            if (m['train_windows'] != EXPECTED_WINDOWS or m['val_windows'] != 0
                    or m['planned_updates'] != 21700 or m['resume'] is not None
                    or t['status'] != 'smoke_complete' or t['completed_updates'] != 10
                    or t['windows_seen'] != 1280):
                raise ValueError('Native training preflight data/update mismatch')
            data = m['data']
            if (data['available_episodes'] != 1712 or len(data['task_counts']) != 40
                    or data['suites'] != list(SUITES) or data['split'] != 'all_train'
                    or any(len(data['coverage'][suite]['task_counts']) != 10 for suite in SUITES)):
                raise ValueError('Full source coverage audit failed')
        for suite in SUITES:
            output = self.out/('smoke_eval_'+suite)
            self.run('smoke_eval_'+suite,evaluation_command(warm,output,suite,smoke=True))
            result = read_json(output/'summary.json')
            if (result['total_episodes'] != 4 or result['mode'] != 'smoke'
                    or result['suite'] != suite or result['version'] != 'v0'):
                raise ValueError('Four-GPU suite inference smoke incomplete')
        cold_t,warm_t = read_json(cold/'timing.json'),read_json(warm/'timing.json')
        cold_s,warm_s = cold_t['measured_mean_update_seconds'],warm_t['measured_mean_update_seconds']
        if not all(math.isfinite(s) and s>0 for s in (cold_s,warm_s)):
            raise ValueError('Invalid timing measurement')
        if hashes != source_hashes():
            raise ValueError('Source changed during preflight')
        prepared = dict(source_hashes=hashes,backend=self.args.backend,microbatch=self.args.microbatch,
            workers=self.args.workers,cold_seconds=cold_s,warm_seconds=warm_s,
            estimated_training_hours=2170*(cold_s+9*warm_s)/3600,
            estimate_note='One cold VAE epoch plus nine warm epochs; excludes saves, setup and simulator evaluation.',
            prepared_unix=time.time(),stage_durations_seconds=self.durations)
        write_json(self.out/'prepared.json',prepared)
        self.status('prepared',estimated_training_hours=prepared['estimated_training_hours'])

    def production(self):
        prepared = read_json(self.out/'prepared.json')
        if source_hashes() != prepared['source_hashes']:
            raise ValueError('Source changed after preflight')
        train = self.out/'train'
        if train.exists() or any((self.out/('evaluation_'+suite)).exists() for suite in SUITES):
            raise ValueError('Production output exists; refusing implicit resume or overwrite')
        self.run('training',training_command(prepared['backend'],prepared['microbatch'],
            prepared['workers'],train,self.out/'latents'))
        if source_hashes() != prepared['source_hashes']:
            raise ValueError('Source changed during training')
        verify_training(train)
        results = {}
        for suite in SUITES:
            if source_hashes() != prepared['source_hashes']:
                raise ValueError('Source changed before evaluation')
            output = self.out/('evaluation_'+suite)
            self.run('evaluation_'+suite,evaluation_command(train,output,suite))
            if source_hashes() != prepared['source_hashes']:
                raise ValueError('Source changed during evaluation')
            results[suite] = verify_evaluation(output,suite,train/'latest.pt')
        checkpoints = {r['checkpoint_sha256'] for r in results.values()}
        if len(checkpoints) != 1:
            raise ValueError('Suites evaluated different checkpoints')
        successes = sum(r['successes'] for r in results.values())
        summary = dict(total_episodes=400,successes=successes,success_rate=successes/400,
            checkpoint_sha256=checkpoints.pop(),checkpoint_step=21700,per_suite=results,
            training=read_json(train/'timing.json'),stage_durations_seconds=self.durations,
            elapsed_seconds=time.time()-self.started)
        write_json(self.out/'summary.json',summary)
        self.status('complete',successes=successes,total_episodes=400,success_rate=successes/400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--phase',choices=['prepare','run','all'],default='all')
    parser.add_argument('--backend',choices=['ddp','zero1','zero2'],default='ddp')
    parser.add_argument('--microbatch',type=int,default=8)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    if args.microbatch<1 or 128%(4*args.microbatch) or args.workers<0:
        parser.error('Invalid four-GPU microbatch or worker count')
    pipeline = Pipeline(args)
    try:
        if args.phase in ('prepare','all'):
            pipeline.prepare()
        if args.phase in ('run','all'):
            pipeline.production()
    except BaseException as exc:
        pipeline.status('failed',error=repr(exc))
        raise


if __name__ == '__main__':
    main()
