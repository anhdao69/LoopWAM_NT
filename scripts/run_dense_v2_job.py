#!/usr/bin/env python3
"""Four-H100 preflight, then fresh Dense-S12 -> rollouts -> fresh v2 -> rollouts."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import statistics
import time

VERSIONS = ('dense_s12', 'v2')
TORCHRUN = ['torchrun', '--standalone', '--nproc_per_node=4']
VAE = 'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def source_hashes():
    files = [*Path('src/fastwam').rglob('*.py'), *Path('scripts').glob('*loopwam*.py'),
             Path(__file__), *Path('configs').glob('loopwam*.yaml')]
    return {str(p.resolve().relative_to(Path.cwd())): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(files))}


def verify_benchmark_sources(result, current):
    required = {'src/fastwam/models/wan22/loopwam.py', 'src/fastwam/models/wan22/loop_mot.py',
        'src/fastwam/models/wan22/loopwam_init.py', 'src/fastwam/models/wan22/fastwam.py',
        'src/fastwam/training_backends.py'}
    recorded = result.get('source_sha256', {})
    if not required.issubset(recorded):
        raise ValueError('Benchmark source evidence incomplete')
    for name in required:
        if recorded[name] != current.get(name):
            raise ValueError(f'Benchmark source changed: {name}')


def select_candidates(results, version):
    if version not in VERSIONS or not results:
        raise ValueError('No valid architecture/benchmark candidates')
    for r in results:
        if (r.get('version') != version or r.get('backend') not in ('ddp', 'zero1', 'zero2')
                or r.get('world_size') != 4 or r.get('global_batch') != 128
                or r.get('loops') != (1 if version == 'dense_s12' else 4)
                or r['microbatch'] * 4 * r['accumulation'] != 128):
            raise ValueError('Benchmark architecture or global batch differs')
        if not r.get('fused') or not r.get('structured_attention'):
            raise ValueError('Benchmark did not use the intended speed configuration')
        precision = r.get('precision', {})
        if precision.get('policy_dtype') != 'float32' or precision.get('optimizer_moment_dtypes') != ['torch.float32']:
            raise ValueError('Benchmark changed policy or optimizer precision')
        if not math.isfinite(r['steady_seconds']) or r['steady_seconds'] <= 0 or len(r['records']) < 5:
            raise ValueError('Insufficient measured benchmark updates')
        if any(not math.isfinite(row[k]) for row in r['records'] for k in ('loss', 'grad_norm', 'seconds')):
            raise ValueError('Nonfinite benchmark')
    groups = {}
    for result in results:
        key = (result['backend'], result['microbatch'], result.get('workers',4))
        groups.setdefault(key, []).append(result)
    combined = []
    for trials in groups.values():
        seconds = statistics.mean(row['seconds'] for r in trials for row in r['records'][2:])
        representative = min(trials, key=lambda r: abs(r['steady_seconds']-seconds))
        combined.append({**representative, 'steady_seconds': seconds,
                         'trial_steady_seconds': [r['steady_seconds'] for r in trials]})
    return sorted(combined, key=lambda r: r['steady_seconds'])


def verify_training(directory, version):
    directory = Path(directory)
    m, t, s = (read_json(directory / (name + '.json')) for name in ('manifest', 'timing', 'trainer_state'))
    if (m['version'] != version or m['resume'] is not None or m['epochs'] != 10
            or m['global_batch'] != 128 or m['world_size'] != 4
            or m['microbatch'] * m['world_size'] * m['gradient_accumulation'] != 128
            or m['loops'] != (1 if version == 'dense_s12' else 4)
            or m['policy_parameters'] != 584536135 or m['train_windows'] != 92678
            or m['planned_updates'] != 7250 or m['planned_windows'] != 926780):
        raise ValueError('Training differs from fresh four-GPU architecture/data contract')
    if (t['status'] != 'complete' or t['completed_updates'] != 7250 or t['windows_seen'] != 926780
            or s['update'] != 7250 or s['windows_seen'] != 926780 or s['epoch'] != 9):
        raise ValueError('Training incomplete; refusing evaluation/next model')
    if not (directory / 'latest.pt').is_file() or (directory / 'latest.pt').stat().st_size == 0:
        raise ValueError('Final checkpoint absent')
    return m


def verify_evaluation(directory, version, checkpoint):
    directory = Path(directory)
    m, s = read_json(directory / 'manifest.json'), read_json(directory / 'summary.json')
    if (m['mode'] != 'final_rollout' or m['version'] != version
            or m['loops'] != (1 if version == 'dense_s12' else 4)
            or Path(m['checkpoint']).resolve() != Path(checkpoint).resolve()):
        raise ValueError('Evaluation used the wrong checkpoint/protocol')
    if (s['mode'] != 'final_rollout' or s['version'] != version or s['checkpoint_step'] != 7250
            or s['checkpoint_sha256'] != m['checkpoint_sha256'] or s['total_episodes'] != 100
            or set(s['per_task']) != set(map(str, range(10)))
            or any(t['episodes'] != 10 for t in s['per_task'].values())):
        raise ValueError('Final 100-episode evaluation incomplete')
    rows = s['episodes']
    expected = {(task, episode) for task in range(10) for episode in range(10)}
    actual = [(r['task_id'], r['episode_index']) for r in rows]
    if len(actual) != 100 or set(actual) != expected:
        raise ValueError('Duplicate or missing evaluation episodes')
    for row in rows:
        if row['checkpoint_sha256'] != m['checkpoint_sha256'] or row['mode'] != 'final_rollout':
            raise ValueError('Episode checkpoint/protocol mismatch')
        video = Path(row['video'])
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError('Evaluation video missing')
    return s


def training_command(candidate, output, cache, smoke_updates=None):
    cmd = TORCHRUN + ['scripts/train_loopwam.py', '--version', candidate['version'],
        '--backend', candidate['backend'], '--microbatch', str(candidate['microbatch']),
        '--global-batch', '128', '--epochs', '10', '--workers', str(candidate.get('workers',4)), '--seed', '42',
        '--fused-optimizer', '--bucket-views', '--structured-attention', '--smoke',
        '--output-dir', str(output), '--latent-cache-dir', str(cache), '--save-every', '725']
    if smoke_updates is not None:
        cmd += ['--max-updates', str(smoke_updates), '--validation-samples', '0']
    return cmd


def evaluation_command(train, output, smoke=False):
    cmd = TORCHRUN + ['scripts/evaluate_loopwam_libero.py', '--checkpoint', str(train / 'latest.pt'),
        '--vae-path', VAE, '--stats', str(train / 'data/dataset_stats.json'), '--output-dir', str(output),
        '--episodes-per-task', '1' if smoke else '10']
    if smoke:
        cmd += ['--smoke', '--task-ids', '0,1,2,3', '--max-steps', '100']
    return cmd


class Pipeline:
    def __init__(self, out):
        self.out = Path(out).resolve()
        self.out.mkdir(exist_ok=True, parents=True)
        self.started = time.time()
        self.durations = {}
        self.state = dict(slurm_job_id=os.getenv('SLURM_JOB_ID'),
            slurm_step_id=os.getenv('SLURM_STEP_ID'), started_unix=self.started,
            source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())

    def status(self, stage, **kwargs):
        self.state.update(stage=stage, elapsed_seconds=time.time()-self.started,
                          stage_durations_seconds=self.durations, **kwargs)
        write_json(self.out / 'pipeline_status.json', self.state)
        print(json.dumps(self.state), flush=True)

    def run(self, stage, cmd, allow_oom=False):
        self.status(stage, command=cmd)
        log = self.out / (stage + '.log')
        start = time.time()
        with log.open('x') as stream:
            result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        self.durations[stage] = time.time()-start
        if result.returncode:
            if allow_oom and 'CUDA out of memory' in log.read_text():
                self.status(stage + '_rejected_oom')
                return False
            raise RuntimeError(f'{stage} failed with exit {result.returncode}: {log}')
        self.status(stage + '_passed')
        return True

    def prepare(self):
        if (self.out / 'prepared.json').exists():
            raise ValueError('Preflight already exists; use --phase run')
        initial_hashes = source_hashes()
        selections = {}
        for version in VERSIONS:
            results = [read_json(p) for p in self.out.glob(f'{version}_*_warm/result.json')]
            for result in results:
                verify_benchmark_sources(result, initial_hashes)
            candidates = select_candidates(results, version)
            rejected = []
            for candidate in candidates:
                label = f"{version}_{candidate['backend']}_mb{candidate['microbatch']}_w{candidate.get('workers',4)}"
                cold = self.out / ('smoke_' + label)
                if self.run('smoke_' + label, training_command(candidate, cold, cold/'latents', 10), allow_oom=True):
                    break
                rejected.append(label)
            else:
                raise RuntimeError(f'No candidate fits native cold training for {version}')
            # Measure the actual production loop with a warm cache, including its diagnostics.
            warm = self.out / ('warm_train_' + label)
            self.run('warm_train_' + label, training_command(candidate, warm, cold/'latents', 10))
            inference = self.out / ('smoke_eval_' + version)
            self.run('smoke_eval_' + version, evaluation_command(warm, inference, smoke=True))
            sm = read_json(inference / 'summary.json')
            if sm['total_episodes'] != 4 or sm['version'] != version:
                raise ValueError('Four GPU inference smoke incomplete')
            cold_t, warm_t = read_json(cold/'timing.json'), read_json(warm/'timing.json')
            cold_s, warm_s = cold_t['measured_mean_update_seconds'], warm_t['measured_mean_update_seconds']
            hours = 725 * (cold_s + 9*warm_s)/3600 if version == 'dense_s12' else 7250*warm_s/3600
            selections[version] = dict(selected=candidate, candidates=candidates, cold_oom_rejected=rejected,
                cold_seconds=cold_s, warm_seconds=warm_s, estimated_training_hours=hours,
                smoke_checkpoint=str(warm/'latest.pt'), smoke_evaluation=str(inference),
                note='Dense fills private cache in epoch1; v2 reuses completed dense cache. Excludes saves, validation and rollouts.')
            write_json(self.out / ('selection_' + version + '.json'), selections[version])
        if source_hashes() != initial_hashes:
            raise ValueError('Source changed during preflight; refusing production approval')
        write_json(self.out/'prepared.json', dict(selections=selections, source_hashes=initial_hashes,
            prepared_unix=time.time(), stage_durations_seconds=self.durations))
        self.status('prepared', estimates_hours={v: selections[v]['estimated_training_hours'] for v in VERSIONS})

    def production(self):
        prepared = read_json(self.out/'prepared.json')
        if prepared['source_hashes'] != source_hashes():
            raise ValueError('Source changed after preflight; refuse unverified production')
        if any((self.out/(v+'_train')).exists() or (self.out/(v+'_evaluation')).exists() for v in VERSIONS):
            raise ValueError('Production outputs exist; refusing restart or implicit resume')
        cache = self.out/'production_latents'
        for version in VERSIONS:
            chosen = prepared['selections'][version]['selected']
            train = self.out/(version+'_train')
            self.run(version+'_training', training_command(chosen, train, cache))
            verify_training(train, version)
            evaluation = self.out/(version+'_evaluation')
            self.run(version+'_evaluation', evaluation_command(train, evaluation))
            result = verify_evaluation(evaluation, version, train/'latest.pt')
            self.status(version+'_complete', success_rate=result['success_rate'])
        self.status('complete')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', required=True)
    p.add_argument('--phase', choices=['prepare','run'], required=True)
    a=p.parse_args()
    pipeline=Pipeline(a.output_root)
    try:
        if a.phase=='prepare': pipeline.prepare()
        else: pipeline.production()
    except BaseException as exc:
        pipeline.status('failed', error=repr(exc))
        raise


if __name__=='__main__': main()
