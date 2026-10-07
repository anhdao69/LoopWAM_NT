#!/usr/bin/env python3
"""Two-H100 fresh v0 video=4/action=1 Long training, followed by 100 video rollouts."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_dense_v2_job import Pipeline as BasePipeline, read_json, write_json

VERSION = 'v0'
PARAMETERS = 584536135
TRAIN_WINDOWS = 92678
UPDATES = 7250
TORCHRUN = ['torchrun', '--standalone', '--nproc_per_node=2']
VAE = 'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'


def source_hashes():
    files = [*Path('src/fastwam').rglob('*.py'), *Path('scripts').glob('*.py'),
             *Path('experiments/libero').rglob('*.py'), *Path('configs').glob('loopwam*.yaml')]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}


def validate_candidate(candidate):
    if (candidate.get('backend') not in ('ddp', 'zero1', 'zero2')
            or type(candidate.get('microbatch')) is not int or candidate['microbatch'] <= 0
            or 128 % (2 * candidate['microbatch'])
            or type(candidate.get('workers')) is not int or candidate['workers'] < 0
            or type(candidate.get('checkpoint_blocks')) is not bool):
        raise ValueError('Invalid two-GPU batch/backend/worker configuration')
    return candidate


def verify_training(directory):
    directory = Path(directory)
    m, t, s = (read_json(directory / (name + '.json')) for name in ('manifest', 'timing', 'trainer_state'))
    data = m.get('data', {})
    if (m.get('version') != VERSION or m.get('resume') is not None or m.get('epochs') != 10
            or m.get('initialization_mode') != 'canonical_wan_artifact_fresh_optimizer'
            or m.get('global_batch') != 128 or m.get('world_size') != 2
            or m.get('microbatch', 0) * 2 * m.get('gradient_accumulation', 0) != 128
            or m.get('loops') != 4 or m.get('action_core_loops') != 1 or m.get('policy_parameters') != PARAMETERS
            or m.get('train_windows') != TRAIN_WINDOWS or m.get('planned_updates') != UPDATES
            or m.get('planned_windows') != 10 * TRAIN_WINDOWS
            or len(data.get('train_episodes', [])) != 344
            or len(data.get('validation_episodes', [])) != 44
            or m.get('policy_dtype') != 'float32' or m.get('optimizer_state_dtype') != 'float32'):
        raise ValueError('Training differs from fresh two-GPU v0 video=4/action=1 Long contract')
    if (t.get('status') != 'complete' or t.get('completed_updates') != UPDATES
            or t.get('windows_seen') != 10 * TRAIN_WINDOWS or s.get('update') != UPDATES
            or s.get('windows_seen') != 10 * TRAIN_WINDOWS or s.get('epoch') != 9
            or s.get('next_micro') != math.ceil(TRAIN_WINDOWS / (2 * m['microbatch']))):
        raise ValueError('Training incomplete; refusing evaluation')
    if not (directory / 'latest.pt').is_file() or (directory / 'latest.pt').stat().st_size == 0:
        raise ValueError('Final checkpoint absent')
    return m


def verify_evaluation(directory, checkpoint, smoke=False):
    directory = Path(directory)
    m, s = read_json(directory / 'manifest.json'), read_json(directory / 'summary.json')
    mode, tasks, episodes = ('smoke', range(2), 1) if smoke else ('final_rollout', range(10), 10)
    if (m.get('mode') != mode or m.get('version') != VERSION or m.get('loops') != 4 or m.get('action_loops') != 1
            or m.get('suite') != 'libero_10' or m.get('world_size') != 2
            or Path(m.get('checkpoint', '')).resolve() != Path(checkpoint).resolve()
            or s.get('checkpoint_sha256') != m.get('checkpoint_sha256')
            or s.get('checkpoint_step') != (10 if smoke else UPDATES)
            or s.get('version') != VERSION or s.get('mode') != mode
            or s.get('suite') != 'libero_10'):
        raise ValueError('Evaluation used the wrong checkpoint/protocol')
    expected = {(task, episode) for task in tasks for episode in range(episodes)}
    rows = s.get('episodes', [])
    actual = [(r['task_id'], r['episode_index']) for r in rows]
    if (s.get('total_episodes') != len(expected) or len(actual) != len(expected)
            or set(actual) != expected or set(s.get('per_task', {})) != set(map(str, tasks))
            or any(t.get('episodes') != episodes for t in s['per_task'].values())):
        raise ValueError('Duplicate or missing evaluation episodes')
    for row in rows:
        if (row.get('checkpoint_sha256') != m['checkpoint_sha256'] or row.get('mode') != mode
                or row.get('suite') != 'libero_10'):
            raise ValueError('Episode checkpoint/protocol mismatch')
        video = Path(row.get('video', ''))
        if not video.is_file() or video.stat().st_size == 0:
            raise ValueError('Evaluation video missing')
    return s


def training_command(candidate, output, cache, smoke_updates=None):
    validate_candidate(candidate)
    cmd = TORCHRUN + ['scripts/train_loopwam.py', '--version', VERSION, '--action-loops', '1',
        '--backend', candidate['backend'], '--microbatch', str(candidate['microbatch']),
        '--global-batch', '128', '--epochs', '10', '--workers', str(candidate['workers']), '--seed', '42',
        '--dataset-scope', 'long_split',
        '--dataset-dir', 'data/lerobot_v30/libero_10_no_noops_lerobot', '--fused-optimizer', '--bucket-views', '--structured-attention', '--smoke',
        '--output-dir', str(output), '--latent-cache-dir', str(cache), '--save-every', '725']
    if candidate['checkpoint_blocks']:
        cmd += ['--checkpoint-blocks']
    if smoke_updates is not None:
        cmd += ['--max-updates', str(smoke_updates), '--validation-samples', '0']
    return cmd


def evaluation_command(train, output, smoke=False):
    cmd = TORCHRUN + ['scripts/evaluate_loopwam_libero.py', '--checkpoint', str(train / 'latest.pt'),
        '--vae-path', VAE, '--stats', str(train / 'data/dataset_stats.json'), '--output-dir', str(output),
        '--suite', 'libero_10', '--episodes-per-task', '1' if smoke else '10']
    if smoke:
        cmd += ['--smoke', '--task-ids', '0,1', '--max-steps', '100']
    return cmd


def timing_estimate(cold, warm):
    cold_s, warm_s = (t['measured_mean_update_seconds'] for t in (cold, warm))
    save_seconds = sum(t['checkpoint_seconds_total'] / t['checkpoint_count'] for t in (cold, warm)) / 2
    if any(not math.isfinite(n) or n <= 0 for n in (cold_s, warm_s, save_seconds)):
        raise ValueError('Native timing/checkpoint measurements must be finite and positive')
    updates_seconds = 725 * (cold_s + 9 * warm_s)
    return dict(cold_seconds=cold_s, warm_seconds=warm_s, checkpoint_seconds_per_save=save_seconds,
                estimated_training_hours=(updates_seconds + 10 * save_seconds) / 3600,
                estimated_checkpoint_seconds=10 * save_seconds,
                note='725*(cold+9*warm) plus ten measured checkpoint saves; excludes setup, validation and final rollouts.')


class Pipeline(BasePipeline):
    def prepare(self, candidate, production_cache):
        validate_candidate(candidate)
        if (self.out / 'prepared.json').exists():
            raise ValueError('Preflight already exists; use --phase run')
        initial = source_hashes()
        cold, warm = self.out / 'cold_train', self.out / 'warm_train'
        cache = self.out / 'preflight_latents'
        if any(p.exists() for p in (cold, warm, cache)):
            raise ValueError('Preflight outputs/cache exist; cold preflight needs a fresh output root')
        self.run('cold_train', training_command(candidate, cold, cache, 10))
        self.run('warm_train', training_command(candidate, warm, cache, 10))
        for directory in (cold, warm):
            t = read_json(directory / 'timing.json')
            if t.get('status') != 'smoke_complete' or t.get('completed_updates') != 10:
                raise ValueError('Native ten-update preflight incomplete')
        inference = self.out / 'smoke_evaluation'
        self.run('smoke_evaluation', evaluation_command(warm, inference, smoke=True))
        verify_evaluation(inference, warm / 'latest.pt', smoke=True)
        estimate = timing_estimate(read_json(cold / 'timing.json'), read_json(warm / 'timing.json'))
        if source_hashes() != initial:
            raise ValueError('Source changed during preflight')
        write_json(self.out / 'prepared.json', dict(selected=candidate, production_cache=str(Path(production_cache).resolve()), source_hashes=initial, **estimate))
        self.status('prepared', **estimate)

    def production(self):
        prepared = read_json(self.out / 'prepared.json')
        if source_hashes() != prepared['source_hashes']:
            raise ValueError('Source changed after preflight; refuse unverified production')
        train, evaluation, cache = (self.out / name for name in ('train', 'evaluation', 'production_latents'))
        if any(p.exists() for p in (train, evaluation)):
            raise ValueError('Production outputs exist; refusing restart or implicit resume')
        cache = Path(prepared['production_cache'])
        self.run('training', training_command(prepared['selected'], train, cache))
        verify_training(train)
        if source_hashes() != prepared['source_hashes']:
            raise ValueError('Source changed during production; refusing evaluation')
        self.run('evaluation', evaluation_command(train, evaluation))
        result = verify_evaluation(evaluation, train / 'latest.pt')
        if source_hashes() != prepared['source_hashes']:
            raise ValueError('Source changed during evaluation; refusing completion')
        self.status('complete', success_rate=result['success_rate'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', required=True)
    p.add_argument('--production-cache', required=True)
    p.add_argument('--phase', choices=['prepare', 'run', 'all'], required=True)
    p.add_argument('--backend', choices=['ddp', 'zero1', 'zero2'], default='zero1')
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--checkpoint-blocks', action='store_true')
    args = p.parse_args()
    pipeline = Pipeline(args.output_root)
    candidate = {key: getattr(args, key) for key in ('backend', 'microbatch', 'workers', 'checkpoint_blocks')}
    try:
        if args.phase in ('prepare', 'all'):
            pipeline.prepare(candidate, args.production_cache)
        if args.phase in ('run', 'all'):
            pipeline.production()
    except BaseException as exc:
        pipeline.status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
