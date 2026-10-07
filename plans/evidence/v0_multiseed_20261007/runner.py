#!/usr/bin/env python3
"""Evaluate one completed v0 checkpoint on two GPUs for three distinct seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path.cwd()))
from scripts.run_dense_v2_job import Pipeline, read_json, verify_evaluation, write_json
from scripts.run_dense_s30_job import source_hashes


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    out = Path(args.output_root).resolve()
    if (out / 'pipeline_status.json').exists():
        raise ValueError('Use a fresh output root; previous results cannot be overwritten')
    pipeline = Pipeline(out)
    checkpoint = Path(args.checkpoint).resolve()
    results = []
    seeds = [42, 43, 44]
    try:
        train = checkpoint.parent
        manifest, timing = read_json(train / 'manifest.json'), read_json(train / 'timing.json')
        if (manifest.get('version') != 'v0' or manifest.get('seed') != 42
                or timing.get('status') != 'complete' or timing.get('completed_updates') != 7250
                or timing.get('windows_seen') != 926780):
            raise ValueError('Expected the completed original LIBERO-Long v0 checkpoint')
        checkpoint_hash, initial_source = sha256(checkpoint), source_hashes()
        write_json(out / 'request.json', dict(checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_hash, evaluation_seeds=seeds,
            training_seed=42, episodes_per_seed=100, world_size=2,
            source_hashes=initial_source,
            initial_states='Same ten suite initial states per task for every seed'))
        for seed in seeds:
            if sha256(checkpoint) != checkpoint_hash or source_hashes() != initial_source:
                raise ValueError('Checkpoint or evaluation source changed')
            directory = out / f'seed_{seed}'
            command = ['torchrun', '--standalone', '--nproc_per_node=2',
                'scripts/evaluate_loopwam_libero.py', '--checkpoint', str(checkpoint),
                '--vae-path', 'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',
                '--stats', str(train / 'data/dataset_stats.json'),
                '--output-dir', str(directory), '--suite', 'libero_10',
                '--episodes-per-task', '10', '--seed', str(seed)]
            pipeline.run(f'evaluation_seed_{seed}', command)
            summary = verify_evaluation(directory, 'v0', checkpoint)
            evaluation = read_json(directory / 'manifest.json')
            if (evaluation.get('world_size') != 2 or evaluation.get('suite') != 'libero_10'
                    or evaluation.get('arguments', {}).get('seed') != seed
                    or evaluation.get('checkpoint_sha256') != checkpoint_hash):
                raise ValueError('Evaluation seed, GPU count, suite or checkpoint mismatch')
            for row in summary['episodes']:
                if (row.get('suite') != 'libero_10'
                        or row.get('initial_state_index') != row['episode_index']
                        or row.get('seed') != seed + row['task_id'] * 100000 + row['episode_index'] * 1000):
                    raise ValueError('Episode seed or initial state mismatch')
            results.append(dict(seed=seed, success_rate=summary['success_rate'],
                                per_task=summary['per_task'], output=str(directory)))
            write_json(out / 'results.json', dict(status='running', results=results))
        if sha256(checkpoint) != checkpoint_hash or source_hashes() != initial_source:
            raise ValueError('Checkpoint or source changed during evaluation')
        rates = [r['success_rate'] for r in results]
        write_json(out / 'results.json', dict(status='complete', checkpoint_sha256=checkpoint_hash,
            episodes=300, results=results, mean_success_rate=statistics.mean(rates),
            sample_std_success_rate=statistics.stdev(rates),
            note='Evaluation-seed variation for one fixed training checkpoint'))
        pipeline.status('complete', results=results, mean_success_rate=statistics.mean(rates))
    except BaseException as exc:
        pipeline.status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
