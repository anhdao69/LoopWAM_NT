#!/usr/bin/env python3
"""Trainer arguments for the RoboCasa-GR1 epochs 6-10 continuation, derived from the parent production manifest.

python gr1cont_args.py OUTPUT_DIR LATENT_DIR [--resume] [--max-updates N]   -> one argument per line (bash mapfile)

Every recipe/layout argument is copied from the five-epoch production manifest (same microbatch, RoPE, text tokens,
manual GC, compile, checkpointing, init/VAE files); only the continuation schedule and the run paths are set here.
"""
import argparse, json, sys
from pathlib import Path

RUN = Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101')
PARENT = RUN / 'train/production/latest.pt'
PARENT_SHA256 = 'e9fa70080709d251ce8e946808ddcfea9df66b007f7389f34d18e4b37a6a1a58'
OVERRIDES = dict(epochs=10, lr_backbone=3e-5, lr_robot_io=3e-5, warmup_fraction=0., save_epochs='6,7,8,9,10')
# parent manifest keys that are not trainer arguments, or are per-launch paths/flags set below
NOT_COPIED = {'contract', 'world_size', 'policy_parameters', 'memory_parameters', 'resumed_from_update', 'git_revision',
              'slurm_job_id', 'cuda_visible_devices', 'output_dir', 'latent_dir', 'resume', 'max_updates', 'no_save',
              'smoke_tail', 'profile_updates', 'allow_partial_cache'}
BOOLEAN_OPTIONAL = {'compile', 'checkpoint_blocks'}           # argparse.BooleanOptionalAction in the trainer


def continuation_args(output_dir, latent_dir, resume=False, max_updates=None):
    parent = json.loads((RUN / 'train/production/manifest.json').read_text())
    assert parent['world_size'] == 4 and parent['contract']['world'] == 4 and parent['git_revision'].startswith('45ee0c07')
    assert (parent['profile_updates'], parent['smoke_tail'], parent['no_save'], parent['allow_partial_cache']) == ('', False, False, False)
    args = []
    for k, v in parent.items():
        if k in NOT_COPIED:
            continue
        v = OVERRIDES.get(k, v); flag = '--' + k.replace('_', '-')
        if k in BOOLEAN_OPTIONAL:
            args.append(flag if v else '--no-' + k.replace('_', '-'))
        else:
            assert not isinstance(v, bool) and v is not None, (k, v)
            args += [flag, str(v)]
    args += ['--output-dir', str(output_dir), '--latent-dir', str(latent_dir),
             '--continuation-from', str(PARENT), '--expected-parent-sha256', PARENT_SHA256]
    if resume:
        args.append('--resume')
    if max_updates is not None:
        args += ['--max-updates', str(int(max_updates))]
    return args


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('output_dir'); p.add_argument('latent_dir')
    p.add_argument('--resume', action='store_true'); p.add_argument('--max-updates', type=int)
    a = p.parse_args()
    sys.stdout.write('\n'.join(continuation_args(a.output_dir, a.latent_dir, a.resume, a.max_updates)) + '\n')
