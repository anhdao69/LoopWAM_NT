#!/usr/bin/env python3
"""Run the verified full-suite KV-mix configuration, retaining epochs 5--10."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*Path('src').rglob('*.py'), *Path('scripts').rglob('*.py')])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    from run_kv_campaign import train_command, FULL_CACHE, verify_train
    release = read(a.release)
    if source_hashes() != release['source_hashes']:
        raise ValueError('Source changed since preflight release')
    prepared_path = Path(release['prepared_path'])
    if hashlib.sha256(prepared_path.read_bytes()).hexdigest() != release['prepared_sha256']:
        raise ValueError('Prepared configuration changed')
    prepared = read(prepared_path)
    expected = dict(label='full_mix', version='v0', video=4, action=1,
                    mode='mix', scope='full_libero')
    if prepared['spec'] != expected:
        raise ValueError('Wrong experiment')
    # Includes headroom for replacement files and other currently running trainers.
    if shutil.disk_usage(a.output.parent).free < release['minimum_free_bytes']:
        raise OSError('Insufficient shared storage for retained epochs and checkpoint headroom')
    a.output.mkdir(exist_ok=False)
    write(a.output/'release.json', release)
    command = train_command(expected, prepared['config'], a.output/'train', FULL_CACHE)
    command += ['--retain-epochs-from', '5']
    write(a.output/'command.json', command)
    write(a.output/'status.json', dict(stage='training', started_unix=time.time(), job=os.environ.get('SLURM_JOB_ID')))
    try:
        with (a.output/'train.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        timing = verify_train(a.output/'train', expected)
        import torch
        epochs = []
        for epoch in range(5, 11):
            path = a.output/'train'/f'epoch_{epoch:03d}.pt'
            payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            state = payload['training_state']
            if (payload['step'] != 2170*epoch or state['epoch'] != epoch-1
                or state['windows_seen'] != 277713*epoch
                or state['next_micro'] != math.ceil(277713/(2*prepared['config']['microbatch']))
                or payload.get('action_kv_mode') != 'mix' or len(state['rng']) != 2):
                raise ValueError(f'Invalid epoch {epoch} checkpoint')
            if prepared['config']['backend'] == 'ddp':
                if not payload.get('optimizer', {}).get('state'):
                    raise ValueError('Missing resumable optimizer')
            else:
                native = state['native_optimizer_checkpoint']
                if not (Path(native['directory'])/native['tag']).is_dir():
                    raise ValueError('Missing native optimizer checkpoint')
            epochs.append(dict(epoch=epoch, path=str(path), bytes=path.stat().st_size, step=payload['step']))
            del payload
        write(a.output/'complete.json', dict(timing=timing, retained_epochs=epochs, completed_unix=time.time()))
        write(a.output/'status.json', dict(stage='complete', completed_unix=time.time()))
    except Exception as e:
        write(a.output/'status.json', dict(stage='failed', error=repr(e), time=time.time()))
        raise


if __name__ == '__main__':
    main()
