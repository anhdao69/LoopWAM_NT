#!/usr/bin/env python3
"""Idempotent, retrying, verified upload of one ChronoLoop run to the Hugging Face Hub.

Layout: <repo>/<experiment folder>/epoch_08|epoch_09|epoch_10/{policy.pt,config.json}
plus run-level manifest.json, metrics.jsonl, timing.json. A file counts as uploaded only
when the Hub reports the same size and (for LFS files) the same SHA-256 as the local copy.
The token is read from a file and never printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(16 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def remote_index(api, repo, folder):
    out = {}
    try:
        for f in api.list_repo_tree(repo, path_in_repo=folder, recursive=True):
            if hasattr(f, 'size'):
                lfs = getattr(f, 'lfs', None)
                out[f.path] = dict(size=f.size, sha256=(lfs.sha256 if lfs is not None else None))
    except Exception as exc:  # folder absent
        if 'not found' not in str(exc).lower() and '404' not in str(exc):
            raise
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--folder', required=True)
    p.add_argument('--repo', default='anhdao69/ChronoLoop')
    p.add_argument('--token-file', default=os.environ.get('HF_TOKEN_FILE', '/groups/yshang/an221229/cache/huggingface/token'))
    p.add_argument('--epochs', default='8,9,10')
    p.add_argument('--retries', type=int, default=6)
    p.add_argument('--require-all', action='store_true', help='exit 2 unless every requested epoch is verified')
    a = p.parse_args()
    run = Path(a.run_dir)
    api = HfApi(token=Path(a.token_file).read_text().strip())
    manifest = json.loads((run / 'manifest.json').read_text()) if (run / 'manifest.json').exists() else {}
    plan = []
    for e in [int(x) for x in a.epochs.split(',')]:
        ckpt = run / 'checkpoints' / f'epoch_{e:02d}.pt'
        if ckpt.exists():
            cfg = run / 'checkpoints' / f'epoch_{e:02d}.config.json'
            if not cfg.exists():
                cfg.write_text(json.dumps(dict(experiment=manifest.get('run_name'), epoch=e, hf_folder=a.folder,
                    chrono=manifest.get('contract', {}).get('chrono'), parent_sha256=manifest.get('parent_sha256'),
                    schedule_sha256=manifest.get('contract', {}).get('schedule_sha256'),
                    epoch_complete_update=manifest.get('schedule_summary', {}).get('epoch_complete_update', {}).get(str(e)),
                    format='chronoloop-v1 (weights only; load with fastwam.models.wan22.chronoloop.create_chronoloop)'),
                    indent=2))
            plan += [(ckpt, f'{a.folder}/epoch_{e:02d}/policy.pt'), (cfg, f'{a.folder}/epoch_{e:02d}/config.json')]
    for name in ('manifest.json', 'metrics.jsonl', 'timing.json'):
        if (run / name).exists():
            plan.append((run / name, f'{a.folder}/{name}'))
    local = {dst: dict(size=src.stat().st_size, sha256=sha256(src)) for src, dst in plan}
    verified = {}
    for attempt in range(a.retries):
        remote = remote_index(api, a.repo, a.folder)
        # Small text files change while training: re-upload when the size differs.
        todo = [(s, d) for s, d in plan if d not in remote or remote[d]['size'] != local[d]['size']
                or (remote[d]['sha256'] is not None and remote[d]['sha256'] != local[d]['sha256'])]
        if not todo:
            verified = {d: dict(local[d], remote=remote[d]) for _, d in plan}
            break
        try:
            big = [(s, d) for s, d in todo if local[d]['size'] > 50 << 20]
            small = [(s, d) for s, d in todo if local[d]['size'] <= 50 << 20]
            for s, d in big:      # one commit per large file: partial progress survives failures
                api.upload_file(path_or_fileobj=str(s), path_in_repo=d, repo_id=a.repo,
                                commit_message=f'Upload {d}')
            if small:
                api.create_commit(a.repo, [CommitOperationAdd(path_in_repo=d, path_or_fileobj=str(s)) for s, d in small],
                                  commit_message=f'Update {a.folder} metadata')
        except Exception as exc:
            wait = min(600, 30 * 2 ** attempt)
            print(json.dumps(dict(event='upload_retry', attempt=attempt + 1, error=type(exc).__name__,
                                  message=str(exc)[:300], wait_seconds=wait)), flush=True)
            time.sleep(wait)
    epochs_ok = sorted({d.split('/')[1] for d in verified if d.endswith('policy.pt')})
    record = dict(repo=a.repo, folder=a.folder, url=f'https://huggingface.co/{a.repo}/tree/main/{a.folder}',
                  verified=bool(verified), epochs_verified=epochs_ok, files=verified, time=time.time())
    (run / 'hf_upload.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(dict(event='upload_done', verified=bool(verified), epochs=epochs_ok, url=record['url'])), flush=True)
    wanted = {f'epoch_{int(x):02d}' for x in a.epochs.split(',')}
    if not verified or (a.require_all and set(epochs_ok) != wanted):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
