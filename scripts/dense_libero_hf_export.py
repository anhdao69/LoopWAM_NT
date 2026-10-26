#!/usr/bin/env python3
"""Verify a finished four-suite Dense-S12/S30 run against the v0 four-suite release, export it, upload it to HF.

verify  --train RUN/train --reference V0_RELEASE_DIR            completion + fairness checks, writes RUN/fairness.json
publish --train ... --reference ... --stage DIR --token-file F  optimizer-free policy.pt (tensor-checked), metadata,
                                                                upload to libero-all-suites/<dense-sNN>, Hub hash check,
                                                                checkpoint_index.json / README.md rows
The reference is the released libero-all-suites/v0-video4-action4 directory (its data/training manifests).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import shutil
from pathlib import Path

REPO = 'anhdao69/LoopWAM_NT'
PARAMETERS = {'dense_s12': 584536135, 'dense_s30': 1416114247}
REPO_PATH = {'dense_s12': 'libero-all-suites/dense-s12', 'dense_s30': 'libero-all-suites/dense-s30'}
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
WINDOWS, UPDATES = 277713, 21700
# Same training fields as the earlier four-suite fairness checks; paths differ between clusters and are ignored.
TRAINING_FIELDS = ['asset_sha256', 'seed', 'epochs', 'global_batch', 'train_windows', 'planned_updates',
                   'planned_windows', 'policy_dtype', 'optimizer_state_dtype', 'compute_dtype', 'initialization_mode']
PATH_FIELDS = {'dataset_dir', 'text_cache_dir', 'normalization_path'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n'); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def strip_paths(data):
    data = {k: v for k, v in data.items() if k not in PATH_FIELDS}
    if 'coverage' in data:
        data['coverage'] = {s: {k: v for k, v in c.items() if k not in PATH_FIELDS} for s, c in data['coverage'].items()}
    return data


def verify(train, reference):
    train, reference = Path(train), Path(reference)
    m, t, s = (read(train / f'{n}.json') for n in ('manifest', 'timing', 'trainer_state'))
    version = m['version']
    world = m['world_size']
    problems = []
    def need(ok, what):
        if not ok:
            problems.append(what)
    need(version in PARAMETERS, f'version {version}')
    need(m['policy_parameters'] == PARAMETERS.get(version), 'policy parameters')
    need(m['loops'] == 1, 'dense runs one traversal')
    need(m['resume'] is None and m['max_updates'] is None and not m['checkpoint_blocks'], 'fresh full run')
    need(m['backend'] == 'ddp' and m['microbatch'] * world * m['gradient_accumulation'] == 128, 'batch contract')
    need(m['val_windows'] == 0 and m['validation_samples'] == 0, 'no validation split')
    need(m['dataset_scope'] == 'full_libero' and m['data']['split'] == 'all_train' and m['data']['suites'] == SUITES,
         'four-suite scope')
    need(t['status'] == 'complete' and t['completed_updates'] == UPDATES and t['windows_seen'] == 10 * WINDOWS,
         'timing complete')
    need(s['update'] == UPDATES and s['epoch'] == 9 and s['windows_seen'] == t['windows_seen']
         and s['next_micro'] == math.ceil(WINDOWS / (world * m['microbatch'])), 'trainer state at the end of epoch 10')
    need((train / 'latest.pt').stat().st_size > 0, 'final checkpoint')
    ref_train = read(reference / 'training_manifest.json')
    differ = [f for f in TRAINING_FIELDS if m[f] != ref_train[f]]
    need(not differ, f'training fields differ from the v0 release: {differ}')
    ref_data, data = strip_paths(read(reference / 'data_manifest.json')), strip_paths(read(train / 'data/data_manifest.json'))
    data_differ = sorted(k for k in set(ref_data) | set(data) if ref_data.get(k) != data.get(k))
    need(not data_differ, f'data manifest differs from the v0 release: {data_differ}')
    resumes = sorted(p.name for p in train.glob('resume_manifest*.json'))
    report = dict(passed=not problems, problems=problems, version=version, reference=str(reference),
                  compared_training_fields=TRAINING_FIELDS, data_fields=sorted(ref_data),
                  ignored_path_fields=sorted(PATH_FIELDS), world_size=world, microbatch=m['microbatch'],
                  gradient_accumulation=m['gradient_accumulation'], gpu=m['gpu'], git_revision=m['git_revision'],
                  resumed=bool(resumes), resume_manifests=resumes,
                  normalization_sha256=m['data']['normalization_sha256'])
    write(train.parent / 'fairness.json', report)
    print(json.dumps(report), flush=True)
    if problems:
        raise SystemExit('verification failed: ' + '; '.join(problems))
    return report


def readme(info, m, t, fair):
    name = {'dense_s12': 'Dense-S12', 'dense_s30': 'Dense-S30'}[info['version']]
    blocks = {'dense_s12': '12 independent block pairs (the 3 + 6 + 3 donor layers of LoopWAM-S)',
              'dense_s30': 'all 30 native Wan2.1 donor layers as independent block pairs'}[info['version']]
    hours = t['elapsed_training_seconds'] / 3600
    resumed = ('Training resumed from a saved checkpoint with the saved optimizer and RNG state; see the resume '
               'manifests listed in fairness.json.' if fair['resumed'] else 'Training ran in one segment.')
    return f"""# LoopWAM {name} — all four LIBERO suites

Completed ten-epoch full-suite training checkpoint (Spatial, Object, Goal and LIBERO-10; all 1,712 locally available
demonstrations, 277,713 windows), fresh canonical Wan initialization, training seed 42, global batch 128
({fair['world_size']} GPUs x microbatch {fair['microbatch']} x accumulation {fair['gradient_accumulation']}),
21,700 updates. {name} runs {blocks} once ({info['parameters']:,} policy parameters), with final-only flow loss and no XSA.
It is the dense control for `libero-all-suites/v0-video4-action4` and uses the same data, normalization
(`{info['normalization_sha256'][:12]}...`), donor initialization, VAE, schedule and precision; `fairness.json` lists the
checked fields. Training took {hours:.1f} h on {fair['world_size']}x {fair['gpu']}. {resumed}

**Not evaluated:** this is a training-only checkpoint; no simulator evaluation has been run and no success rate is claimed.

`policy.pt` is the native `loopwam-s-v1` checkpoint with the Adam optimizer state removed; every model tensor was
checked for exact equality with the final training checkpoint. Use it for inference or fresh-optimizer fine-tuning,
not exact optimizer-state resume. Keep `dataset_stats.json` matched to it. Load with the LoopWAM code (revision
`{info['source_revision']}` or a compatible later one): `create_loopwam(checkpoint_path=..., vae_path=...)`; the depth
is rebuilt from the checkpoint metadata. Wan VAE, text embeddings and observation preprocessing are external.
"""


def publish(train, reference, stage, token_file, delete_token):
    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from fastwam.models.wan22.loopwam import _validate_checkpoint_depth
    train, stage = Path(train), Path(stage)
    token_file = Path(token_file)
    api = HfApi(token=token_file.read_text().strip())
    try:
        assert api.whoami()['name'] == 'anhdao69'
        fair = verify(train, reference)
        m, t = read(train / 'manifest.json'), read(train / 'timing.json')
        version = m['version']; repo_path = REPO_PATH[version]
        dst = stage / repo_path; dst.mkdir(parents=True, exist_ok=True)
        source, target = train / 'latest.pt', dst / 'policy.pt'
        source_sha = sha(source)
        payload = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
        assert payload['format_version'] == 'loopwam-s-v1' and payload['step'] == UPDATES and payload['version'] == version
        policy = {k: v for k, v in payload.items() if k != 'optimizer'}
        tmp = target.with_suffix('.pt.tmp'); torch.save(policy, tmp); tmp.replace(target)
        restored = torch.load(target, map_location='cpu', mmap=True, weights_only=False)
        _validate_checkpoint_depth(restored)
        for group in ('mot', 'proprio_encoder'):
            assert restored[group].keys() == payload[group].keys()
            for k, v in payload[group].items():
                assert torch.equal(v, restored[group][k]), f'tensor mismatch {group}/{k}'
        assert restored['step'] == payload['step'] and restored['architecture'] == payload['architecture']
        assert 'optimizer' not in restored
        info = dict(id=f'{version}_full_libero_hpg', repo_path=repo_path, training_path=str(train),
                    source_sha256=source_sha, source_bytes=source.stat().st_size, export_sha256=sha(target),
                    export_bytes=target.stat().st_size, format='loopwam-s-v1', optimizer_included=False,
                    step=payload['step'], version=version, video_loops=1, action_loops=1,
                    parameters=m['policy_parameters'], dataset_scope='full_libero', epochs=10, global_batch=128,
                    training_seed=42, evaluation_status='not_run_training_only', success_rate=None, successes=None,
                    episodes=0, training_seconds=t['elapsed_training_seconds'],
                    initialization_mode=m['initialization_mode'], tensor_equality_verified=True,
                    source_revision=m['git_revision'], normalization_sha256=m['data']['normalization_sha256'],
                    fairness_check_passed=fair['passed'], gpu=m['gpu'], world_size=m['world_size'],
                    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        del restored, policy, payload
        shutil.copy2(train / 'data/dataset_stats.json', dst / 'dataset_stats.json')
        assert sha(dst / 'dataset_stats.json') == info['normalization_sha256']
        for src, name in ((train / 'manifest.json', 'training_manifest.json'),
                          (train / 'data/data_manifest.json', 'data_manifest.json'),
                          (train / 'timing.json', 'training_timing.json'), (train.parent / 'fairness.json', 'fairness.json')):
            shutil.copy2(src, dst / name)
        write(dst / 'export.json', info)
        (dst / 'README.md').write_text(readme(info, m, t, fair))
        expected = {f'{repo_path}/{p.name}': dict(size=p.stat().st_size, sha256=sha(p)) for p in dst.iterdir() if p.is_file()}
        print(json.dumps(dict(event='uploading', files=len(expected), bytes=sum(x['size'] for x in expected.values()))), flush=True)
        api.upload_folder(repo_id=REPO, repo_type='model', folder_path=str(dst), path_in_repo=repo_path,
                          commit_message=f'Add {repo_path} (four-suite {version}, training only)')
        # Index and model card: add or replace this checkpoint's row.
        meta = stage / 'root'; meta.mkdir(exist_ok=True)
        index = read(hf_hub_download(REPO, 'checkpoint_index.json', token=api.token, local_dir=meta, force_download=True))
        index['checkpoints'] = [c for c in index['checkpoints'] if c['repo_path'] != repo_path] + [info]
        index['updated_utc'] = info['created_utc']
        write(meta / 'checkpoint_index.json', index)
        card = Path(hf_hub_download(REPO, 'README.md', token=api.token, local_dir=meta, force_download=True)).read_text()
        row = (f"| [{repo_path}]({repo_path}) | {version} | 1/1 | {info['parameters']:,} | "
               f"Training only; evaluation not run |")
        head = '## October 2026 additions: full-suite dense controls'
        if head not in card:
            section = (f"{head}\n\n| Directory | Model | Video/action loops | Parameters | Evaluation |\n"
                       f"|---|---|---:|---:|---|\n\nDense-S12 / Dense-S30 trained on all four suites with the same data, "
                       f"normalization, initialization and schedule as `libero-all-suites/v0-video4-action4` "
                       f"(21,700 updates, global batch 128, seed 42) on two B200 GPUs. Not evaluated.\n\n")
            card = card.replace('## Checkpoint index', section + '## Checkpoint index', 1)
        lines = [l for l in card.split('\n') if not l.startswith(f'| [{repo_path}]')]
        lines.insert(lines.index(head) + 4, row)
        lines.insert(lines.index('## Checkpoint index') + 4, row)
        n = len(index['checkpoints'])
        card = '\n'.join(lines)
        import re
        card = re.sub(r'This repository contains \*\*\d+ checkpoints\*\*', f'This repository contains **{n} checkpoints**', card, 1)
        (meta / 'README.md').write_text(card)
        api.upload_folder(repo_id=REPO, repo_type='model', folder_path=str(meta), allow_patterns=['checkpoint_index.json', 'README.md'],
                          commit_message=f'Index {repo_path}')
        remote = {f.rfilename: f for f in api.model_info(REPO, files_metadata=True).siblings}
        for name, x in expected.items():
            assert name in remote and remote[name].size == x['size'], f'missing or wrong-size Hub file {name}'
            lfs = remote[name].lfs
            if lfs is not None:
                assert (lfs.sha256 if hasattr(lfs, 'sha256') else lfs['sha256']) == x['sha256'], f'Hub checksum mismatch {name}'
        result = dict(repo=REPO, repo_path=repo_path, private=api.model_info(REPO).private,
                      revision=api.model_info(REPO).sha, verified_files=expected, checkpoints_in_index=n,
                      completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(train.parent / 'upload_verification.json', result)
        print(json.dumps(dict(event='uploaded', **{k: v for k, v in result.items() if k != 'verified_files'})), flush=True)
    finally:
        if delete_token:
            token_file.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['verify', 'publish'])
    p.add_argument('--train', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--stage')
    p.add_argument('--token-file')
    p.add_argument('--delete-token', action='store_true')
    a = p.parse_args()
    if a.command == 'verify':
        verify(a.train, a.reference)
    else:
        publish(a.train, a.reference, a.stage, a.token_file, a.delete_token)


if __name__ == '__main__':
    main()
