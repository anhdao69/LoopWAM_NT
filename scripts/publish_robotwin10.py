#!/usr/bin/env python3
"""Verify and publish a finished RoboTwin-10 run to Hugging Face (anhdao69/LoopWAM_NT, robotwin-10/<variant>/).

python scripts/publish_robotwin10.py --variant v0-video4-action4 [--dry-run]
python scripts/publish_robotwin10.py --index      # after all seven: folder README, root README rows, checkpoint_index

The uploaded policy.pt is the unmodified epoch-5 checkpoint (native loopwam-s-robot-v1, FP32, no optimizer). Before
upload: payload/variant validation, final cursor, every model tensor finite FP32 and exactly equal to the final
recovery checkpoint (latest.pt). After upload: Hub LFS SHA256 and sizes are checked against the local files.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from fastwam.datasets.chronoloop_robot_tasks import LILAWAM_10
from fastwam.models.wan22.loopwam_robot import VARIANTS, payload_variant, robot_parameter_count, validate_robot_payload

REPO = 'anhdao69/LoopWAM_NT'
PREFIX = 'robotwin-10'
RUNS = Path('/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10')
WORK = Path('/groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_work')
TOKEN = Path('/groups/yshang/an221229/cache/huggingface/token')
WINDOWS, UPDATES, EPOCHS = 1163351, 45445, 5
LABEL = {'v0-video4-action4': ('v0', '4/4'), 'v0-video4-action2': ('v0 aligned', '4/2'), 'v0-video4-action1': ('v0 aligned', '4/1'),
         'v0-video2-action2': ('v0 aligned', '2/2'), 'v0-video1-action4': ('v0 aligned', '1/4'),
         'dense-s12': ('dense_s12', '1/1'), 'dense-s30': ('dense_s30', '1/1')}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


def model_tensors(payload):
    out = {f'mot.{k}': v for k, v in payload['mot'].items()}
    out.update({f'proprio_encoder.{k}': v for k, v in payload['proprio_encoder'].items()})
    return out


def segments(train):
    """Training segments from the resume manifests (job, host, GPUs, start update)."""
    rows = []
    for path in sorted(train.glob('*manifest*.json'), key=lambda p: p.stat().st_mtime):
        if path.name not in ('manifest.json',) and not path.name.startswith('resume_manifest_u'):
            continue
        m = json.loads(path.read_text())
        rows.append(dict(file=path.name, start_update=m.get('resumed_from_update') or 0, slurm_job_id=m.get('slurm_job_id'),
                         hostname=m.get('hostname'), world_size=m.get('world_size'), microbatch=m.get('microbatch'),
                         gpu=m.get('gpu'), git_revision=m.get('git_revision'), git_dirty=m.get('git_dirty')))
    return sorted(rows, key=lambda r: r['start_update'])


def metrics_summary(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert [r['update'] for r in rows] == list(range(1, UPDATES + 1)), 'metrics are not one record per update'
    per_epoch = {}
    for r in rows:
        e = per_epoch.setdefault(r['epoch'], dict(n=0, loss=0., video=0., action=0.))
        e['n'] += 1; e['loss'] += r['loss']; e['video'] += r['loss_video']; e['action'] += r['loss_action']
    last = rows[-100:]
    return rows, dict(
        final_loss=rows[-1]['loss'], final_loss_video=rows[-1]['loss_video'], final_loss_action=rows[-1]['loss_action'],
        last100_loss=sum(r['loss'] for r in last) / len(last), last100_loss_video=sum(r['loss_video'] for r in last) / len(last),
        last100_loss_action=sum(r['loss_action'] for r in last) / len(last),
        epoch_mean={int(k): {m: v[m] / v['n'] for m in ('loss', 'video', 'action')} for k, v in sorted(per_epoch.items())},
        nonfinite=sum(not all(map(lambda x: x == x and abs(x) != float('inf'), (r['loss'], r['grad_norm']))) for r in rows),
        median_seconds_per_update=sorted(r['seconds'] for r in rows)[len(rows) // 2],
        peak_allocated_gb=max(r['peak_allocated_gb'] for r in rows))


def readme(variant, cfg, prov, timing, summary):
    version, loops = LABEL[variant]
    depth = ('Dense-S30 runs the 30 native Wan2.1 block pairs once' if variant == 'dense-s30' else
             'Dense-S12 runs the 12 LoopWAM-S donor block pairs (3 + 6 + 3) once' if variant == 'dense-s12' else
             f'v0 with stored partition 3/6/3, {cfg["video_loops"]} video and {cfg["action_loops"]} action core loops, aligned (late) action KV')
    hours = timing['elapsed_training_seconds'] / 3600
    segs = timing['segments']
    gpus = sorted({s['gpu'] for s in segs if s.get('gpu')})
    return f"""# LoopWAM {variant}: RoboTwin-10, five epochs

`policy.pt` is the unmodified, completed **epoch-5** checkpoint (update {UPDATES:,}) of LoopWAM `{variant}`:
{depth}; final-exit flow loss only, no memory or history. {cfg['policy_parameters']:,} policy parameters.

**Data: RoboTwin-10**, the 10-task LiLa-WAM ablation subset of the
[FastWAM RoboTwin 2.0 release](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam):
{', '.join(LILAWAM_10)}. 50 clean + 500 randomized demonstrations per task (5,500 episodes),
{WINDOWS:,} frame-anchor windows per epoch, three cameras packed at 384x320, nine frames at offsets 0,4,...,32,
32 absolute 14-D joint actions, published global z-score normalization (`dataset_stats.json`).

**Recipe** (the full-RoboTwin v0 recipe, restricted to these tasks, identical for all seven RoboTwin-10 models):
fresh canonical Wan2.1 donor initialization (no LIBERO or embodied pretraining) plus CPU-seeded (42) 14-D action
encoder/head and proprio encoder; exactly 128 real windows per update, every window once per epoch
(`randperm(42 + epoch)`), the 87-window tail averaged over its count; 5 epochs = {UPDATES:,} updates and
{WINDOWS * EPOCHS:,} window presentations; per-window keyed noise/timesteps; AdamW lr 1e-4, betas (0.9, 0.95),
weight decay 0.01, clip 1.0, 5% warmup, cosine to 1e-6; FP32 weights and Adam moments, BF16 compute. Data order,
noise and initialization are the same for every RoboTwin-10 model; only the architecture differs.

Training took {hours:.1f} h of training time on {', '.join(gpus) or 'H100'} in {len(segs)} segment(s)
(see `training_timing.json`). Final-100-update mean loss {summary['last100_loss']:.4f}
(video {summary['last100_loss_video']:.4f}, action {summary['last100_loss_action']:.4f}); no nonfinite values.
`training_metrics.jsonl.gz` holds every update.

**Not evaluated:** this is a training-only release; no RoboTwin simulator success rate is claimed.

## Loading

Use the robot loader from the LoopWAM_NT code (`fastwam.models.wan22.loopwam_robot`, revision
`{cfg['training_source_revision'][:12]}` or later with the RoboTwin-10 variant support); the variant is read from
the checkpoint:

```python
from fastwam.models.wan22.loopwam_robot import create_loopwam_robot

model, payload = create_loopwam_robot(
    vae_path="/path/to/Wan2.1_VAE.pth",
    checkpoint_path="/path/to/policy.pt",
    device="cuda",
    checkpoint_blocks=False,
)
model.eval()
```

Native format `loopwam-s-robot-v1` (FP32 weights, BF16 autocast for inference, no optimizer state). The Wan2.1 VAE and
T5 text encoder/cache are external. Use `dataset_stats.json` for actions and proprioception.

## Integrity

SHA-256 of `policy.pt`: `{prov['sha256']}`. `SHA256SUMS` covers every file in this folder. Every model tensor equals
the final optimizer checkpoint of the run exactly.
"""


def publish(variant, dry_run=False):
    run = RUNS / variant; train = run / 'train'
    assert (train / 'COMPLETE').exists(), 'run not complete'
    timing = json.loads((train / 'timing.json').read_text())
    assert timing['status'] == 'complete' and timing['update'] == UPDATES and timing['windows_seen'] == WINDOWS * EPOCHS
    final, latest = train / 'checkpoints' / 'epoch_05.pt', train / 'latest.pt'
    p = torch.load(final, map_location='cpu', mmap=True, weights_only=False)
    validate_robot_payload(p, variant); assert payload_variant(p) == variant and 'optimizer' not in p
    st = p['training_state']; c = st['contract']
    assert (p['step'], st['update'], st['windows_seen'], st['next_epoch'], st['next_batch']) == (UPDATES, UPDATES, WINDOWS * EPOCHS, 5, 0)
    assert c['variant'] == variant and c['planned_updates'] == UPDATES and c['tasks'] == list(LILAWAM_10)
    q = torch.load(latest, map_location='cpu', mmap=True, weights_only=False)
    assert q['step'] == UPDATES and q['training_state']['update'] == UPDATES
    a, b = model_tensors(p), model_tensors(q)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), 'epoch_05 differs from final latest.pt'
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in a.values())
    count = sum(v.numel() for v in a.values()); assert count == robot_parameter_count(variant)
    del q, b
    manifest = json.loads((train / 'manifest.json').read_text())
    rows, summary = metrics_summary(train / 'metrics.jsonl')
    assert summary['nonfinite'] == 0
    prov = dict(epoch=5, step=UPDATES, sha256=sha256(final), bytes=final.stat().st_size, native_checkpoint_unmodified=True,
                optimizer_included=False, finite_fp32_tensors_verified=len(a), policy_parameters=count,
                equal_to_final_recovery_checkpoint=True, created_utc=utc())
    version, video_loops, action_loops = VARIANTS[variant]
    cfg = dict(format_version=p['format_version'], model=f'LoopWAM {variant}', variant=variant, version=version, epoch=5,
               optimizer_update=UPDATES, stored_partition=c['partition'], video_loops=video_loops, action_loops=action_loops,
               action_kv_mode='aligned', loop_alignment='late', memory_tokens=0, history_frame=0, action_dim=14, proprio_dim=14,
               policy_parameters=count, training_contract=c, initialization=p['architecture']['initialization_mode'],
               init_artifact_sha256=p['architecture']['init_artifact_sha256'], donor_indices=p['architecture']['donor_indices'],
               loader='fastwam.models.wan22.loopwam_robot.create_loopwam_robot',
               training_source_revision=manifest['git_revision'], training_source_dirty=manifest['git_dirty'],
               training_code_sha256=manifest['code_files_sha256'], model_weights_dtype='float32', compute_dtype='bfloat16')
    stats = json.loads((WORK / 'shared' / 'data' / 'dataset_stats.json').read_text())
    data_manifest = json.loads((WORK / 'shared' / 'data' / 'data_manifest.json').read_text())
    data = dict(source='https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam', subset='RoboTwin-10 (LiLa-WAM ablation subset, arXiv 2608.03701 Sec. 4.2)',
                tasks=list(LILAWAM_10), clean_demonstrations_per_task=50, randomized_demonstrations_per_task=500,
                total_demonstrations=5500, windows_per_epoch=WINDOWS, group_windows=data_manifest['group_windows'], epochs=EPOCHS,
                windows_presented=WINDOWS * EPOCHS, global_batch=128, held_out_training_subset=False, video_shape=[9, 3, 384, 320],
                frame_offsets=list(range(0, 33, 4)), action_horizon=32, action_dim=14, proprio_dim=14,
                normalization='published global z-score', normalization_sha256=c['normalization_sha256'],
                dataset_stats_file_sha256=sha256('/groups/yshang/an221229/data/robotwin2.0-fastwam/dataset_stats.json'),
                data_manifest_sha256=c['data_manifest_sha256'], latent_cache='frozen Wan2.1 VAE, per-clip reset, exact BF16 bits')
    segs = segments(train)
    timing_out = dict(status='complete', update=UPDATES, planned_updates=UPDATES, windows_seen=WINDOWS * EPOCHS,
                      elapsed_training_seconds=timing['elapsed_training_seconds'], segments=segs,
                      first_update_utc=datetime.fromtimestamp((train / 'manifest.json').stat().st_mtime, timezone.utc).isoformat(),
                      complete_utc=datetime.fromtimestamp((train / 'COMPLETE').stat().st_mtime, timezone.utc).isoformat(),
                      median_seconds_per_update=summary['median_seconds_per_update'], peak_allocated_gb=summary['peak_allocated_gb'])
    stage = run / 'hf_stage'; stage.mkdir(exist_ok=True)
    files = {'config.json': cfg, 'checkpoint_provenance.json': prov, 'data_provenance.json': data,
             'training_timing.json': timing_out, 'training_summary.json': summary, 'dataset_stats.json': stats}
    for name, payload in files.items():
        (stage / name).write_text(json.dumps(payload, indent=2) + '\n')
    with gzip.open(stage / 'training_metrics.jsonl.gz', 'wt') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)
    (stage / 'README.md').write_text(readme(variant, cfg, prov, timing_out, summary))
    local = {n: stage / n for n in sorted(os.listdir(stage)) if n != 'SHA256SUMS'}
    local['policy.pt'] = final
    sums = {n: (prov['sha256'] if n == 'policy.pt' else sha256(f)) for n, f in local.items()}
    (stage / 'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n, h in sorted(sums.items())))
    local['SHA256SUMS'] = stage / 'SHA256SUMS'; sums['SHA256SUMS'] = sha256(stage / 'SHA256SUMS')
    if dry_run:
        print(json.dumps(dict(dry_run=True, variant=variant, files={n: str(f) for n, f in local.items()}, provenance=prov,
                              summary={k: v for k, v in summary.items() if k != 'epoch_mean'}), indent=2))
        return
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=TOKEN.read_text().strip())
    ops = [CommitOperationAdd(path_in_repo=f'{PREFIX}/{variant}/{n}', path_or_fileobj=str(f)) for n, f in local.items()]
    info = api.create_commit(repo_id=REPO, operations=ops,
                             commit_message=f'Add RoboTwin-10 {variant} (5 epochs, {UPDATES} updates, training only)')
    revision = info.oid
    remote = {s.rfilename: s for s in api.model_info(REPO, revision=revision, files_metadata=True).siblings
              if s.rfilename.startswith(f'{PREFIX}/{variant}/')}
    check = {}
    for n, f in local.items():
        s = remote[f'{PREFIX}/{variant}/{n}']
        lfs = None if s.lfs is None else (s.lfs.sha256 if hasattr(s.lfs, 'sha256') else s.lfs['sha256'])
        ok = s.size == f.stat().st_size and (lfs == sums[n] if n == 'policy.pt' else lfs in (None, sums[n]))
        check[n] = dict(size=s.size, lfs_sha256=lfs, local_sha256=sums[n], ok=ok)
    assert all(v['ok'] for v in check.values()), check
    record = dict(variant=variant, repo=REPO, path=f'{PREFIX}/{variant}', revision=revision, policy_sha256=prov['sha256'],
                  files=check, published_utc=utc(), host=socket.gethostname())
    (run / 'upload_verification.json').write_text(json.dumps(record, indent=2))
    (run / 'PUBLISHED').write_text(json.dumps(dict(revision=revision, time=utc())) + '\n')
    print(json.dumps(dict(published=variant, revision=revision), indent=2))


def index(dry_run=False):
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    api = HfApi(token=TOKEN.read_text().strip())
    order = ['v0-video4-action4', 'dense-s12', 'dense-s30', 'v0-video4-action2', 'v0-video4-action1', 'v0-video2-action2',
             'v0-video1-action4']
    recs = {}
    for v in order:
        run = RUNS / v
        ver = json.loads((run / 'upload_verification.json').read_text())
        st = json.loads((run / 'hf_stage' / 'training_timing.json').read_text())
        su = json.loads((run / 'hf_stage' / 'training_summary.json').read_text())
        cf = json.loads((run / 'hf_stage' / 'config.json').read_text())
        recs[v] = dict(ver=ver, timing=st, summary=su, config=cf)
    head = api.model_info(REPO).sha
    rows = []
    for v in order:
        r = recs[v]; version, loops = LABEL[v]
        rows.append(f'| [{PREFIX}/{v}]({PREFIX}/{v}) | {version} | {loops} | {r["config"]["policy_parameters"]:,} | '
                    f'{r["timing"]["elapsed_training_seconds"] / 3600:.1f} h | {r["summary"]["last100_loss"]:.4f} | Training only; evaluation not run |')
    table = ('| Directory | Model | Video/action loops | Parameters | Training time | Final loss (last 100) | Evaluation |\n'
             '|---|---|---|---:|---:|---:|---|\n' + '\n'.join(rows) + '\n')
    folder = f"""# RoboTwin-10 checkpoints

Seven LoopWAM models trained on **RoboTwin-10**, the 10-task LiLa-WAM ablation subset of the FastWAM RoboTwin 2.0
release ({', '.join(LILAWAM_10)}; 50 clean + 500 randomized demonstrations per task, 5,500 episodes,
{WINDOWS:,} windows per epoch). Every model uses the same recipe as `robotwin/v0-video4-action4` restricted to these
tasks: fresh canonical Wan2.1 donors plus seeded 14-D robot interfaces, exactly 128 windows per update, five epochs
({UPDATES:,} updates), seed 42, identical data order and per-window noise, AdamW 1e-4 with 5% warmup and cosine to
1e-6. Only the architecture differs. Each folder has its own README, config, provenance, metrics and SHA256SUMS.

{table}
**Not evaluated:** training-only releases; no RoboTwin success rate is claimed.
"""
    readme_path = hf_hub_download(REPO, 'README.md', revision=head, token=api.token)
    root = Path(readme_path).read_text()
    section = (f'## November 2026 addition: RoboTwin-10 (LiLa-WAM 10-task subset)\n\n'
               f'Seven LoopWAM architectures trained for five epochs on the 10-task RoboTwin-10 subset with the same '
               f'data order, noise, initialization recipe and schedule; see [{PREFIX}/README.md]({PREFIX}/README.md).\n\n'
               + table + '\nTraining only; evaluation not run.\n\n')
    if f'{PREFIX}/v0-video4-action4' not in root:
        marker = '## Checkpoint index'
        if marker in root:
            root = root.replace(marker, section + marker, 1)
            lines = root.split('\n'); i = next(j for j, line in enumerate(lines) if line.strip() == marker)
            # Insert rows at the top of the index table (after its header separator).
            k = next(j for j in range(i, len(lines)) if lines[j].startswith('|---'))
            idx_rows = [f'| [{PREFIX}/{v}]({PREFIX}/{v}) | {LABEL[v][0]} (RoboTwin-10) | {LABEL[v][1]} | '
                        f'{recs[v]["config"]["policy_parameters"]:,} | Training only; evaluation not run |' for v in order]
            lines[k + 1:k + 1] = idx_rows
            root = '\n'.join(lines)
        else:
            root = root.rstrip('\n') + '\n\n' + section
        m = re.search(r'This repository contains \*\*(\d+) checkpoints\*\*', root)
        if m:
            root = root.replace(m.group(0), f'This repository contains **{int(m.group(1)) + len(order)} checkpoints**', 1)
    idx_path = hf_hub_download(REPO, 'checkpoint_index.json', revision=head, token=api.token)
    idx = json.loads(Path(idx_path).read_text())
    have = {c.get('repo_path') for c in idx['checkpoints']}
    for v in order:
        if f'{PREFIX}/{v}' in have:
            continue
        r = recs[v]
        idx['checkpoints'].append(dict(
            id=f'robotwin10_{v.replace("-", "_")}', repo_path=f'{PREFIX}/{v}', training_path=str(RUNS / v / 'train'),
            export_sha256=r['ver']['policy_sha256'], export_bytes=r['ver']['files']['policy.pt']['size'],
            step=UPDATES, format='loopwam-s-robot-v1', optimizer_included=False, version=r['config']['version'],
            video_loops=r['config']['video_loops'], action_loops=r['config']['action_loops'], loop_alignment='late',
            parameters=r['config']['policy_parameters'], dataset_scope='robotwin10', tasks=list(LILAWAM_10), epochs=EPOCHS,
            global_batch=128, training_seed=42, initialization_mode=r['config']['initialization'],
            evaluation_status='not_evaluated', training_seconds=r['timing']['elapsed_training_seconds'],
            final_loss_last100=r['summary']['last100_loss'], tensor_equality_verified=True,
            source_revision=r['config']['training_source_revision'], hf_revision=r['ver']['revision']))
    idx['updated_utc'] = utc()
    stage = RUNS / 'index_stage'; stage.mkdir(exist_ok=True)
    (stage / 'README.md').write_text(root); (stage / 'checkpoint_index.json').write_text(json.dumps(idx, indent=2) + '\n')
    (stage / 'robotwin10_README.md').write_text(folder)
    if dry_run:
        print(folder); print(section); return
    info = api.create_commit(repo_id=REPO, parent_commit=head, commit_message='Index: RoboTwin-10 seven-model release',
                             operations=[CommitOperationAdd('README.md', str(stage / 'README.md')),
                                         CommitOperationAdd('checkpoint_index.json', str(stage / 'checkpoint_index.json')),
                                         CommitOperationAdd(f'{PREFIX}/README.md', str(stage / 'robotwin10_README.md'))])
    files = {s.rfilename: s.size for s in api.model_info(REPO, revision=info.oid, files_metadata=True).siblings}
    assert all(f'{PREFIX}/{v}/policy.pt' in files for v in order) and f'{PREFIX}/README.md' in files
    (RUNS / 'INDEX_PUBLISHED').write_text(json.dumps(dict(revision=info.oid, time=utc())) + '\n')
    print(json.dumps(dict(index=info.oid), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=sorted(VARIANTS)); ap.add_argument('--index', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.index:
        index(args.dry_run)
    else:
        publish(args.variant, args.dry_run)
