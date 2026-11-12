#!/usr/bin/env python3
"""Publish the GR1-E0 x5 elastic student (final epoch-5 export) to anhdao69/LoopWAM_NT under Robocasa-GR1/Elastic/E0/.

User, 2026-11-11: "Push to hf for me" / "push the epoch 5 for me" (GR1-E0 training finished 02:59 UTC, export verified
03:01; the official evaluations of epoch 5 are still running, so the README lists the ones that are valid at upload time).
Adapted from elastic_robotwin/ep5/hf_publish/publish_rt50_ep5_e2.py.

    env -u HF_HUB_OFFLINE python -I publish_gr1_e0_x5.py [--dry-run]   (login node; HF token from the local cache, never printed)

Refuses to upload unless: run status complete at update 235,160 (epoch 5); the orchestrator's export verification passed
for epoch_05.pt (after the file was last written, on the same bytes; incl. the eval-server check); the export's format,
step, parent and teacher hashes match; the teacher on HF (Robocasa-GR1/epoch_09.pt) has the same sha256; every floating
tensor is finite; the [1, 1] evaluation of this export is valid; nothing exists yet under Robocasa-GR1/Elastic/.
Robocasa-GR1/README.md gains one section at its end; no other existing file is touched. One commit uploads the folder;
afterwards every remote file must match the local bytes. A receipt is committed and kept locally.
"""
import calendar, gzip, hashlib, json, sys, time
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

REPO, PREFIX = 'anhdao69/LoopWAM_NT', 'Robocasa-GR1'
SUB = f'{PREFIX}/Elastic/E0'
B = Path('/mnt/data/vmo-ai-task/anhdh35/elastic_gr1')
R = B / 'runs/GR1-E0'
EV = B / 'eval'
EPOCH, STEP, WARMUP = 5, 235160, 7055
CK = R / 'epoch_05.pt'
VERIFY = B / 'validation/verify_GR1-E0_e05.json'
SELECTION = B / 'teacher_selection.json'
OUT = B / 'hf_publish'
TEACHER_SHA = '694a679b4c07934b57995dc3a2ab62e2f65357cea29108b109b30049531e2748'
TEACHER_FILE = 'Robocasa-GR1/epoch_09.pt'
ORDER = ['flat_k1_n2', 'flat_k2_n2', 'flat_k4_n2', 'flat_k4_n5', 'flat_k1_n5', 'flat_k2_n5', 'flat_k1_n10', 'flat_k2_n10', 'flat_k4_n10']
DRY = '--dry-run' in sys.argv


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def git_blob(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def tensors(obj):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from tensors(v)


def kn(s):
    k, n = (int(x[1:]) for x in s.split('_')[1:])
    return k, n


def label(s):
    k, n = kn(s)
    return f'[{k}, {k}]' if n == 2 else f'[{k}]x{n}'


def local_checks():
    st = json.loads((R / 'status.json').read_text())
    assert (st['status'], st['update'], st['epoch'], st['next_micro']) == ('complete', STEP, EPOCH, 0), st
    assert st['scheduler'] == dict(step=STEP, total=STEP, warmup=WARMUP), st['scheduler']
    ver = json.loads(VERIFY.read_text())
    e = ver['pairs']['epoch_05.pt']
    assert ver['passed'] is True and e['passed'] is True and all(e['checks'].values()), e['checks']
    assert (e['update'], e['epoch'], e['kind'], e['recovery']) == (STEP, EPOCH, 'T0', 'epoch_05_recovery.pt'), e
    assert e['rollout'] and all(x['bit_equal'] and x['finite'] for x in e['rollout']), e['rollout']
    assert sorted({x['schedule'] for x in ver['server']}) == sorted(ORDER) and all(x['bit_equal'] and x['finite'] for x in ver['server'])
    stat0 = CK.stat()
    assert stat0.st_mtime < ver['started'], 'epoch_05.pt changed after its verification'
    sha, size = sha256(CK), stat0.st_size
    assert e['export_sha256'] == sha, (e['export_sha256'], sha)
    cfg = json.loads((R / 'config.json').read_text())
    assert cfg['run'] == 'GR1-E0' and cfg['config']['kind'] == 'T0' and cfg['planned_updates'] == STEP and cfg['epochs'] == EPOCH
    par = cfg['config']['parent']
    assert cfg['asset_sha256']['teacher'] == TEACHER_SHA == par['sha256'] and (par['epoch'], par['step']) == (9, 211644), par
    sel = json.loads(SELECTION.read_text())
    assert (sel['chosen_epoch'], sel['teacher_sha256'], sel['step']) == (9, TEACHER_SHA, 211644)
    assert (R / 'dataset_stats.json').read_bytes() == (R / 'data/dataset_stats.json').read_bytes()
    p = torch.load(CK, map_location='cpu', mmap=True, weights_only=False)
    assert (p['format_version'], p['base_format_version'], p['step']) == ('loopwam-s-gr1-rt-v1', 'loopwam-s-gr1-v1', STEP)
    ts = p['training_state']
    assert (ts['update'], ts['epoch']) == (STEP, EPOCH), (ts['update'], ts['epoch'])
    assert p['transport_config']['kind'] == 'T0' and p['transport_config']['dataset'] == 'robocasa_gr1', p['transport_config']
    n = bad = 0
    for t in tensors({k: p[k] for k in ('mot', 'proprio_encoder', 'student_action', 'step_condition', 'transport')}):
        if t.is_floating_point():
            n += 1
            bad += not bool(torch.isfinite(t).all())
    assert n > 0 and bad == 0, (n, bad)
    assert sum(t.numel() for t in tensors(p['transport'])) == 0, 'T0 has no transport parameters'
    student = sum(t.numel() for t in tensors(p['student_action'])) + sum(t.numel() for t in tensors(p['step_condition']))
    assert student == cfg['trainable_parameters'], (student, cfg['trainable_parameters'])
    del p
    metrics = (R / 'metrics.jsonl').read_bytes()
    rows = [json.loads(l) for l in metrics.splitlines() if l.strip()]
    ups = [r['update'] for r in rows]
    assert ups == list(range(1, STEP + 1)), (len(ups), ups[:3], ups[-3:])
    last = rows[-2000:]
    mean = lambda k: sum(r[k] for r in last) / len(last)
    launches = [json.loads(l) for l in (R / 'launches.jsonl').read_text().splitlines() if l.strip()]
    assert [x['target_epoch'] for x in launches] == [1, 2, 3, 4, 5, 5] and all(x['revision'] == cfg['git_revision'] for x in launches), launches
    t0 = calendar.timegm(time.strptime(launches[0]['time'], '%Y-%m-%dT%H:%M:%SZ'))
    summ = dict(updates=STEP, total=mean('total'), end=mean('end'), local=mean('local'), anchor=mean('anchor'),
                grad_norm=mean('grad_norm'), final_lr=rows[-1]['lr'], wall_hours=(rows[-1]['time'] - t0) / 3600,
                started=launches[0]['time'], finished=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(rows[-1]['time'])),
                launches=len(launches))
    # official evaluations: this export (epoch 5), and the earlier epochs for context
    evals = {}
    for s in ORDER:
        f = EV / f'e05_{s}_validation.json'
        if not f.is_file():
            continue
        raw = f.read_bytes(); v = json.loads(raw)
        if v.get('valid') is True:
            assert (v['step'], v['schedule'], v['export_sha256'], v['episodes'], len(v['per_task'])) == (STEP, s, sha, 1200, 24), (s, v['step'])
            evals[s] = dict(raw=raw, successes=v['successes'])
    assert 'flat_k1_n2' in evals, 'the [1, 1] evaluation of epoch 5 is not valid yet'
    def old(e_, s):
        v = json.loads((EV / f'e{e_:02d}_{s}_validation.json').read_text())
        assert (v['step'], v['schedule'], v['episodes']) == (e_ * 47032, s, 1200), (e_, s)
        return v['successes']
    earlier = dict(k1=[old(i, 'flat_k1_n2') for i in (1, 2, 3, 4)], e4={s: old(4, s) for s in ('flat_k2_n2', 'flat_k4_n2', 'flat_k4_n5')})
    return dict(sha=sha, size=size, mtime=stat0.st_mtime, finite_tensors=n, student_params=student, cfg=cfg, ver=ver, summ=summ,
                metrics=metrics, evals=evals, earlier=earlier, sel=sel)


SECTION = """
## Elastic student (E0)

[`Elastic/E0/epoch_05.pt`](https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/Robocasa-GR1/Elastic/E0) is an elastic student distilled from `epoch_09.pt` with the
elastic E0 recipe (5 epochs): one checkpoint that runs with fewer denoising steps and fewer core loops than the teacher.
Files, recipe and official evaluation results: `Elastic/E0/README.md`.
"""

README = """# RoboCasa-GR1 elastic student E0 (T0), 5 epochs, distilled from the LoopWAM v0 GR1 epoch-9 teacher

Run `GR1-E0`: the elastic E0 recipe (T0 transport, FLAT9 schedules) distilled for five epochs of the RoboCasa-GR1 tabletop
data from the frozen teacher [`Robocasa-GR1/epoch_09.pt`](https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/Robocasa-GR1) (continuation epoch 9, step 211,644, sha256 `{teacher}`;
official eval 52.17%, 626/1200; chosen over epoch 10, 620/1200).
Training complete {finished} UTC; export verified.

One checkpoint serves every schedule. A schedule gives the number of core loops used at each denoising step: `[2, 2]` is
2 steps with 2 loops each; the teacher runs 10 steps with 4 loops each (`[4]x10`). Action-block passes per query are
steps x (3 entry + 6 x loops + 3 exit).

## Official evaluation of this checkpoint

Protocol (the teacher's): starVLA `examples/simBenchmarks/Robocasa_tabletop`, 24 tasks x 50 episodes, n_envs 1,
max_episode_steps 720, n_action_steps 12, state sent, env seed 0, fixed noise seed 42 per query, EGL rendering, H100.

| schedule | steps x loops | action-block passes | success |
|---|---|---|---|
{eval_rows}| teacher `epoch_09.pt`, [4]x10 | 10 x 4 | 300 | 52.17% (626/1200) |

{pending}

Earlier epochs of this run under the same protocol (their exports are not published): `[1, 1]` at the end of epochs 1-4 =
{k1_text}; epoch 4 at `[2, 2]` = {e4_22}, `[4, 4]` = {e4_44}, `[4]x5` = {e4_45}.

## Files

| file | content |
|---|---|
| `epoch_05.pt` | final export, update {steps:,} (end of epoch 5), format `loopwam-s-gr1-rt-v1`, sha256 `{sha}` |
| `training_config.json` | run config: recipe, teacher / VAE / init hashes, arguments, code revision |
| `dataset_stats.json`, `data_manifest.json` | 29-D normalization (required at inference) and data manifest of the training data |
| `schedule_sequence.json` | the loop schedule drawn for every update |
| `training_metrics.jsonl.gz`, `training_status.json`, `launches.jsonl` | per-update metrics and run records |
| `export_verification.json` | export check: format, frozen tensors bit-equal to the FP32 teacher, student == training state, bit-equal rollouts, eval server bit-equal for all nine schedules |
| `teacher_selection.json`, `teacher_provenance.json` | teacher choice (epoch 9 vs 10) and the Hugging Face LFS hash of the teacher |
| `eval/e05_<schedule>_validation.json` | official eval validation of this checkpoint, per-task successes |
| `SHA256SUMS` | hashes of the files in this folder |

The export is self-contained: the teacher's FP32 frozen video expert, proprio encoder, architecture and loop metadata,
plus the trained student action expert and step condition ({student:,} parameters).
{frozen}; {changed} student tensors trained. Load it with
`fastwam.models.wan22.transport_inference_gr1.load_gr1_transport(path, vae_path)` (Wan2.1 VAE); the starVLA evaluation is served by
`experiments/robocasa_gr1/elastic_policy_server.py --schedule <name>`. Code: branch `elastic_gr1` of LoopWAM_NT at revision `{rev}`
(not on GitHub at upload time).

Recipe: T0 (no transport), scope `pc` (whole student action expert trainable), step conditioning, anchor, regime `fu`, schedules FLAT9
(flat_k{{1,2,4}}_n{{2,5,10}}: 1, 2 or 4 core loops at each of 2, 5 or 10 denoising steps) with the balanced sampler, init `common_v1`, seed 42.
GR1 adaptations of the LIBERO recipe: endpoint loss over the 12 executed actions, no binarized gripper loss.
Data: RoboCasa-GR1 tabletop (same data and normalization as the teacher), {windows:,} windows, 5 epochs ({steps:,} updates, 47,032 per epoch).
Global batch 128 (4 x H100, microbatch 32). AdamW betas (0.9, 0.95), eps 1e-8; peak lr 3e-5 for the student action expert
(weight decay 0.01) and 1e-3 for the step condition (weight decay 0); 3% warmup ({warmup:,} updates), cosine to 1% at update {steps:,}; grad clip 1.0.
Elapsed {hours:.1f} h from the first launch to the last update ({launches} launches; every launch after the first resumed from `latest.pt`); this includes the
official evaluation run with training paused after each of epochs 1-4 and a 4-hour stop in epoch 5 at update 191,359 for
diagnostic evaluations. Last 2,000 updates: total loss {total:.5f} (end {end:.5f}, local {local:.5f}, anchor {anchor:.5f}),
grad norm {gn:.4f}.
"""


def pct(x):
    return f'{100 * x / 1200:.2f}% ({x}/1200)'


def main():
    L = local_checks()
    s, cfg, ver = L['summ'], L['cfg'], L['ver']
    e = ver['pairs']['epoch_05.pt']
    print(json.dumps(dict(file=str(CK), sha256=L['sha'], size=L['size'], finite_float_tensors=L['finite_tensors'],
                          student_params=L['student_params'], evals={k: v['successes'] for k, v in L['evals'].items()},
                          earlier=L['earlier'], **{k: v for k, v in s.items()})), flush=True)
    api = HfApi()
    assert api.whoami()['name'] == 'anhdao69'
    info0 = api.model_info(REPO)
    files = api.list_repo_files(REPO)
    assert not [f for f in files if f.startswith(f'{PREFIX}/Elastic/')], f'{PREFIX}/Elastic/ already exists on HF'
    print(json.dumps(dict(repo=REPO, private=info0.private, head=info0.sha)), flush=True)
    tr = api.get_paths_info(REPO, [TEACHER_FILE], expand=True, revision=info0.sha)
    assert len(tr) == 1 and tr[0].lfs is not None and tr[0].lfs.sha256 == TEACHER_SHA, tr
    provenance = json.dumps({TEACHER_FILE: dict(lfs_sha256=tr[0].lfs.sha256, size=tr[0].size)}, indent=1).encode()
    extra = {'training_config.json': (R / 'config.json').read_bytes(),
             'dataset_stats.json': (R / 'data/dataset_stats.json').read_bytes(),
             'data_manifest.json': (R / 'data/data_manifest.json').read_bytes(),
             'schedule_sequence.json': (R / 'schedule_sequence.json').read_bytes(),
             'training_metrics.jsonl.gz': gzip.compress(L['metrics'], mtime=0),
             'training_status.json': (R / 'status.json').read_bytes(),
             'launches.jsonl': (R / 'launches.jsonl').read_bytes(),
             'export_verification.json': VERIFY.read_bytes(),
             'teacher_selection.json': SELECTION.read_bytes(),
             'teacher_provenance.json': provenance}
    for sch, v in L['evals'].items():
        extra[f'eval/e05_{sch}_validation.json'] = v['raw']
    rows = ''
    for sch in ORDER:
        if sch in L['evals']:
            k, n = kn(sch)
            rows += f"| `{sch}` {label(sch)} | {n} x {k} | {n * (6 + 6 * k)} | {pct(L['evals'][sch]['successes'])} |\n"
    missing = [x for x in ORDER if x not in L['evals']]
    pending = ('All nine FLAT9 schedules are evaluated.' if not missing else
               'The other FLAT9 schedules (' + ', '.join(f'`{x}`' for x in missing) + ') were still being evaluated at upload time ('
               + time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()) + ').')
    ea = L['earlier']
    readme = README.format(teacher=TEACHER_SHA, finished=s['finished'].replace('T', ' ').rstrip('Z'), steps=STEP, sha=L['sha'],
                           student=L['student_params'], frozen=e['frozen'].replace('bit-equal', 'bit-equal to the FP32 teacher'),
                           changed=e['student_tensors_changed'], rev=cfg['git_revision'][:7], windows=cfg['windows'], warmup=WARMUP,
                           hours=s['wall_hours'], launches=s['launches'], total=s['total'], end=s['end'], local=s['local'],
                           anchor=s['anchor'], gn=s['grad_norm'], eval_rows=rows, pending=pending,
                           k1_text=', '.join(pct(x) for x in ea['k1']), e4_22=pct(ea['e4']['flat_k2_n2']),
                           e4_44=pct(ea['e4']['flat_k4_n2']), e4_45=pct(ea['e4']['flat_k4_n5'])).encode()
    extra['README.md'] = readme
    sums = ''.join(f'{hashlib.sha256(b).hexdigest()}  {n}\n' for n, b in sorted(extra.items())) + f"{L['sha']}  epoch_05.pt\n"
    extra['SHA256SUMS'] = sums.encode()
    ops = [CommitOperationAdd(f'{SUB}/epoch_05.pt', str(CK))] + [CommitOperationAdd(f'{SUB}/{n}', b) for n, b in extra.items()]
    expect = {f'{SUB}/epoch_05.pt': ('sha256', L['sha'], L['size'])}
    expect.update({f'{SUB}/{n}': ('bytes', b, len(b)) for n, b in extra.items()})
    top = Path(hf_hub_download(REPO, f'{PREFIX}/README.md', revision=info0.sha)).read_text()
    assert '## Elastic student' not in top, 'Robocasa-GR1/README.md already has the elastic section'
    top = top.rstrip('\n') + '\n' + SECTION
    ops.append(CommitOperationAdd(f'{PREFIX}/README.md', top.encode()))
    expect[f'{PREFIX}/README.md'] = ('bytes', top.encode(), len(top.encode()))
    assert all(o.path_in_repo.startswith(SUB + '/') or o.path_in_repo == f'{PREFIX}/README.md' for o in ops), [o.path_in_repo for o in ops]
    print('PLAN ' + json.dumps({p: v[2] for p, v in expect.items()}), flush=True)
    OUT.mkdir(exist_ok=True)
    if DRY:
        (OUT / 'dry_run_README.md').write_bytes(readme)
        (OUT / 'dry_run_top_README.md').write_bytes(top.encode())
        print('DRY RUN: nothing uploaded', flush=True)
        return
    t = time.time()
    info = api.create_commit(REPO, operations=ops, commit_message=f'{SUB}: GR1-E0 x5 elastic student (T0, epoch-9 teacher), epoch-5 export')
    print(json.dumps(dict(commit=info.oid, seconds=round(time.time() - t))), flush=True)
    st1 = CK.stat()
    assert (st1.st_size, st1.st_mtime) == (L['size'], L['mtime']), 'epoch_05.pt changed during upload'
    remote = {r.path: r for r in api.get_paths_info(REPO, list(expect), expand=True, revision=info.oid)}
    checked = {}
    for path, (kind, val, size) in expect.items():
        r = remote[path]
        if kind == 'sha256':
            ok = r.lfs is not None and r.lfs.sha256 == val and r.size == size
        else:
            ok = (r.lfs.sha256 == hashlib.sha256(val).hexdigest()) if r.lfs is not None else (r.blob_id == git_blob(val))
            ok = ok and r.size == size
        assert ok, f'remote verification failed for {path}: {r}'
        checked[path] = size
    rec = dict(repo=REPO, path=f'{SUB}/epoch_05.pt', sha256=L['sha'], size=L['size'], step=STEP, run='GR1-E0 x5',
               evals_at_upload={k: v['successes'] for k, v in L['evals'].items()},
               revision=info.oid, files_verified=checked, verified_utc=time.time())
    (OUT / 'GR1-E0.epoch_05.pt.uploaded.json').write_text(json.dumps(rec, indent=2))
    api.create_commit(REPO, operations=[CommitOperationAdd(f'{SUB}/epoch_05.pt.uploaded.json', json.dumps(rec, indent=2).encode())],
                      commit_message=f'{SUB}: upload receipt')
    print('VERIFIED ' + json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
