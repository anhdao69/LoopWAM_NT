#!/usr/bin/env python3
"""Export the scored records of the nine RoboTwin-50 elastic evaluations run on sn4622128794 (2026-11-07 .. 2026-11-14).

Run on sn4622128794. Sources are only read: the evaluation output folders /data/anh/robotwin_eval/results/<run>, the
pre-scoring verification records /data/anh/robotwin_eval/verification and the evaluation checkout
/data/anh/code/LoopWAM-robotwin-eval. Before anything is written, every run is checked again:
  * audit.json and the reference seed check passed; COMPLETE.json, manifest.json and audit.json name the same checkpoint
    and schedule; exactly the 100 task/setting cells of the contract, each with 100 accepted episodes;
  * each accepted cell's result.json equals its COMPLETE.json entry, success counts equal the episode flags, scene
    seeds are unique within a cell, and every episode video lies in the accepted attempt and exists and is non-empty;
  * one video per cell (900 in total) is rehashed against the SHA-256 recorded when it was written;
  * the 789 runtime files the run manifest hashed still have those hashes (the code that ran is the code archived here).

    python export_records.py OUT_DIR      (OUT_DIR = this report folder)
"""
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

RES = Path('/data/anh/robotwin_eval/results')
VER = Path('/data/anh/robotwin_eval/verification')
CODE = Path('/data/anh/code/LoopWAM-robotwin-eval')
REPORT_ROOT = Path(__file__).resolve().parents[2]          # evaluation_report/
HOST = 'sn4622128794'

E0 = dict(group='e0_teacher10_5ep', name='E0-epoch10-teacher-5ep', reference='robotwin_v0_epoch10',
          checkpoint=dict(hf_repo='anhdao69/LoopWAM_NT', hf_path='robotwin/Elastic/E0-epoch10-teacher-5ep/epoch_05.pt',
                          hf_revision='b72f355505d98c473fb91e45d10e16d4cc7ce536', size=2340331885, kind='T0',
                          epoch=5, step=237310,
                          sha256='07d24ff77410e516ac90d24d76d651e2102b09b0458d9d5d86ea18dff39e823e'),
          teacher=dict(hf_path='robotwin/v0-video4-action4-epoch10/policy.pt', epoch=10, step=474620,
                       sha256='70cf5af73db5fe582c50c5f9fb68296d5271a9c0fea51509b3cdb5f9696be1ea'))
E1 = dict(group='e1_epoch1', name='E1 (RT50-ep5-E1)', reference='robotwin_v0',
          checkpoint=dict(hf_repo='anhdao69/LoopWAM_NT', hf_path='robotwin/Elastic/E1/epoch_01.pt',
                          hf_revision='a03864919c3932c2af686d137cdab75328e41da7', size=2340331629, kind='T1',
                          epoch=1, step=47462,
                          sha256='9eba3aade1417b4425be5593c88e81441cf71bb4b6f16a8d81012ad6146e1bcb'),
          teacher=dict(hf_path='robotwin/v0-video4-action4/policy.pt', epoch=5, step=237310,
                       sha256='38975220e9441303878d2dfb9956d57e6416c6a906c99c43d9bca015b092fc51'))
RUNS = [(E1, s, f'e1_{s}') for s in ('flat_k1_n2', 'flat_k4_n2', 'flat_k4_n5')] + \
       [(E0, s, f'e0ep10_{s}') for s in ('flat_k1_n5', 'flat_k1_n10', 'flat_k2_n5', 'flat_k2_n10', 'flat_k4_n5',
                                          'flat_k4_n10')]
SETTINGS = ('demo_clean', 'demo_randomized')
EP_FIELDS = ['task', 'setting', 'episode_index', 'seed', 'success', 'video', 'video_sha256', 'video_verified',
             'instruction', 'queries', 'query_trace_sha256']
sources = []                                               # rows of SOURCES.tsv


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def label(schedule):
    k, n = (int(x[1:]) for x in schedule.split('_')[1:])
    return f'[{k},{k}]' if n == 2 else f'[{k}]x{n}'


def utc(t):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t)) + ' UTC'


def put(out, rel, data, source=None, transform='derived'):
    """Write bytes to OUT/rel; record its origin in SOURCES.tsv."""
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    if source is not None:
        sources.append([rel, f'{HOST}:{source}', sha256_file(source), str(os.path.getsize(source)), transform])
    else:
        sources.append([rel, 'derived', '', '', transform])


def copy(out, rel, src):
    put(out, rel, Path(src).read_bytes(), src, 'copy')


def copy_gz(out, rel, src):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0, filename='') as g:
        g.write(Path(src).read_bytes())
    put(out, rel, buf.getvalue(), src, 'gzip (mtime 0)')


def check_run(model, schedule, run):
    d = RES / run
    problems = []
    manifest = json.loads((d / 'manifest.json').read_text())
    complete = json.loads((d / 'COMPLETE.json').read_text())
    audit = json.loads((d / 'audit.json').read_text())
    seedck = json.loads((d / 'campaign_seed_check.json').read_text())
    sha = model['checkpoint']['sha256']
    if not (audit['passed'] and not audit['problems']): problems.append('audit not passed')
    if not seedck['passed']: problems.append('reference seed check not passed')
    for name, rec in (('manifest', manifest), ('COMPLETE', complete), ('audit', audit), ('seed check', seedck)):
        if rec['checkpoint_sha256'] != sha: problems.append(f'{name} checkpoint sha')
    for name, rec in (('manifest', manifest), ('COMPLETE', complete), ('audit', audit), ('seed check', seedck)):
        if rec['schedule'] != schedule: problems.append(f'{name} schedule')
    if manifest['expected_step'] != model['checkpoint']['step'] or manifest['transport_kind'] != model['checkpoint']['kind']:
        problems.append('manifest step/kind')
    if (manifest['seed'], manifest['episodes'], manifest['action_horizon'], manifest['replan_steps']) != (42, 100, 32, 24):
        problems.append('manifest protocol fields')
    if complete['failed'] or complete['running'] or complete['pending']: problems.append('COMPLETE not final')
    cells = complete['finished']
    want = {f'{t}/{s}' for t in manifest['tasks'] for s in manifest['settings']}
    if set(cells) != want or len(want) != 100: problems.append('cell coverage')
    rows, tasks, rehashed = [], [], 0
    for i, key in enumerate(sorted(cells)):
        rec = cells[key]
        cell_dir = Path(rec['directory']).parent
        if json.loads((cell_dir / 'result.json').read_text()) != rec: problems.append(f'{key}: result.json != COMPLETE')
        eps = rec['episodes']
        if len(eps) != 100: problems.append(f'{key}: {len(eps)} episodes')
        if sum(bool(e['success']) for e in eps) != rec['successes']: problems.append(f'{key}: success count')
        if len({e['seed'] for e in eps}) != len(eps): problems.append(f'{key}: duplicate seeds')
        for j, e in enumerate(eps):
            v = Path(e['video'])
            if v.parent != Path(rec['directory']) or not v.is_file() or v.stat().st_size == 0 or not e['video_verified']:
                problems.append(f'{key}: video {j}')
            rows.append([rec['task'], rec['config'], j, e['seed'], e['success'], e['video'], e['video_sha256'],
                         e['video_verified'], e['instruction'], e['queries'], e['query_trace_sha256']])
        j = (i * 37) % len(eps)                                  # one deterministic video per cell
        if sha256_file(eps[j]['video']) != eps[j]['video_sha256']: problems.append(f'{key}: video {j} sha')
        rehashed += 1
        tasks.append([rec['task'], rec['config'], rec['successes'], len(eps),
                      round(100 * rec['successes'] / len(eps), 2), rec['seconds'], Path(rec['directory']).name])
    stale = [p for p, h in manifest['runtime_hashes'].items() if not Path(p).is_file() or sha256_file(p) != h]
    if stale: problems.append(f'{len(stale)} runtime files changed since the run')
    return dict(manifest=manifest, complete=complete, audit=audit, seedck=seedck, rows=rows, tasks=tasks,
                rehashed=rehashed, runtime_files=len(manifest['runtime_hashes']), problems=problems)


def export_run(out, model, schedule, run, r):
    d = RES / run
    rel = f"{model['group']}/{schedule}"
    buf = io.StringIO(); w = csv.writer(buf, lineterminator='\n'); w.writerow(EP_FIELDS); w.writerows(r['rows'])
    put(out, f'{rel}/episodes.csv', buf.getvalue().encode(), transform='derived from COMPLETE.json (accepted attempts)')
    buf = io.StringIO(); w = csv.writer(buf, lineterminator='\n')
    w.writerow(['task', 'setting', 'successes', 'episodes', 'sr_percent', 'seconds', 'accepted_attempt'])
    w.writerows(r['tasks'])
    put(out, f'{rel}/task_results.csv', buf.getvalue().encode(), transform='derived from COMPLETE.json (accepted attempts)')
    copy(out, f'{rel}/final_audit.json', d / 'audit.json')
    copy(out, f'{rel}/manifest.json', d / 'manifest.json')
    copy_gz(out, f'{rel}/COMPLETE.json.gz', d / 'COMPLETE.json')
    copy_gz(out, f'{rel}/reference_seed_check.json.gz', d / 'campaign_seed_check.json')
    launches = sorted((json.loads(p.read_text()) for p in d.glob('launch_*.json')), key=lambda x: x['started'])
    put(out, f'{rel}/launches.json', (json.dumps(launches, indent=1) + '\n').encode(),
        transform='derived: launch_*.json of the run, sorted by start')
    a, c, m = r['audit'], r['complete'], r['manifest']
    cells = c['finished'].values()
    attempt_starts = [os.path.getmtime(p) for p in d.glob('*/attempt_*/command.json')]
    tot = {s: dict(successes=sum(x['successes'] for x in cells if x['config'] == s),
                   episodes=sum(len(x['episodes']) for x in cells if x['config'] == s)) for s in SETTINGS}
    tot['combined'] = dict(successes=sum(v['successes'] for v in tot.values()),
                           episodes=sum(v['episodes'] for v in tot.values()))
    for v in tot.values(): v['sr_percent'] = round(100 * v['successes'] / v['episodes'], 2)
    for s in ('demo_clean', 'demo_randomized', 'combined'):
        if tot[s]['successes'] != a['totals'][s]['successes'] or tot[s]['episodes'] != a['totals'][s]['episodes']:
            raise SystemExit(f'{run}: totals disagree with the audit for {s}')
    final = dict(
        host=HOST, model=model['name'], group=model['group'], schedule=schedule, label=label(schedule),
        depths=m['depths'], nfe=m['num_inference_steps'], checkpoint=model['checkpoint'], teacher=model['teacher'],
        reference_report=f"../../../{model['reference']}", protocol={k: m[k] for k in (
            'seed', 'episodes', 'instruction_type', 'action_horizon', 'replan_steps', 'noise_seed', 'image_preprocessing',
            'instruction_rng', 't5_loader', 'cuda_graph', 'renderer_shader')},
        totals=tot, reference_comparison={s: a['totals'][s] for s in ('demo_clean', 'demo_randomized', 'combined')},
        cells=len(c['finished']),
        cells_accepted_after_a_rejected_attempt=sum(not x['directory'].endswith('attempt_01') for x in cells),
        rejected_attempts=a['rejected_attempts'],
        policy_queries=sum(e['queries'] for x in cells for e in x['episodes']),
        accepted_cell_seconds_sum=round(sum(x['seconds'] for x in cells), 1),
        launches_utc=[utc(x['started']) for x in json.loads((out / rel / 'launches.json').read_text())],
        first_attempt_started_utc=utc(min(attempt_starts)), completed_utc=utc(c['updated']),
        harness_sha256=m['harness_sha256'], policy_sha256=m['policy_sha256'], stats_sha256=m['stats_sha256'],
        vae_sha256=m['vae_sha256'], runtime_files_rechecked=r['runtime_files'], videos_checked_present=len(r['rows']),
        videos_rehashed=r['rehashed'], source_dir=f'{HOST}:{d}',
        complete_json_sha256=sha256_file(d / 'COMPLETE.json'), audit_json_sha256=sha256_file(d / 'audit.json'))
    put(out, f'{rel}/final_results.json', (json.dumps(final, indent=1) + '\n').encode(),
        transform='derived from COMPLETE.json, audit.json, manifest.json, launch_*.json')
    return final


def export_verification(out):
    for f in ('e0ep10_static.json', 'e1_static.json', 'curobo_nondeterminism.json', 'renderer_preflight_gpu0.json',
              't5_lean_full_gpu.json', 'assets_vs_campaign_manifest.json'):
        copy(out, f'verification/{f}', VER / f)
    for src, dst in (('gates_e0ep10', 'gates_e0_teacher10_5ep'), ('gates_eval3', 'gates_e1')):
        for p in sorted((VER / src).glob('*.json')):
            copy(out, f'verification/{dst}/{p.name}', p)
    copy_gz(out, 'verification/runtime_manifest.json.gz', '/data/anh/robotwin_eval/runtime_manifest.json')
    for src, dst in (('smoke_e0ep10_20261111', 'smoke_e0_teacher10_5ep'), ('smoke_queue_20261107', 'smoke_e1')):
        for run in sorted((RES / src).iterdir()):
            for f in ('audit.json', 'campaign_seed_check.json', 'manifest.json'):
                if (run / f).is_file():
                    copy(out, f'verification/{dst}/{run.name}/{f}', run / f)


def export_code(out, runtime_files):
    """The evaluation code exactly as it ran: all hashed runtime files of the checkout except the four public UMT5
    tokenizer files (their hashes are in every manifest.json), plus the gate/check/audit scripts and setup scripts."""
    root = str(CODE) + '/'
    files = sorted(p[len(root):] for p in runtime_files if p.startswith(root) and '/checkpoints/' not in p)
    extra = ['scripts/elastic_robotwin_gates.py', 'scripts/verify_elastic_robotwin_export.py',
             'scripts/robotwin_smoke_check.py', 'scripts/robotwin_eval_audit.py', 'docs/robotwin_local_e1_setup.md']
    extra += sorted(str(p.relative_to(CODE)) for p in (CODE / 'scripts/robotwin_local_setup').iterdir() if p.is_file())
    names = sorted(set(files) | set(extra))
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0, filename='') as g:
        with tarfile.open(fileobj=g, mode='w', format=tarfile.PAX_FORMAT) as t:
            for n in names:
                data = (CODE / n).read_bytes()
                ti = tarfile.TarInfo(f'LoopWAM-robotwin-eval/{n}')
                ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, '', ''
                t.addfile(ti, io.BytesIO(data))
    put(out, 'setup/evaluation_code.tar.gz', buf.getvalue(),
        transform=f'deterministic tar.gz of {len(names)} files from {HOST}:{CODE}')
    copy(out, 'setup/robotwin_local_setup.md', CODE / 'docs/robotwin_local_e1_setup.md')
    return names


def main():
    out = Path(sys.argv[1]).resolve()
    results, checked, runtime = [], [], None
    for model, schedule, run in RUNS:
        r = check_run(model, schedule, run)
        print(f'{run}: {len(r["rows"])} episodes, {r["rehashed"]} videos rehashed, {r["runtime_files"]} runtime files, '
              f'problems {r["problems"] or "none"}', flush=True)
        if r['problems']:
            raise SystemExit(f'{run}: not exported: {r["problems"][:10]}')
        if runtime is None: runtime = r['manifest']['runtime_hashes']
        elif r['manifest']['runtime_hashes'] != runtime: raise SystemExit(f'{run}: runtime files differ between runs')
        checked.append((model, schedule, run, r))
    for model, schedule, run, r in checked:
        results.append(export_run(out, model, schedule, run, r))
    export_verification(out)
    names = export_code(out, runtime)
    prov = dict(
        host=HOST, checkout=str(CODE), branch='eval/robotwin-local-e1-20261107',
        head='9cffb61 (local commit; parent 9cdceaa rebuilds the campaign evaluator from 0f9e494 + archived patch)',
        uncommitted_files_at_export=sorted(l[3:] for l in os.popen(f'git -C {CODE} status --porcelain').read().splitlines()
                                           if not l[3:].startswith('third_party/RoboTwin/assets')),
        runtime_files=len(runtime), runtime_files_unchanged_since_every_run=True,
        identical_runtime_hashes_in_all_nine_manifests=True, archived_files=len(names),
        archived_file_sha256={n: sha256_file(CODE / n) for n in names})
    put(out, 'setup/code_provenance.json', (json.dumps(prov, indent=1) + '\n').encode(),
        transform='derived from the checkout and the nine manifest.json files')
    rows = sorted(sources)
    put(out, 'SOURCES.tsv', ('path\tsource\tsource_sha256\tsource_size\ttransform\n' +
                             ''.join('\t'.join(x) + '\n' for x in rows)).encode())
    print(f'exported {len(results)} runs, {len(sources)} files -> {out}')


if __name__ == '__main__':
    main()
