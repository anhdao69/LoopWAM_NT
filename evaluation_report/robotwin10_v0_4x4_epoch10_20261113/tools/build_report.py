#!/usr/bin/env python3
"""Build evaluation_report/robotwin10_v0_4x4_epoch10_20261113 from the H100 evaluation records.

Sources (H100 cluster, read-only):
  epoch 10  /groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_elastic_eval/cells/v0-video4-action4-e10/
  epoch 5   /groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_eval/cells/v0-video4-action4/   (same harness)
  training  /groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_e10/v0-video4-action4/
            /groups/yshang/an221229/checkpoints/LoopWAM_NT/robotwin10_e10_work/parents.json
Each cell directory holds result.json (accepted scene seeds and outcomes written by the eval agent) and attempt_NN/
(command.json, eval.log from RoboTwin script/eval_policy.py, _result_clean.txt, one video per accepted episode).
The builder re-derives every accepted episode from eval.log, checks it against result.json, hashes the videos, copies
the per-cell evidence (eval.log gzipped; videos are not copied) and writes the tables. Repo references for the same
epoch-5 checkpoint are read from the sibling report folders of this checkout.

python tools/build_report.py [--out DIR]
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path

ROOT = Path('/groups/yshang/an221229/checkpoints/LoopWAM_NT')
E10_CELLS = ROOT / 'robotwin10_elastic_eval/cells/v0-video4-action4-e10'
E05_CELLS = ROOT / 'robotwin10_eval/cells/v0-video4-action4'
E10_TRAIN = ROOT / 'robotwin10_e10/v0-video4-action4'
PARENTS = ROOT / 'robotwin10_e10_work/parents.json'
HARNESS = Path('/groups/yshang/an221229/code/RoboTwin_eval/RoboTwin')
POLICY_CODE = Path('/lustre/fs1/home/an221229/code/LoopWAM_rt10_eval')
TASKS = ['adjust_bottle', 'grab_roller', 'hanging_mug', 'move_stapler_pad', 'open_microwave', 'press_stapler',
         'scan_object', 'stack_blocks_two', 'stamp_seal', 'turn_switch']
SETTINGS = ['demo_clean', 'demo_randomized']
E10_SHA = 'b85ad555fa60f5aab61a35231f26d6ca9c01e32d6111d6893abb815db5b985b8'
E05_SHA = 'bb1563f16bcd102b3f4ad4a1a14377ee49de7e461d862b7bbc5dba497f036127'
REPO_E05 = [('robotwin10_v0_4x4', 'original release, local evaluation'),
            ('robotwin10_v0_4x4_remote189', 'rerun on server189'),
            ('robotwin10_v0_4x4_remote137_20261105', 'rerun on server137')]
RECIPE = ('epochs 6-10 continuation of the epoch-5 release (zero-based epochs 5-9): AdamW state continued from '
          'update 45,445, LR 3e-5 for backbone and robot I/O, no warmup, cosine to 1% over 45,445 updates, global batch 128 '
          'on 4 H100 GPUs')
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def parse_log(path):
    """Accepted episodes from an eval_policy.py log: the instruction line, then Success!/Fail!, then the running
    'Success rate: k/n ... current seed: S' line that closes the episode."""
    eps, cur, outcome = [], None, None
    for raw in Path(path).read_text(errors='replace').splitlines():
        line = ANSI.sub('', raw).strip()
        if line.startswith('{"loopwam_episode"'):
            cur, outcome = json.loads(line), None
        elif line in ('Success!', 'Fail!'):
            outcome = line == 'Success!'
        elif line.startswith('Success rate:') and 'current seed:' in line:
            k, n = map(int, re.search(r'Success rate:\s*(\d+)/(\d+)', line).groups())
            seed = int(line.rsplit('current seed:', 1)[1].strip())
            eps.append(dict(seed=seed, success=outcome, running=(k, n),
                            instruction=cur['instruction'] if cur else None,
                            log_episode=cur['loopwam_episode'] if cur else None,
                            noise_seed=cur.get('noise_seed') if cur else None))
            cur, outcome = None, None
    return eps


def load_cell(cells, task, setting, expect_sha, out_evidence, hash_videos):
    cdir = cells / f'{task}__{setting}'
    res = json.loads((cdir / 'result.json').read_text())
    assert res['task'] == task and res['setting'] == setting and res['episodes'] == 100, cdir
    assert res['checkpoint_sha256'] == expect_sha, (cdir, res['checkpoint_sha256'])
    assert len(res['seeds']) == len(res['success']) == 100 and sum(res['success']) == res['successes'], cdir
    attempts = sorted(cdir.glob('attempt_*'))
    logged, ev = [], out_evidence / f'{task}__{setting}'
    ev.mkdir(parents=True, exist_ok=True)
    (ev / 'result.json').write_bytes((cdir / 'result.json').read_bytes())
    sources = [(cdir / 'result.json', ev / 'result.json')]
    for a in attempts:
        eps = parse_log(a / 'eval.log')
        for e in eps:
            e['attempt'] = a.name
        logged += eps
        dst = ev / a.name; dst.mkdir(exist_ok=True)
        for name in ('command.json', '_result_clean.txt'):
            if (a / name).exists():
                (dst / name).write_bytes((a / name).read_bytes()); sources.append((a / name, dst / name))
        with open(a / 'eval.log', 'rb') as f, gzip.GzipFile(dst / 'eval.log.gz', 'wb', mtime=0) as g:
            shutil.copyfileobj(f, g)
        sources.append((a / 'eval.log', dst / 'eval.log.gz'))
    # the scored episodes are the first 100 logged episodes in seed order (later attempts resume at seed + 1)
    by_seed = {}
    for e in logged:
        assert e['seed'] not in by_seed, (cdir, 'seed logged twice', e['seed'])
        by_seed[e['seed']] = e
    rows = []
    for i, (seed, ok) in enumerate(zip(res['seeds'], res['success'])):
        e = by_seed.get(seed)
        assert e is not None, (cdir, 'scored seed missing from logs', seed)
        assert e['success'] is bool(ok), (cdir, seed, e['success'], ok)
        assert e['noise_seed'] == 42, (cdir, seed)
        video, vsha = '', ''
        if hash_videos:
            v = sorted((cdir / e['attempt']).glob(f'episode{e["log_episode"]}_*.mp4'))
            if v:
                assert len(v) == 1 and ('success-true' in v[0].name) == bool(ok), (cdir, seed, v)
                video, vsha = f'{e["attempt"]}/{v[0].name}', sha256(v[0])
        rows.append(dict(task=task, setting=setting, episode_index=i, seed=seed, success=bool(ok),
                         instruction=e['instruction'], attempt=e['attempt'], video=video, video_sha256=vsha))
    assert sorted(res['seeds']) == res['seeds'] and len(set(res['seeds'])) == 100, cdir
    cmd0 = json.loads((attempts[0] / 'command.json').read_text())
    seconds = res['completed'] - cmd0['started']
    return res, rows, sources, dict(attempts=len(attempts), seconds=seconds, job=res.get('last_job'),
                                    host=res.get('last_host'), logged_episodes=len(logged))


def repo_reference(er, folder):
    rows = list(csv.DictReader(open(er / folder / 'task_results.csv')))
    out = {(r['task'], r['setting']): int(r['successes']) for r in rows}
    audit = json.loads((er / folder / 'final_audit.json').read_text())
    return out, audit.get('checkpoint_sha256') == E05_SHA and audit.get('status') == 'passed'


def main():
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(here)); a = ap.parse_args()
    out = Path(a.out); er = out.parent
    evidence = out / 'evidence'
    if evidence.exists():
        shutil.rmtree(evidence)
    sources, cells = [], {}
    for label, root, sha, vids in (('epoch10', E10_CELLS, E10_SHA, True), ('epoch05_same_harness', E05_CELLS, E05_SHA, False)):
        for t in TASKS:
            for s in SETTINGS:
                res, rows, src, meta = load_cell(root, t, s, sha, evidence / label, vids)
                cells[(label, t, s)] = dict(res=res, rows=rows, meta=meta)
                sources += src
    # training and publication records of the epoch-10 checkpoint
    tr = evidence / 'training'; tr.mkdir(parents=True, exist_ok=True)
    for src, name in ((E10_TRAIN / 'train/manifest.json', 'manifest.json'), (E10_TRAIN / 'train/timing.json', 'timing.json'),
                      (E10_TRAIN / 'train/precision.json', 'precision.json'),
                      (E10_TRAIN / 'train/resume_audit_u45445.json', 'resume_audit_u45445.json'),
                      (E10_TRAIN / 'upload_verification_epoch_10.json', 'hf_upload_verification_epoch_10.json')):
        (tr / name).write_bytes(src.read_bytes()); sources.append((src, tr / name))
    parent = json.loads(PARENTS.read_text())['v0-video4-action4']
    (tr / 'parent_verification.json').write_text(json.dumps(parent, indent=1) + '\n')
    sources.append((PARENTS, tr / 'parent_verification.json'))
    metrics = [json.loads(l) for l in (E10_TRAIN / 'train/metrics.jsonl').read_text().splitlines() if l.strip()]
    metrics = [m for m in metrics if 'update' in m and 'loss' in m]
    by_epoch = {}
    for m in metrics:
        by_epoch.setdefault(m.get('epoch'), []).append(m['loss'])
    man = json.loads((E10_TRAIN / 'train/manifest.json').read_text())
    timing = json.loads((E10_TRAIN / 'train/timing.json').read_text())
    upload = json.loads((E10_TRAIN / 'upload_verification_epoch_10.json').read_text())
    assert upload['policy_sha256'] == E10_SHA and parent['epoch_05_sha256'] == E05_SHA and parent['tensors_equal_epoch_05']
    assert timing['status'] == 'complete' and timing['update'] == 90890

    # episode tables
    fields = ['task', 'setting', 'episode_index', 'seed', 'success', 'instruction', 'attempt', 'video', 'video_sha256']
    for label, fname in (('epoch10', 'episodes.csv'), ('epoch05_same_harness', 'reference_epoch05_episodes.csv')):
        with open(out / fname, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields if label == 'epoch10' else fields[:7]); w.writeheader()
            for t in TASKS:
                for s in SETTINGS:
                    for r in cells[(label, t, s)]['rows']:
                        w.writerow({k: r[k] for k in w.fieldnames})

    # per task and paired statistics
    refs = {}
    for folder, desc in REPO_E05:
        counts, sha_ok = repo_reference(er, folder)
        refs[folder] = dict(description=desc, counts=counts, checkpoint_sha_matches=sha_ok)
    task_rows, paired, instr_same, instr_total = [], {}, 0, 0
    for t in TASKS:
        for s in SETTINGS:
            a = {r['seed']: r for r in cells[('epoch10', t, s)]['rows']}
            b = {r['seed']: r for r in cells[('epoch05_same_harness', t, s)]['rows']}
            shared = sorted(set(a) & set(b))
            ao = sum(a[k]['success'] and not b[k]['success'] for k in shared)
            bo = sum(b[k]['success'] and not a[k]['success'] for k in shared)
            instr_same += sum(a[k]['instruction'] == b[k]['instruction'] for k in shared); instr_total += len(shared)
            paired[(t, s)] = (len(shared), ao, bo)
            k10 = cells[('epoch10', t, s)]['res']['successes']; k05 = cells[('epoch05_same_harness', t, s)]['res']['successes']
            task_rows.append(dict(task=t, setting=s, epoch10_successes=k10, epoch10_episodes=100, epoch10_sr_percent=k10,
                                  epoch05_same_harness_successes=k05, epoch05_same_harness_sr_percent=k05,
                                  difference_pp=k10 - k05, shared_seeds=len(shared), epoch10_only=ao, epoch05_only=bo,
                                  mcnemar_p=round(mcnemar(ao, bo), 6),
                                  **{f'repo_{f}_successes': refs[f]['counts'][(t, s)] for f, _ in REPO_E05},
                                  epoch10_cell_hours=round(cells[('epoch10', t, s)]['meta']['seconds'] / 3600, 3)))
    with open(out / 'task_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(task_rows[0])); w.writeheader(); w.writerows(task_rows)

    def agg(label, setting=None):
        rs = [r for (l, t, s), c in cells.items() if l == label and (setting is None or s == setting) for r in c['rows']]
        k = sum(r['success'] for r in rs); return k, len(rs)

    summary = {}
    for name, st in (('clean', 'demo_clean'), ('randomized', 'demo_randomized'), ('combined', None)):
        k10, n10 = agg('epoch10', st); k05, n05 = agg('epoch05_same_harness', st)
        keys = [(t, s) for t in TASKS for s in SETTINGS if st is None or s == st]
        sh = sum(paired[k][0] for k in keys); ao = sum(paired[k][1] for k in keys); bo = sum(paired[k][2] for k in keys)
        summary[name] = dict(epoch10=dict(successes=k10, episodes=n10, sr_percent=round(100 * k10 / n10, 2),
                                          wilson95=[round(x, 2) for x in wilson(k10, n10)]),
                             epoch05_same_harness=dict(successes=k05, episodes=n05, sr_percent=round(100 * k05 / n05, 2),
                                                       wilson95=[round(x, 2) for x in wilson(k05, n05)]),
                             paired=dict(shared_seeds=sh, epoch10_only=ao, epoch05_only=bo,
                                         difference_pp=round(100 * (ao - bo) / sh, 2), mcnemar_p=mcnemar(ao, bo)))
        for f, _ in REPO_E05:
            kk = sum(refs[f]['counts'][k] for k in keys)
            summary[name][f'repo_{f}'] = dict(successes=kk, episodes=100 * len(keys), sr_percent=round(kk / len(keys), 2))
    def window(label):
        st = [json.loads((c / 'command.json').read_text())['started'] for (l, t, s), x in cells.items() if l == label
              for c in sorted(((E10_CELLS if label == 'epoch10' else E05_CELLS) / f'{t}__{s}').glob('attempt_*'))]
        en = [x['res']['completed'] for (l, _, _), x in cells.items() if l == label]
        f = lambda u: time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(u))
        return f(min(st)), f(max(en)), len(st)
    e05_window = window('epoch05_same_harness'); e10_window = window('epoch10')
    hours10 = sum(c['meta']['seconds'] for (l, _, _), c in cells.items() if l == 'epoch10') / 3600
    e10_meta = [c['meta'] for (l, _, _), c in cells.items() if l == 'epoch10']
    results = dict(
        report='robotwin10_v0_4x4_epoch10_20261113',
        model='LoopWAM v0-video4-action4 (video/action loops 4/4), RoboTwin-10, epoch 10 (update 90,890)',
        checkpoint=dict(sha256=E10_SHA, step=90890, hf_repo=upload['repo'], hf_path=upload['path'],
                        hf_revision=upload['revision'], parent_epoch05_sha256=E05_SHA),
        protocol=cells[('epoch10', TASKS[0], SETTINGS[0])]['res']['protocol'],
        evaluation=dict(hardware='NVIDIA H100 80GB HBM3 (Slurm job ' + ','.join(sorted({m['job'] for m in e10_meta})) +
                        ', node ' + ','.join(sorted({m['host'] for m in e10_meta})) + ')',
                        completed=time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(max(c['res']['completed'] for (l, _, _), c in cells.items() if l == 'epoch10'))),
                        cell_hours_sum=round(hours10, 2), attempts_per_cell=sorted({m['attempts'] for m in e10_meta}),
                        episodes_logged=sum(m['logged_episodes'] for m in e10_meta)),
        epoch05_same_harness_evaluation=dict(started=e05_window[0], completed=e05_window[1], attempts=e05_window[2],
                                             note='interrupted cells resumed at the next scene seed; no seed is scored twice'),
        epoch10_window=dict(started=e10_window[0], completed=e10_window[1], attempts=e10_window[2]),
        summary=summary,
        instruction_agreement_on_shared_seeds=dict(same=instr_same, shared=instr_total),
        repo_references={f: dict(description=d, checkpoint_sha256_matches_epoch05=refs[f]['checkpoint_sha_matches'])
                         for f, d in REPO_E05},
        training=dict(recipe=RECIPE,
                      git_revision=man.get('git_revision'), git_dirty=man.get('git_dirty'), world_size=man.get('world_size'),
                      global_batch=man.get('global_batch'), microbatch=man.get('microbatch'),
                      lr_backbone=man.get('lr_backbone'), lr_robot_io=man.get('lr_robot_io'),
                      updates=[45446, timing['update']], windows_seen=timing['windows_seen'],
                      training_hours=round(timing['elapsed_training_seconds'] / 3600, 2),
                      mean_loss_by_epoch={str(k): round(sum(v) / len(v), 5) for k, v in sorted(by_epoch.items(), key=lambda x: (x[0] is None, x[0]))}),
        tasks=task_rows)
    (out / 'results.json').write_text(json.dumps(results, indent=1) + '\n')

    # provenance of the evaluation code
    code = {}
    for p in [HARNESS / 'script/eval_policy.py', HARNESS / 'policy/loopwam_policy/deploy_policy.py',
              HARNESS / 'policy/loopwam_policy/deploy_policy.yml']:
        code[str(p)] = sha256(p)
    code['policy/loopwam_policy -> '] = str((HARNESS / 'policy/loopwam_policy').resolve())
    mtimes = {str(p): time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(p.stat().st_mtime)) for p in
              [HARNESS / 'script/eval_policy.py', (HARNESS / 'policy/loopwam_policy').resolve() / 'deploy_policy.py',
               (HARNESS / 'policy/loopwam_policy').resolve() / 'deploy_policy.yml']}
    prov = dict(epoch10_eval_root=str(E10_CELLS), epoch05_eval_root=str(E05_CELLS), harness_root=str(HARNESS),
                harness_files_sha256=code, harness_files_last_modified=mtimes,
                harness_note='last modified before the first epoch-5 attempt started (' + e05_window[0] + '), so both runs used these files', policy_code_repo=str(POLICY_CODE),
                policy_code_git_revision='c8db891a773ae1ad8e81e2aa7b3a965d1c4b2a6e',
                policy_code_note='LoopWAM_rt10_eval worktree, detached at c8db891 (branch robotwin10_training_20261101); '
                                 'experiments/ and src/ unmodified; only the untracked queue scripts scripts/robotwin10/'
                                 'eval_agent.py, eval_dispatcher.py, eval_report.py were added',
                data_dir=cells[('epoch10', TASKS[0], SETTINGS[0])]['rows'] and json.loads(
                    (E10_CELLS / f'{TASKS[0]}__{SETTINGS[0]}/attempt_01/command.json').read_text())['cmd'][
                    json.loads((E10_CELLS / f'{TASKS[0]}__{SETTINGS[0]}/attempt_01/command.json').read_text())['cmd'].index('--data_dir') + 1],
                videos='one video per accepted epoch-10 episode stays on the cluster; names and SHA-256 are in episodes.csv')
    (out / 'provenance.json').write_text(json.dumps(prov, indent=1) + '\n')
    with open(out / 'SOURCES.tsv', 'w') as f:
        f.write('source\tcopied_to\tsource_sha256\n')
        for src, dst in sources:
            f.write(f'{src}\t{dst.relative_to(out)}\t{sha256(src)}\n')
    print(json.dumps(summary, indent=1))
    print('instructions identical on shared seeds:', instr_same, '/', instr_total)
    for f, d in refs.items():
        print(f, 'sha matches epoch 5:', d['checkpoint_sha_matches'])


if __name__ == '__main__':
    main()
