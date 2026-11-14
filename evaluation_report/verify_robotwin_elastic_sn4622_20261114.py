#!/usr/bin/env python3
"""Independent check of robotwin_elastic_sn4622_20261114 (nine RoboTwin-50 elastic evaluations on sn4622128794).

Uses only files in evaluation_report/ and the Python standard library, and none of the report's own tools. For each of
the nine runs it recomputes from episodes.csv:
  * coverage: 50 tasks x 2 settings x 100 episodes, unique scene seeds rising from 4,300,000 within each cell;
  * clean / randomized / overall successes, checked against final_results.json, final_audit.json, task_results.csv,
    results.json and the README tables;
  * the same-scene comparison with the teacher reference (robotwin_v0_epoch10 for E0, robotwin_v0 for E1), checked
    against the saved audit;
  * COMPLETE.json.gz: the decompressed bytes have the SHA-256 recorded for the original, and every accepted episode in it
    equals its episodes.csv row;
  * checkpoint SHA-256, schedule depths and NFE in final_results.json, final_audit.json and manifest.json.
It also checks every copied or gzipped file against SOURCES.tsv, and the archived evaluation code against
code_provenance.json and the runtime hashes of the run manifests.

    python evaluation_report/verify_robotwin_elastic_sn4622_20261114.py
"""
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

ER = Path(__file__).resolve().parent
R = ER / 'robotwin_elastic_sn4622_20261114'
CKPT = {'e0_teacher10_5ep': '07d24ff77410e516ac90d24d76d651e2102b09b0458d9d5d86ea18dff39e823e',
        'e1_epoch1': '9eba3aade1417b4425be5593c88e81441cf71bb4b6f16a8d81012ad6146e1bcb'}
REF = {'e0_teacher10_5ep': ER / 'robotwin_v0_epoch10/episodes.csv', 'e1_epoch1': ER / 'robotwin_v0/episodes.csv'}
RUNS = [('e1_epoch1', s) for s in ('flat_k1_n2', 'flat_k4_n2', 'flat_k4_n5')] + \
       [('e0_teacher10_5ep', s) for s in ('flat_k1_n5', 'flat_k1_n10', 'flat_k2_n5', 'flat_k2_n10', 'flat_k4_n5',
                                          'flat_k4_n10')]
problems = []


def check(ok, msg):
    if not ok:
        problems.append(msg)
    return ok


def sha(b):
    return hashlib.sha256(b).hexdigest()


def episodes(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault((r['task'], r['setting']), []).append(r)
    return out


def main():
    readme = (R / 'README.md').read_text()
    results = json.loads((R / 'results.json').read_text())
    for group, s in RUNS:
        d = R / group / s
        k, n = (int(x[1:]) for x in s.split('_')[1:])
        cells = episodes(d / 'episodes.csv')
        check(len(cells) == 100 and len({t for t, _ in cells}) == 50, f'{s}: cells')
        for key, rows in cells.items():
            seeds = [int(r['seed']) for r in sorted(rows, key=lambda r: int(r['episode_index']))]
            check(len(rows) == 100 and [int(r['episode_index']) for r in rows] == list(range(100)), f'{s} {key}: episodes')
            check(seeds[0] >= 4300000 and all(a < b for a, b in zip(seeds, seeds[1:])), f'{s} {key}: seeds')
            check(all(r['success'] in ('True', 'False') for r in rows), f'{s} {key}: success values')
        succ = {st: sum(r['success'] == 'True' for (t, x), rows in cells.items() if x == st for r in rows)
                for st in ('demo_clean', 'demo_randomized')}
        total = sum(succ.values())
        f = json.loads((d / 'final_results.json').read_text())
        a = json.loads((d / 'final_audit.json').read_text())
        m = json.loads((d / 'manifest.json').read_text())
        for st in ('demo_clean', 'demo_randomized'):
            check(f['totals'][st]['successes'] == succ[st] == a['totals'][st]['successes'], f'{s}: {st} totals')
        check(f['totals']['combined']['successes'] == total == a['totals']['combined']['successes'], f'{s}: totals')
        check(a['passed'] and not a['problems'], f'{s}: audit')
        for name, rec in (('final_results', f['checkpoint']), ('audit', a), ('manifest', m)):
            check(rec['sha256' if name == 'final_results' else 'checkpoint_sha256'] == CKPT[group], f'{s}: {name} sha')
        check(m['depths'] == [k] * n and m['num_inference_steps'] == n and a['depths'] == [k] * n, f'{s}: depths/nfe')
        check((m['seed'], m['episodes'], m['action_horizon'], m['replan_steps']) == (42, 100, 32, 24), f'{s}: protocol')
        tr = list(csv.DictReader(open(d / 'task_results.csv')))
        check(len(tr) == 100 and all(int(r['successes']) == sum(e['success'] == 'True' for e in cells[(r['task'], r['setting'])])
                                     for r in tr), f'{s}: task_results.csv')
        ref = episodes(REF[group])
        n_p = s_a = s_b = 0
        for key, rows in cells.items():
            mine = {r['seed']: r['success'] == 'True' for r in rows}
            theirs = {r['seed']: r['success'] == 'True' for r in ref[key]}
            common = set(mine) & set(theirs)
            n_p += len(common); s_a += sum(mine[x] for x in common); s_b += sum(theirs[x] for x in common)
        at = a['totals']['combined']
        check(n_p == at['paired_episodes'] and round(100 * s_a / n_p, 2) == at['paired_e1_sr_percent']
              and round(100 * s_b / n_p, 2) == at['paired_v0_sr_percent'], f'{s}: same-scene comparison')
        raw = gzip.decompress((d / 'COMPLETE.json.gz').read_bytes())
        check(sha(raw) == f['complete_json_sha256'], f'{s}: COMPLETE.json digest')
        comp = json.loads(raw)
        check(comp['checkpoint_sha256'] == CKPT[group] and comp['schedule'] == s and len(comp['finished']) == 100,
              f'{s}: COMPLETE identity')
        for key, rec in comp['finished'].items():
            rows = sorted(cells[tuple(key.split('/'))], key=lambda r: int(r['episode_index']))
            check(rec['successes'] == sum(r['success'] == 'True' for r in rows), f'{s} {key}: COMPLETE successes')
            for e, r in zip(rec['episodes'], rows):
                if not check(str(e['seed']) == r['seed'] and str(e['success']) == r['success'] and
                             e['video_sha256'] == r['video_sha256'] and e['query_trace_sha256'] == r['query_trace_sha256'],
                             f'{s} {key}: COMPLETE episode differs'):
                    break
        res = results[group][s]
        check(res['successes'] == total and res['clean'] == succ['demo_clean'], f'{s}: results.json')
        lab = f'[{k},{k}]' if n == 2 else f'[{k}]×{n}'
        row = [l for l in readme.splitlines() if l.startswith(f'| `{lab}`') and ('Successes' not in l)]
        sr = f'{total / 100:.2f}%'
        check(any(f'**{sr}**' in l for l in row), f'{s}: README row for {lab} lacks {sr}')
        print(f'{group}/{s}: {total}/10000 = {sr} (clean {succ["demo_clean"]}, randomized {succ["demo_randomized"]}); '
              f'same scenes {100 * s_a / n_p:.2f}% vs {100 * s_b / n_p:.2f}% on {n_p}')
    # files copied from sn4622128794
    for row in csv.DictReader(open(R / 'SOURCES.tsv'), delimiter='\t'):
        p = R / row['path']
        if not check(p.is_file(), f'missing {row["path"]}'):
            continue
        if row['transform'] == 'copy':
            check(sha(p.read_bytes()) == row['source_sha256'], f'{row["path"]}: differs from its source')
        elif row['transform'].startswith('gzip'):
            check(sha(gzip.decompress(p.read_bytes())) == row['source_sha256'], f'{row["path"]}: differs from its source')
    # archived evaluation code
    prov = json.loads((R / 'setup/code_provenance.json').read_text())
    runtime = json.loads((R / 'e0_teacher10_5ep/flat_k4_n10/manifest.json').read_text())['runtime_hashes']
    root = '/data/anh/code/LoopWAM-robotwin-eval/'
    with tarfile.open(R / 'setup/evaluation_code.tar.gz') as t:
        members = {mm.name[len('LoopWAM-robotwin-eval/'):]: sha(t.extractfile(mm).read()) for mm in t.getmembers()}
    check(members == prov['archived_file_sha256'], 'evaluation_code.tar.gz differs from code_provenance.json')
    bound = {p[len(root):]: h for p, h in runtime.items() if p.startswith(root) and '/checkpoints/' not in p}
    check(all(members.get(p) == h for p, h in bound.items()), 'archived code differs from the run manifests')
    for group, s in RUNS:
        mm = json.loads((R / group / s / 'manifest.json').read_text())['runtime_hashes']
        check(mm == runtime, f'{s}: runtime hashes differ from the other runs')
    print(f'archived code: {len(members)} files; {len(bound)} runtime files match all nine manifests')
    if problems:
        print(f'FAILED: {len(problems)} problem(s)'); print('\n'.join(problems[:40])); sys.exit(1)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
