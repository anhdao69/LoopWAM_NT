#!/usr/bin/env python3
"""Build results.json, task_comparison.csv and README.md of this report from the exported records only.

Inputs (all inside evaluation_report/): the nine exported runs of this folder (<group>/<schedule>/episodes.csv,
final_results.json, final_audit.json), the v0 teacher references robotwin_v0_epoch10/ and robotwin_v0/, and, for
context, the NFE-2 results of the same E0 checkpoint (robotwin_elastic_e0_teacher10_5ep_20261111/,
robotwin_elastic_e0_teacher10_5ep_44_20261112/) and the epoch-1 E0 results (robotwin_elastic_e0_20261107/).
README.md is tools/README.template.md with every <!-- TABLE:name --> marker replaced by a generated table.

    python tools/build_report.py        (run from anywhere; writes into this report folder)
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ER = HERE.parent
E0_RUNS = ['flat_k1_n5', 'flat_k1_n10', 'flat_k2_n5', 'flat_k2_n10', 'flat_k4_n5', 'flat_k4_n10']
E1_RUNS = ['flat_k1_n2', 'flat_k4_n2', 'flat_k4_n5']
REF_E0 = 'robotwin_v0_epoch10/episodes.csv'          # v0 epoch-10 teacher of E0-epoch10-teacher-5ep
REF_E1 = 'robotwin_v0/episodes.csv'                  # v0 epoch-5 teacher of E1
E0_NFE2 = {'flat_k1_n2': 'robotwin_elastic_e0_teacher10_5ep_20261111/local/flat_k1_n2/episodes.csv',
           'flat_k2_n2': 'robotwin_elastic_e0_teacher10_5ep_20261111/remote137/flat_k2_n2/episodes.csv',
           'flat_k4_n2': 'robotwin_elastic_e0_teacher10_5ep_44_20261112/flat_k4_n2/episodes.csv'}
E0_EPOCH1 = {'flat_k1_n2': 'robotwin_elastic_e0_20261107/flat_k1_n2/episodes.csv',
             'flat_k4_n2': 'robotwin_elastic_e0_20261107/parallel137_20261108/remote_mirror/flat_k4_n2/episodes.csv',
             'flat_k4_n5': 'robotwin_elastic_e0_20261107/flat_k4_n5/episodes.csv'}
SETTINGS = ('demo_clean', 'demo_randomized')


def label(s):
    k, n = (int(x[1:]) for x in s.split('_')[1:])
    return f'[{k},{k}]' if n == 2 else f'[{k}]×{n}'


def nfe(s):
    return int(s.split('_n')[1])


def load(path):
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 10000, path
    cells = defaultdict(dict)
    for r in rows:
        assert r['success'] in ('True', 'False'), path
        cells[(r['task'], r['setting'])][int(r['seed'])] = r['success'] == 'True'
    assert len(cells) == 100 and all(len(v) == 100 for v in cells.values()), path
    return cells


def stats(cells):
    s = {st: sum(sum(v.values()) for (t, x), v in cells.items() if x == st) for st in SETTINGS}
    task = {}
    for (t, x), v in cells.items():
        task.setdefault(t, {})[x] = sum(v.values())
    return dict(clean=s['demo_clean'], randomized=s['demo_randomized'], successes=sum(s.values()), episodes=10000,
                sr=sum(s.values()) / 100, clean_sr=s['demo_clean'] / 50, randomized_sr=s['demo_randomized'] / 50,
                tasks={t: dict(clean=v['demo_clean'], randomized=v['demo_randomized'],
                               sr=(v['demo_clean'] + v['demo_randomized']) / 2) for t, v in sorted(task.items())})


def paired(a, b):
    """Successes of a and b on the scene seeds both accepted, cell by cell."""
    n = sa = sb = 0
    for k in a:
        common = set(a[k]) & set(b[k])
        n += len(common); sa += sum(a[k][x] for x in common); sb += sum(b[k][x] for x in common)
    return dict(episodes=n, a_successes=sa, b_successes=sb, a_sr=100 * sa / n, b_sr=100 * sb / n)


def task_counts(a, b):
    up = sum(a['tasks'][t]['sr'] > b['tasks'][t]['sr'] for t in a['tasks'])
    tie = sum(a['tasks'][t]['sr'] == b['tasks'][t]['sr'] for t in a['tasks'])
    return up, tie, 50 - up - tie


def pct(x):
    return f'{x:.2f}%'


def pp(x):
    return f'{x:+.2f} pp'


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join('---' for _ in header) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def main():
    runs, cells = {}, {}
    for group, names, ref in (('e0_teacher10_5ep', E0_RUNS, REF_E0), ('e1_epoch1', E1_RUNS, REF_E1)):
        for s in names:
            d = HERE / group / s
            c = load(d / 'episodes.csv'); f = json.loads((d / 'final_results.json').read_text())
            a = json.loads((d / 'final_audit.json').read_text())
            st = stats(c)
            assert a['passed'] and not a['problems'] and f['schedule'] == s == a['schedule']
            assert st['successes'] == f['totals']['combined']['successes'] == a['totals']['combined']['successes']
            assert st['clean'] == a['totals']['demo_clean']['successes']
            assert st['randomized'] == a['totals']['demo_randomized']['successes']
            pr = paired(c, load(ER / ref))
            at = a['totals']['combined']
            assert pr['episodes'] == at['paired_episodes'], (s, pr, at)
            assert round(pr['a_sr'], 2) == at['paired_e1_sr_percent'] and round(pr['b_sr'], 2) == at['paired_v0_sr_percent']
            runs[(group, s)] = dict(stats=st, paired=pr, final=f, audit=a)
            cells[(group, s)] = c
    t10, t5 = load(ER / REF_E0), load(ER / REF_E1)
    T10, T5 = stats(t10), stats(t5)
    nfe2 = {s: stats(load(ER / p)) for s, p in E0_NFE2.items()}
    ep1 = {s: stats(load(ER / p)) for s, p in E0_EPOCH1.items()}
    e0 = {s: runs[('e0_teacher10_5ep', s)] for s in E0_RUNS}
    e1 = {s: runs[('e1_epoch1', s)] for s in E1_RUNS}
    T = {}

    rows = []
    for s in E0_RUNS:
        r = e0[s]; st, p = r['stats'], r['paired']
        rows.append([f'`{label(s)}`', s.split('_n')[0][-2:].replace('k', ''), nfe(s), pct(st['clean_sr']),
                     pct(st['randomized_sr']), f"**{pct(st['sr'])}**", f"{st['successes']:,}",
                     f"{pct(p['a_sr'])} vs {pct(p['b_sr'])} ({p['episodes']:,})", 'passed, 100/100 cells'])
    rows.append(['v0 epoch-10 teacher', 4, 10, pct(T10['clean_sr']), pct(T10['randomized_sr']), pct(T10['sr']),
                 f"{T10['successes']:,}", '—', '[robotwin_v0_epoch10](../robotwin_v0_epoch10/report.md)'])
    T['e0_main'] = table(['Schedule', 'Action loops', 'NFE', 'Clean', 'Randomized', 'Overall', 'Successes / 10,000',
                          'Same scenes: student vs teacher (episodes)', 'Final audit'], rows)

    def cell(st):
        return f"{pct(st['sr'])}"
    grid = []
    for k in (1, 2, 4):
        grid.append([f'{k}', f"{cell(nfe2[f'flat_k{k}_n2'])} ¹", f"**{cell(e0[f'flat_k{k}_n5']['stats'])}**",
                     f"**{cell(e0[f'flat_k{k}_n10']['stats'])}**"])
    T['e0_grid'] = table(['Action loops per step', 'NFE 2 (`[k,k]`)', 'NFE 5 (`[k]×5`)', 'NFE 10 (`[k]×10`)'], grid)

    def delta(name, a, b):
        up, tie, down = task_counts(a, b)
        return [name, pp(a['clean_sr'] - b['clean_sr']), pp(a['randomized_sr'] - b['randomized_sr']),
                f"**{pp(a['sr'] - b['sr'])}**", f"{a['successes'] - b['successes']:+d}", f'{up} / {tie} / {down}']
    st = {s: e0[s]['stats'] for s in E0_RUNS}
    drows = [delta(f'`[{k}]×10` − `[{k}]×5`', st[f'flat_k{k}_n10'], st[f'flat_k{k}_n5']) for k in (1, 2, 4)]
    for n in (5, 10):
        drows += [delta(f'`[2]×{n}` − `[1]×{n}`', st[f'flat_k2_n{n}'], st[f'flat_k1_n{n}']),
                  delta(f'`[4]×{n}` − `[2]×{n}`', st[f'flat_k4_n{n}'], st[f'flat_k2_n{n}']),
                  delta(f'`[4]×{n}` − `[1]×{n}`', st[f'flat_k4_n{n}'], st[f'flat_k1_n{n}'])]
    T['e0_deltas'] = table(['Comparison', 'Clean', 'Randomized', 'Overall', 'Successes / 10,000',
                            'Tasks better / tied / worse'], drows)

    trows = []
    for s in E0_RUNS:
        a, p = st[s], e0[s]['paired']
        up, tie, down = task_counts(a, T10)
        trows.append([f'`{label(s)}`', pp(a['sr'] - T10['sr']), pp(p['a_sr'] - p['b_sr']),
                      f"{p['a_successes']:,} vs {p['b_successes']:,} of {p['episodes']:,}", f'{up} / {tie} / {down}'])
    T['e0_teacher'] = table(['Student schedule', 'Overall minus teacher', 'Same scenes: minus teacher',
                             'Same-scene successes, student vs teacher', 'Tasks above / tied / below teacher'], trows)

    gap = sorted(((st['flat_k4_n5']['tasks'][t]['sr'] - T10['tasks'][t]['sr'], t) for t in T10['tasks']))
    T['e0_task_gaps'] = table(['Task', '`[4]×5` SR', '`[4]×10` SR', 'Teacher SR', '`[4]×5` − teacher'],
                              [[t, pct(st['flat_k4_n5']['tasks'][t]['sr']), pct(st['flat_k4_n10']['tasks'][t]['sr']),
                                pct(T10['tasks'][t]['sr']), pp(g)] for g, t in gap[:6] + gap[-4:]])

    hdr = ['Task'] + [f'`{label(s)}`' for s in E0_RUNS] + ['Teacher']
    T['e0_tasks'] = table(hdr, [[t] + [f"{st[s]['tasks'][t]['clean']}/{st[s]['tasks'][t]['randomized']} · "
                                       f"{st[s]['tasks'][t]['sr']:.1f}" for s in E0_RUNS] +
                                [f"{T10['tasks'][t]['clean']}/{T10['tasks'][t]['randomized']} · {T10['tasks'][t]['sr']:.1f}"]
                                for t in T10['tasks']])

    e1rows = []
    for s in E1_RUNS:
        a, p = e1[s]['stats'], e1[s]['paired']
        e1rows.append([f'`{label(s)}`', nfe(s), pct(a['clean_sr']), pct(a['randomized_sr']), f"**{pct(a['sr'])}**",
                       f"{pct(p['a_sr'])} vs {pct(p['b_sr'])} ({p['episodes']:,})", pct(ep1[s]['sr']),
                       pp(a['sr'] - ep1[s]['sr'])])
    e1rows.append(['v0 epoch-5 teacher', 10, pct(T5['clean_sr']), pct(T5['randomized_sr']), pct(T5['sr']), '—', '—', '—'])
    T['e1_main'] = table(['E1 schedule', 'NFE', 'Clean', 'Randomized', 'Overall', 'Same scenes: E1 vs teacher (episodes)',
                          'E0 epoch 1, same schedule ²', 'E1 − E0 epoch 1'], e1rows)

    rr = []
    for g, s in [('e1_epoch1', s) for s in E1_RUNS] + [('e0_teacher10_5ep', s) for s in E0_RUNS]:
        f = runs[(g, s)]['final']
        reasons = defaultdict(int)
        for z in f['rejected_attempts']:
            e = z['failure'].get('error') if isinstance(z['failure'], dict) else z['failure']
            e = ('out-of-memory kill (exit −9)' if e == 'Worker exit-9' else
                 'stopped by restart' if 'interrupted' in e else
                 'no-progress watchdog (exit −15)' if e == 'Worker exit-15' else
                 'worker exit 1' if e == 'Worker exit1' else 'render/GPU-memory fault' if 'Render' in e else e)
            reasons[e] += 1
        rr.append([('E1 ' if g == 'e1_epoch1' else 'E0 ') + f'`{label(s)}`', f['first_attempt_started_utc'][5:16],
                   f['completed_utc'][5:16], f"{f['accepted_cell_seconds_sum'] / 3600:.1f}", f"{f['policy_queries']:,}",
                   len(f['rejected_attempts']), ', '.join(f'{v} {k}' for k, v in sorted(reasons.items())) or '—',
                   f['cells_accepted_after_a_rejected_attempt']])
    T['runs'] = table(['Evaluation', 'First attempt (UTC)', 'Complete (UTC)', 'Accepted cell-hours', 'Policy queries',
                       'Rejected attempts', 'Reasons', 'Cells accepted on a retry'], rr)

    results = dict(
        report='robotwin_elastic_sn4622_20261114', host='sn4622128794',
        e0_teacher10_5ep={s: dict(label=label(s), nfe=nfe(s), **{k: v for k, v in e0[s]['stats'].items() if k != 'tasks'},
                                  paired_vs_teacher=e0[s]['paired'], tasks=e0[s]['stats']['tasks']) for s in E0_RUNS},
        e1_epoch1={s: dict(label=label(s), nfe=nfe(s), **{k: v for k, v in e1[s]['stats'].items() if k != 'tasks'},
                           paired_vs_teacher=e1[s]['paired'], tasks=e1[s]['stats']['tasks']) for s in E1_RUNS},
        teacher_epoch10={k: v for k, v in T10.items()}, teacher_epoch5={k: v for k, v in T5.items() if k != 'tasks'},
        context_e0_teacher10_5ep_nfe2={s: {k: v for k, v in x.items() if k != 'tasks'} for s, x in nfe2.items()},
        context_e0_epoch1={s: {k: v for k, v in x.items() if k != 'tasks'} for s, x in ep1.items()},
        sources=dict(reference_e0=REF_E0, reference_e1=REF_E1, e0_nfe2=E0_NFE2, e0_epoch1=E0_EPOCH1))
    (HERE / 'results.json').write_text(json.dumps(results, indent=1) + '\n')

    with open(HERE / 'task_comparison.csv', 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        cols = [('e0', s, st[s]) for s in E0_RUNS] + [('e0_nfe2_context', s, nfe2[s]) for s in E0_NFE2] + \
               [('teacher_epoch10', 'v0', T10)] + [('e1', s, e1[s]['stats']) for s in E1_RUNS] + \
               [('teacher_epoch5', 'v0', T5)]
        w.writerow(['task'] + [f'{g}:{s}:{x}' for g, s, _ in cols for x in ('clean', 'randomized', 'sr_percent')])
        for t in T10['tasks']:
            w.writerow([t] + [v for _, _, d in cols for v in (d['tasks'][t]['clean'], d['tasks'][t]['randomized'],
                                                             f"{d['tasks'][t]['sr']:.2f}")])

    text = (HERE / 'tools' / 'README.template.md').read_text()
    for name, body in T.items():
        text = text.replace(f'<!-- TABLE:{name} -->', body)
    left = re.findall(r'<!-- TABLE:(\w+) -->', text)
    assert not left, left
    (HERE / 'README.md').write_text(text)
    print('wrote README.md, results.json, task_comparison.csv')


if __name__ == '__main__':
    main()
