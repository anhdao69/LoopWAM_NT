#!/usr/bin/env python3
"""Independent check of robotwin10_v0_4x4_epoch10_20261113 (RoboTwin-10 v0 4/4 epoch 10, H100 evaluation).

Uses only files in this checkout. For both the epoch-10 run and the same-harness epoch-5 reference it re-reads every
per-cell eval.log.gz (accepted episodes: instruction line, Success!/Fail!, the running "Success rate ... current seed"
line), checks it against the per-cell result.json and the episode CSV, then recomputes the per-setting rates, Wilson
intervals, the paired epoch-10 versus epoch-5 statistics (exact McNemar), the per-task table, the repository
references for the same epoch-5 checkpoint, and the numbers quoted in README.md. Also checks SOURCES.tsv hashes of
the copied evidence and the checkpoint identity across the result files and the Hugging Face upload record.

python evaluation_report/verify_robotwin10_v0_4x4_epoch10_20261113.py
"""
import csv
import gzip
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ER = Path(__file__).resolve().parent
D = ER / 'robotwin10_v0_4x4_epoch10_20261113'
TASKS = ['adjust_bottle', 'grab_roller', 'hanging_mug', 'move_stapler_pad', 'open_microwave', 'press_stapler',
         'scan_object', 'stack_blocks_two', 'stamp_seal', 'turn_switch']
SETTINGS = ['demo_clean', 'demo_randomized']
FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print('FAIL', msg)


def episodes_from_log(text):
    out, instr, ok = [], None, None
    for line in re.sub(r'\x1b\[[0-9;]*m', '', text).splitlines():
        line = line.strip()
        if line.startswith('{"loopwam_episode"'):
            d = json.loads(line); instr, ok = (d['instruction'], d['noise_seed']), None
        elif line == 'Success!' or line == 'Fail!':
            ok = line == 'Success!'
        else:
            m = re.match(r'Success rate:\s*(\d+)/(\d+).*current seed:\s*(\d+)', line)
            if m:
                out.append((int(m.group(3)), ok, instr[0] if instr else None, instr[1] if instr else None))
                instr, ok = None, None
    return out


def wilson(k, n):
    z = 1.959964; ph = k / n; den = 1 + z * z / n
    mid = (ph + z * z / (2 * n)) / den; half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return round(100 * (mid - half), 2), round(100 * (mid + half), 2)


def exact_mcnemar(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, sum(math.comb(n, i) for i in range(k + 1)) * 2 / 2 ** n)


def load_run(csv_name, ev_dir, sha):
    rows = list(csv.DictReader(open(D / csv_name)))
    check(len(rows) == 2000, f'{csv_name}: {len(rows)} rows')
    cells = {}
    for t in TASKS:
        for s in SETTINGS:
            rs = [r for r in rows if r['task'] == t and r['setting'] == s]
            check(len(rs) == 100 and [int(r['episode_index']) for r in rs] == list(range(100)), f'{csv_name} {t} {s} rows')
            seeds = [int(r['seed']) for r in rs]
            check(seeds == sorted(set(seeds)), f'{csv_name} {t} {s}: seeds not unique/sorted')
            cdir = D / 'evidence' / ev_dir / f'{t}__{s}'
            res = json.loads((cdir / 'result.json').read_text())
            check(res['checkpoint_sha256'] == sha, f'{ev_dir} {t} {s}: checkpoint')
            check(res['seeds'] == seeds, f'{ev_dir} {t} {s}: result.json seeds')
            check([bool(x) for x in res['success']] == [r['success'] == 'True' for r in rs], f'{ev_dir} {t} {s}: outcomes')
            check(res['successes'] == sum(res['success']) and res['episodes'] == 100, f'{ev_dir} {t} {s}: totals')
            logged = {}
            for lg in sorted(cdir.glob('attempt_*/eval.log.gz')):
                for seed, ok, ins, noise in episodes_from_log(gzip.open(lg, 'rt', errors='replace').read()):
                    check(seed not in logged, f'{ev_dir} {t} {s}: seed {seed} logged twice')
                    logged[seed] = (ok, ins, noise)
            for r in rs:
                ok, ins, noise = logged.get(int(r['seed']), (None, None, None))
                check(ok is (r['success'] == 'True'), f'{ev_dir} {t} {s} seed {r["seed"]}: log outcome')
                check(ins == r['instruction'], f'{ev_dir} {t} {s} seed {r["seed"]}: instruction')
                check(noise == 42, f'{ev_dir} {t} {s} seed {r["seed"]}: noise seed')
            cells[(t, s)] = {int(r['seed']): (r['success'] == 'True', r['instruction']) for r in rs}
    return rows, cells


def main():
    R = json.loads((D / 'results.json').read_text())
    sha10, sha05 = R['checkpoint']['sha256'], R['checkpoint']['parent_epoch05_sha256']
    up = json.loads((D / 'evidence/training/hf_upload_verification_epoch_10.json').read_text())
    check(up['policy_sha256'] == sha10 and up['revision'] == R['checkpoint']['hf_revision'], 'HF upload record')
    par = json.loads((D / 'evidence/training/parent_verification.json').read_text())
    check(par['epoch_05_sha256'] == sha05 and par['tensors_equal_epoch_05'] is True and par['update'] == 45445, 'parent check')
    tim = json.loads((D / 'evidence/training/timing.json').read_text())
    check(tim['status'] == 'complete' and tim['update'] == R['checkpoint']['step'] == 90890, 'training timing')

    rows10, c10 = load_run('episodes.csv', 'epoch10', sha10)
    rows05, c05 = load_run('reference_epoch05_episodes.csv', 'epoch05_same_harness', sha05)
    check(all(len(r['video_sha256']) == 64 and r['video'] for r in rows10), 'epoch-10 video hashes')

    tr = {(r['task'], r['setting']): r for r in csv.DictReader(open(D / 'task_results.csv'))}
    agree = shared_total = 0
    per = {}
    for t in TASKS:
        for s in SETTINGS:
            a, b = c10[(t, s)], c05[(t, s)]
            sh = sorted(set(a) & set(b))
            ao = sum(a[k][0] and not b[k][0] for k in sh); bo = sum(b[k][0] and not a[k][0] for k in sh)
            agree += sum(a[k][1] == b[k][1] for k in sh); shared_total += len(sh)
            k10 = sum(v[0] for v in a.values()); k05 = sum(v[0] for v in b.values())
            per[(t, s)] = (k10, k05, len(sh), ao, bo)
            row = tr[(t, s)]
            check((int(row['epoch10_successes']), int(row['epoch05_same_harness_successes']), int(row['shared_seeds']),
                   int(row['epoch10_only']), int(row['epoch05_only'])) == (k10, k05, len(sh), ao, bo), f'task_results {t} {s}')
            check(abs(float(row['mcnemar_p']) - round(exact_mcnemar(ao, bo), 6)) < 1e-9, f'task_results p {t} {s}')
    check((agree, shared_total) == (R['instruction_agreement_on_shared_seeds']['same'],
                                    R['instruction_agreement_on_shared_seeds']['shared']), 'instruction agreement')
    check(agree == shared_total, 'instructions differ on shared seeds')

    refs = {}
    for f in ('robotwin10_v0_4x4', 'robotwin10_v0_4x4_remote189', 'robotwin10_v0_4x4_remote137_20261105'):
        audit = json.loads((ER / f / 'final_audit.json').read_text())
        check(audit['checkpoint_sha256'] == sha05 and audit['status'] == 'passed', f'{f}: audit checkpoint')
        refs[f] = {(r['task'], r['setting']): int(r['successes']) for r in csv.DictReader(open(ER / f / 'task_results.csv'))}
        for k, v in refs[f].items():
            check(int(tr[k][f'repo_{f}_successes']) == v, f'task_results repo {f} {k}')

    readme = (D / 'README.md').read_text()
    for name, sel in (('clean', ['demo_clean']), ('randomized', ['demo_randomized']), ('combined', SETTINGS)):
        keys = [(t, s) for t in TASKS for s in sel]
        k10 = sum(per[k][0] for k in keys); k05 = sum(per[k][1] for k in keys); n = 100 * len(keys)
        sh = sum(per[k][2] for k in keys); ao = sum(per[k][3] for k in keys); bo = sum(per[k][4] for k in keys)
        s = R['summary'][name]
        check(s['epoch10']['successes'] == k10 and s['epoch10']['episodes'] == n, f'summary {name} epoch10')
        check(s['epoch10']['sr_percent'] == round(100 * k10 / n, 2) and s['epoch10']['wilson95'] == list(wilson(k10, n)), f'summary {name} rate')
        check(s['epoch05_same_harness']['successes'] == k05, f'summary {name} epoch05')
        pr = s['paired']
        check((pr['shared_seeds'], pr['epoch10_only'], pr['epoch05_only']) == (sh, ao, bo), f'summary {name} paired')
        check(pr['difference_pp'] == round(100 * (ao - bo) / sh, 2) and abs(pr['mcnemar_p'] - exact_mcnemar(ao, bo)) < 1e-12, f'summary {name} p')
        for f in refs:
            check(s[f'repo_{f}']['successes'] == sum(refs[f][k] for k in keys), f'summary {name} repo {f}')
        check(f'{k10}/{n} | **{100 * k10 / n:.2f}%**' in readme, f'README {name} headline')
        check(f'{100 * k05 / n:.2f}% | {100 * k10 / n:.2f}% | {sh:,} | {ao} | {bo} | {100 * (ao - bo) / sh:+.2f} pp' in readme,
              f'README {name} paired row')
    for t in TASKS:
        c, r = per[(t, 'demo_clean')], per[(t, 'demo_randomized')]
        check(f'| {t} | {c[0]} | {r[0]} | {(c[0] + r[0]) / 2:.2f}% | {c[1]} | {r[1]} |' in readme, f'README task row {t}')
    for f, label in (('robotwin10_v0_4x4', 'original local'), ('robotwin10_v0_4x4_remote189', 'server189 rerun'),
                     ('robotwin10_v0_4x4_remote137_20261105', 'server137 rerun')):
        kc = sum(refs[f][(t, 'demo_clean')] for t in TASKS); kr = sum(refs[f][(t, 'demo_randomized')] for t in TASKS)
        check(f'| epoch 5 — {label} | {kc / 10:.2f}% | {kr / 10:.2f}% | {(kc + kr) / 20:.2f}% | {kc + kr}/2000 |' in readme,
              f'README reference {f}')

    n_src = 0
    for line in list(open(D / 'SOURCES.tsv'))[1:]:
        src, dst, h = line.rstrip('\n').split('\t')
        data = (D / dst).read_bytes()
        if dst.endswith('.gz') and not src.endswith('.gz'):
            data = gzip.decompress(data)
        if dst.endswith('parent_verification.json'):
            continue  # an extract of parents.json (one entry), not a byte copy
        check(hashlib.sha256(data).hexdigest() == h, f'SOURCES hash {dst}'); n_src += 1

    if FAIL:
        print(f'{len(FAIL)} check(s) failed'); sys.exit(1)
    print(f'PASS: 2 x 2,000 episodes re-derived from {len(list(D.glob("evidence/*/*/attempt_*/eval.log.gz")))} logs; '
          f'epoch 10 clean {R["summary"]["clean"]["epoch10"]["sr_percent"]}% randomized '
          f'{R["summary"]["randomized"]["epoch10"]["sr_percent"]}% combined {R["summary"]["combined"]["epoch10"]["sr_percent"]}%; '
          f'paired vs epoch 5 {R["summary"]["combined"]["paired"]["difference_pp"]:+.2f} pp; {n_src} copied files match SOURCES.tsv')


if __name__ == '__main__':
    main()
