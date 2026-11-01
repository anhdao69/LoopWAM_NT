"""GR1 v0 launcher: microbatch timing (200-update smoke), native gates, production with bounded auto-resume.

python scripts/run_loopwam_gr1.py --root <run>/train --shared <run>/shared --latent-dir <stage>/latents --phase timing|gates|production
"""
import argparse, hashlib, json, statistics, subprocess, sys, time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True); p.add_argument('--shared', required=True); p.add_argument('--latent-dir', required=True)
p.add_argument('--phase', choices=['timing', 'gates', 'production'], required=True)
p.add_argument('--micro', default='16', help='layouts: 16 = microbatch 16 without checkpointing; 32c20 = microbatch 32, '
               'checkpointed blocks except the last 20 virtual layers')
p.add_argument('--rope', choices=('complex', 'real'), default='real')
p.add_argument('--text-tokens', type=int, default=48, help='longest training prompt is 41 T5 tokens')
p.add_argument('--compile-mode', default='default', choices=('default', 'max-autotune-no-cudagraphs')); p.add_argument('--timing-updates', type=int, default=200)
p.add_argument('--max-restarts', type=int, default=3)
a = p.parse_args(); r = Path(a.root); r.mkdir(parents=True, exist_ok=True)
read = lambda q: json.loads(Path(q).read_text())


def source_hashes():
    return {str(q): hashlib.sha256(q.read_bytes()).hexdigest() for root in ('src', 'scripts') for q in sorted(Path(root).rglob('*.py'))}


def layout(token, rope, text_tokens, compile_mode='default'):
    micro, _, keep = token.partition('c')
    return dict(micro=int(micro), checkpoint_blocks=bool(_), keep_last=int(keep or 0), rope=rope, text_tokens=text_tokens,
                compile_mode=compile_mode)


def tag(cfg):
    return f"m{cfg['micro']}" + (f"c{cfg['keep_last']}" if cfg.get('checkpoint_blocks') else '') + f"_{cfg.get('rope', 'complex')}" + (f"_t{cfg['text_tokens']}" if cfg.get('text_tokens', 128) < 128 else '') + \
        ('_autotune' if cfg.get('compile_mode', 'default') != 'default' else '')


def base(cfg):
    blocks = ['--checkpoint-blocks', '--keep-last', str(cfg.get('keep_last', 0))] if cfg.get('checkpoint_blocks') else ['--no-checkpoint-blocks']
    return ['torchrun', '--standalone', '--nproc_per_node=4', 'scripts/train_loopwam_gr1.py', '--dataset', 'robocasa_gr1',
            '--run-name', 'LoopWAM-v0-RoboCasa-GR1', '--init-artifact', 'checkpoints/LoopWAM/wan21_compact_donors.pt',
            '--shared-dir', a.shared, '--latent-dir', a.latent_dir, '--epochs', '5', '--global-batch', '256', '--seed', '42',
            '--lr-backbone', '0.0001', '--lr-robot-io', '0.0001', '--warmup-fraction', '0.05', '--save-epochs', '3,4,5',
            '--checkpoint-every', '1000', '--text-threads', '4', '--microbatch', str(cfg['micro']), *blocks,
            '--rope', cfg.get('rope', 'complex'), '--text-tokens', str(cfg.get('text_tokens', 128)), '--compile',
            '--compile-mode', cfg.get('compile_mode', 'default')] + \
           (['--allow-partial-cache'] if a.phase != 'production' else [])


def run(name, cmd, check=True):
    (r / (name + '.command.json')).write_text(json.dumps(cmd, indent=2))
    with (r / (name + '.log')).open('a') as log:
        log.write(f'=== {time.strftime("%Y-%m-%d %H:%M:%S")} {name}\n'); log.flush()
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if code and check:
        raise RuntimeError(f'{name} failed ({code}); see {r / (name + ".log")}')
    return code


def metrics(path):
    return [json.loads(l) for l in (Path(path) / 'metrics.jsonl').read_text().splitlines() if l.strip()]


sources = source_hashes()
prep = read(Path(a.shared) / 'data/prepare_report.json')
if a.phase == 'timing':
    rows = {}
    cfgs = {}
    for token in a.micro.split(','):
        cfg = layout(token, a.rope, a.text_tokens, a.compile_mode); m = tag(cfg); cfgs[m] = cfg
        name = f'timing_{m}'
        code = run(name, base(cfg) + ['--output-dir', str(r / name), '--max-updates', str(a.timing_updates), '--no-save',
                                      '--profile-updates', '40:42'], check=False)
        got = metrics(r / name) if (r / name / 'metrics.jsonl').exists() else []
        steady = [x['seconds'] for x in got if x['update'] > 20]
        rows[m] = dict(exit_code=code, updates=len(got), complete=code == 0 and len(got) == a.timing_updates,
                       median_s_per_update=statistics.median(steady) if steady else None,
                       mean_s_per_update=statistics.fmean(steady) if steady else None,
                       first_update_s=got[0]['seconds'] if got else None,
                       max_data_wait_s=max((x['data_wait_seconds'] for x in got if x['update'] > 20), default=None),
                       peak_allocated_gb=max((x['peak_allocated_gb'] for x in got), default=None),
                       peak_reserved_gb=max((x['peak_reserved_gb'] for x in got), default=None),
                       loss_first=got[0]['loss'] if got else None, loss_last=got[-1]['loss'] if got else None,
                       action_loss_last=got[-1]['loss_action'] if got else None,
                       **{f'median_{k}': statistics.median([x[k] for x in got if x['update'] > 20 and x['update'] not in (41, 42)])
                          for k in ('gpu_micro_seconds', 'gpu_optimizer_seconds', 'host_issue_seconds', 'prefetch_build_seconds')
                          if steady and k in got[-1]})
        if rows[m]['median_s_per_update']:
            s = rows[m]['median_s_per_update']
            rows[m]['projected_hours'] = round(s * prep['total_updates'] / 3600, 1)
        print(json.dumps({m: rows[m]}), flush=True)
    ok = {m: v for m, v in rows.items() if v['complete']}
    if not ok:
        raise SystemExit('No microbatch completed the timing smoke')
    pick = min(ok, key=lambda m: ok[m]['median_s_per_update'])
    selected = dict(cfgs[pick], name=f'timing_{pick}', compile=True, rows=rows,
                    total_updates=prep['total_updates'], updates_per_epoch=prep['updates_per_epoch'])
    (r / 'selected.json').write_text(json.dumps(selected, indent=2)); print(json.dumps(dict(selected=pick)), flush=True)
elif a.phase == 'gates':
    selected = read(r / 'selected.json'); cmd = base(selected)
    run('native_smoke', cmd + ['--output-dir', str(r / 'smoke'), '--max-updates', '10'])
    run('native_reload', [sys.executable, 'scripts/check_loopwam_gr1_checkpoint.py', '--checkpoint', str(r / 'smoke/latest.pt'),
                          '--shared', a.shared, '--latent-dir', a.latent_dir, '--output', str(r / 'reload_check.json')])
    run('native_resume', cmd + ['--output-dir', str(r / 'smoke'), '--resume', '--max-updates', '12'])
    rows = metrics(r / 'smoke'); assert [x['update'] for x in rows] == list(range(1, 13)) and all(x['valid_windows'] == 256 for x in rows)
    reference = metrics(r / selected['name']); drift = []
    for x, y in zip(rows, reference):
        for k in ('loss_video', 'loss_action'):
            d = abs(x[k] - y[k]); drift.append(dict(update=x['update'], metric=k, absolute=d, relative=d / max(abs(y[k]), 1e-12)))
            assert d <= 1e-4 + 1e-3 * abs(y[k]), (x['update'], k, x[k], y[k])
    (r / 'independent_run_drift.json').write_text(json.dumps(drift, indent=2))
    assert read(r / 'smoke/resume_audit_u10.json')['optimizer_exact']
    assert read(r / 'smoke/timing.json')['windows_seen'] == 12 * 256
    run('native_tail', cmd + ['--output-dir', str(r / 'tail_smoke'), '--smoke-tail', '--max-updates', '1', '--no-save'])
    tail = metrics(r / 'tail_smoke'); assert len(tail) == 1 and tail[0]['valid_windows'] == prep['epoch_tail_windows'] and tail[0]['epochs_complete'] == 1
    assert read(r / 'reload_check.json')['passed']
    assert source_hashes() == sources, 'Source changed during gates'
    (r / 'gates_passed.json').write_text(json.dumps(dict(passed=True, sources=sources, selected=selected, updates=12,
        optimizer_restoration_exact=True, resume_cursor_verified=True, independent_run_numerically_consistent=True,
        tail_real_windows=prep['epoch_tail_windows'], max_relative_drift=max(x['relative'] for x in drift), time=time.time()), indent=2))
    print(json.dumps(dict(gates='passed')), flush=True)
else:
    gates = read(r / 'gates_passed.json'); assert gates['passed'] and gates['sources'] == sources, 'Sources changed since gates'
    out = r / 'production'; cmd = base(gates['selected']) + ['--output-dir', str(out)]
    code = run('production', cmd, check=False) if not out.exists() else 1
    restarts, last = 0, -1
    while code:
        timing = read(out / 'timing.json') if (out / 'timing.json').exists() else {}
        progress = timing.get('update', 0)
        if restarts >= a.max_restarts or not (out / 'latest.pt').exists() or progress == last:
            raise SystemExit(f'production stopped (exit {code}) at update {progress}; restarts={restarts}')
        restarts += 1; last = progress
        (r / f'production_restart_{restarts}.json').write_text(json.dumps(dict(exit_code=code, update=progress, time=time.time())))
        time.sleep(60)
        code = run('production', cmd + ['--resume'], check=False)
    t = read(out / 'timing.json')
    assert t['status'] == 'complete' and t['update'] == prep['total_updates'] and t['windows_seen'] == prep['windows'] * 5
    for epoch in (3, 4, 5):
        assert (out / 'checkpoints' / f'epoch_{epoch:02d}.pt').is_file()
    (r / 'verified_complete.json').write_text(json.dumps(dict(passed=True, timing=t, restarts=restarts, completed_unix=time.time()), indent=2))
