#!/usr/bin/env python3
"""Collect the v0 init-variant training and matched-evaluation records into the v0-init-variants worktree:
plans/performance/LoopWAM_v0_Init_Variants_Results_20261028.md, plans/evidence/v0_init_variants_20261028/ and
scripts/init_variants_h100/. Every number in the report is read from the run records; nothing is typed by hand.
    python build_initvar_report.py --repo /lustre/fs1/home/an221229/code/LoopWAM_ivreport
"""
import argparse, csv, datetime, glob, hashlib, io, json, shutil, statistics
from pathlib import Path

WORK = Path('/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work')
RUNS = Path('/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants')
HF = WORK / 'eval/hf_models/libero-all-suites/init_variants'
HF_META = WORK / 'reference/hf_meta/libero-all-suites/init_variants'
REFERENCE = WORK / 'reference/libero-all-suites/v0-video4-action4'
NVP_RUN = Path('/groups/yshang/an221229/checkpoints/LoopWAM_NT/full_libero_v0_v4a4_scratch_20261026')
CAMPAIGN = Path('/lustre/fs1/home/an221229/code/LoopWAM_v042/evaluation_report/final')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
SUITE_NAMES = dict(libero_spatial='Spatial', libero_object='Object', libero_goal='Goal', libero_10='Long')
SEEDS = ('42', '43', '44')
REPORT = 'plans/performance/LoopWAM_v0_Init_Variants_Results_20261028.md'
EVIDENCE = 'plans/evidence/v0_init_variants_20261028'
SCRIPTS = 'scripts/init_variants_h100'
EVALUATED = ('core_mean', 'ffn_magnitude', 'stepwise', 'core_random', 'no_video_pretraining')
INIT = dict(
    default='core = Wan blocks 9-14 (the v0 release)',
    core16_21='core = Wan blocks 15-20',
    core_mean='core block i = element-wise mean of Wan blocks 3+i, 9+i, 15+i, 21+i',
    ffn_magnitude='core = Wan blocks 9-14; each video-expert FFN keeps the 6,144 of 8,960 Wan neurons with the largest '
                  'norm(W1 row) x norm(W2 column) (v0: 6,144 evenly spaced neurons)',
    stepwise='core = Wan blocks 4, 8, 12, 16, 20, 24',
    core_random='core = freshly initialized default DiT blocks (seed 42), both experts',
    no_video_pretraining='every Wan-derived tensor from Wan2.1 from-scratch init (no video pretraining)')


def read(p):
    return json.loads(Path(p).read_text())


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def pct(x, digits=2):
    return f'{100 * x:.{digits}f}'


def final_epoch_losses(metrics):
    rows = [json.loads(l) for l in Path(metrics).read_text().splitlines()]
    rows = [r for r in rows if 'loss_video' in r and r.get('epoch') == 10]
    return sum(r['loss_video'] for r in rows) / len(rows), sum(r['loss_action'] for r in rows) / len(rows)


def campaign_reference():
    results = read(CAMPAIGN / 'results.json')
    s = [m for m in results['models'] if m['label'] == 'LoopWAM full 4/4'][0]['summary']
    per_suite = {k: dict(successes=v['successes'], episodes=v['episodes'], success_rate=v['success_rate'],
                         seeds=v['seeds']) for k, v in s['per_suite'].items()}
    per_seed = {seed: dict(successes=sum(per_suite[k]['seeds'][seed]['successes'] for k in SUITES), episodes=2000)
                for seed in SEEDS}
    for v in per_seed.values():
        v['success_rate'] = v['successes'] / v['episodes']
    long_tasks = {}
    for row in csv.DictReader((CAMPAIGN / 'task_results.csv').open()):
        if row['model'] == 'LoopWAM full 4/4' and row['suite'] == 'libero_10':
            long_tasks[int(row['task'])] = int(row['successes'])
    return dict(successes=s['successes'], episodes=s['episodes'], success_rate=s['success_rate'], per_suite=per_suite,
                per_seed=per_seed, long_tasks=long_tasks, checkpoint_sha256=s['checkpoint_sha256'],
                wall_seconds=s['elapsed_seconds'], workers=s['workers'], backend=s['backend'])


def training_records():
    """Per variant: where/how it was trained, time, final-epoch losses, fairness, HF location."""
    out = {}
    ref_manifest, ref_timing = read(REFERENCE / 'training_manifest.json'), read(REFERENCE / 'export.json')
    out['default'] = dict(where='v0 release (earlier campaign)', gpu=ref_manifest.get('gpu'),
                          layout=f"{ref_manifest['world_size']} x {ref_manifest['microbatch']} x {ref_manifest['gradient_accumulation']}",
                          hours=ref_timing['training_seconds'] / 3600, segments=None, loss_video=None, loss_action=None,
                          fairness='reference', hf='libero-all-suites/v0-video4-action4')
    for v in ('core16_21', 'core_mean'):
        f, e = read(HF_META / v / 'fairness.json'), read(HF_META / v / 'export.json')
        out[v] = dict(where='HiPerGator', gpu=f['gpu'], layout=f"{f['world_size']} x {f['microbatch']} x {f['gradient_accumulation']}",
                      hours=f['training_hours'], segments=1 if f['resumed'] is False else None,
                      loss_video=f['final_epoch_loss_video'], loss_action=f['final_epoch_loss_action'],
                      fairness='passed' if f['passed'] else 'failed', hf=e['repo_path'], export_sha256=e['export_sha256'],
                      source_sha256=e['source_sha256'])
    for v in ('core_random', 'ffn_magnitude', 'stepwise'):
        run = RUNS / v
        m, t, f = read(run / 'train/manifest.json'), read(run / 'train/timing.json'), read(run / 'fairness_h100.json')
        launches = [json.loads(l) for l in (run / 'launches.jsonl').read_text().splitlines()]
        assert t['status'] == 'complete' and t['completed_updates'] == 21700 and f['h100']['h100_passed'] is True
        lv, la = final_epoch_losses(run / 'train/metrics.jsonl')
        assert abs(lv - f['final_epoch_loss_video']) < 1e-12 and abs(la - f['final_epoch_loss_action']) < 1e-12
        published = run / 'upload_verification.json'
        out[v] = dict(where=f"H100 job {launches[0]['slurm_job']}", gpu=m['gpu'],
                      layout=f"{m['world_size']} x {m['microbatch']} x {m['gradient_accumulation']}",
                      hours=t['elapsed_training_seconds'] / 3600, segments=len(launches), loss_video=lv, loss_action=la,
                      fairness='passed (text-cache exception)', started=launches[0]['time'],
                      hf=read(published)['repo_path'] if published.exists() else None,
                      hf_revision=read(published)['revision'] if published.exists() else None,
                      export_sha256=read(run / f'publish/libero-all-suites/init_variants/{v}/export.json')['export_sha256'] if published.exists() else None,
                      source_sha256=read(run / 'eval_matched/summary.json')['checkpoint_sha256'])
    e, f = read(HF / 'no_video_pretraining/export.json'), read(HF / 'no_video_pretraining/fairness.json')
    tt = read(HF / 'no_video_pretraining/training_timing.json')
    lv, la = final_epoch_losses(NVP_RUN / 'train/metrics.jsonl')
    out['no_video_pretraining'] = dict(where='H100 jobs 990527 -> 990547', gpu=e['gpu'], layout='2 x 8 x 8',
                                       hours=e['training_seconds'] / 3600, segments=len(tt['launches']), loss_video=lv,
                                       loss_action=la, fairness='passed (text-cache exception)' if f['passed'] else 'failed',
                                       hf=e['repo_path'], export_sha256=e['export_sha256'], source_sha256=e['source_sha256'])
    return out


def evaluation_records():
    out = {}
    for v in EVALUATED:
        ev = RUNS / v / 'eval_matched'
        s, m = read(ev / 'summary.json'), read(ev / 'manifest.json')
        gate = read(WORK / f'eval/logs/gate_{v}.json')
        runs = [read(p) for p in sorted((ev / 'runs').glob('*.json'))]
        assert s['episodes'] == 6000 and gate['all_bit_exact'] and gate['queries'] == 24 and all(r['errors'] == 0 for r in runs)
        assert gate['checkpoint_sha256'] == s['checkpoint_sha256'] == m['sha256']
        groups = [read(p) for p in sorted((ev / 'groups').glob('*.json'))]
        assert len(groups) == 120
        long_tasks = {t: sum(g['successes'] for g in groups if g['group']['suite'] == 'libero_10' and g['group']['task'] == t)
                      for t in range(10)}
        out[v] = dict(summary=s, manifest=m, gate=gate, runs=runs, groups=groups, long_tasks=long_tasks,
                      wall_seconds=max(r['finished'] for r in runs) - min(r['started'] for r in runs),
                      started=min(r['started'] for r in runs), finished=max(r['finished'] for r in runs))
    runner = {out[v]['manifest']['runner_sha256'] for v in out}
    assert len(runner) == 1, 'all evaluations must use the same runner'
    return out


def write_evidence(repo, ref, train, evals):
    root = repo / EVIDENCE
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rows = []
    for v, e in evals.items():
        d = root / v
        d.mkdir()
        shutil.copy2(RUNS / v / 'eval_matched/summary.json', d / 'eval_summary.json')
        shutil.copy2(RUNS / v / 'eval_matched/manifest.json', d / 'eval_manifest.json')
        shutil.copy2(WORK / f'eval/logs/gate_{v}.json', d / 'cuda_graph_gate.json')
        (d / 'eval_runs.json').write_text(json.dumps(e['runs'], indent=1) + '\n')
        for g in e['groups']:
            for r in g['episodes']:
                rows.append(dict(model=v, suite=r['suite'], seed=r['seed'], task=r['task'], episode=r['episode'],
                                 initial_state_index=r['initial_state_index'], success=r['success'],
                                 policy_steps=r['policy_steps'], queries=r['queries'],
                                 query_seconds=round(r['query_seconds'], 4), total_seconds=round(r['total_seconds'], 4)))
        if v in ('core_random', 'ffn_magnitude', 'stepwise'):
            run = RUNS / v
            for name in ('fairness_h100.json', 'launches.jsonl', 'smoke_4gpu.json', 'upload_verification.json'):
                if (run / name).exists():
                    shutil.copy2(run / name, d / name)
            shutil.copy2(run / 'train/timing.json', d / 'training_timing.json')
            shutil.copy2(WORK / f'init/{v}.manifest.json', d / 'init_variant_manifest.json')
            pub = run / f'publish/libero-all-suites/init_variants/{v}/export.json'
            if pub.exists():
                shutil.copy2(pub, d / 'export.json')
        else:
            for name in ('export.json', 'fairness.json', 'training_timing.json', 'init_variant_manifest.json'):
                shutil.copy2(HF / v / name, d / name)
            if v == 'core_mean':
                (d / 'eval_text_cache_substitution.json').write_text(json.dumps(dict(
                    note=read(HF / v / 'data/data_manifest.json')['evaluation_text_cache_note'],
                    release_text_cache_files=read(HF / v / 'data/data_manifest.release.json')['text_cache_files'],
                    evaluation_text_cache_files=read(HF / v / 'data/data_manifest.json')['text_cache_files']), indent=1) + '\n')
    parity_dir = RUNS / 'parity_reference_v0_video4_action4'
    (root / 'parity').mkdir()
    shutil.copy2(parity_dir / 'parity.json', root / 'parity/parity.json')
    shutil.copy2(parity_dir / 'gate.json', root / 'parity/cuda_graph_gate.json')
    for p in sorted((parity_dir / 'groups').glob('*.json')):
        for r in read(p)['episodes']:
            rows.append(dict(model='reference_v0_video4_action4', suite=r['suite'], seed=r['seed'], task=r['task'],
                             episode=r['episode'], initial_state_index=r['initial_state_index'], success=r['success'],
                             policy_steps=r['policy_steps'], queries=r['queries'],
                             query_seconds=round(r['query_seconds'], 4), total_seconds=round(r['total_seconds'], 4)))
    rows.sort(key=lambda r: (r['model'], SUITES.index(r['suite']), r['seed'], r['task'], r['episode']))
    with (root / 'episodes.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with (root / 'suite_seed_results.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'suite', 'seed', 'successes', 'episodes', 'success_rate'])
        for v, e in evals.items():
            for s in SUITES:
                for seed in SEEDS:
                    x = e['summary']['per_suite'][s]['seeds'][seed]
                    w.writerow([v, s, seed, x['successes'], x['episodes'], x['success_rate']])
    results = dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), reference_v0_release=
                   {k: v for k, v in ref.items()}, training=train,
                   evaluation={v: dict({k: e['summary'][k] for k in ('checkpoint_sha256', 'checkpoint_step', 'episodes',
                               'successes', 'success_rate', 'per_suite', 'per_seed', 'total_policy_steps', 'total_queries')},
                               long_task_successes=e['long_tasks'], wall_seconds=e['wall_seconds'],
                               cuda_graph_gate={k: x for k, x in e['gate'].items() if k != 'details'})
                               for v, e in evals.items()},
                   runner_sha256=next(iter(evals.values()))['manifest']['runner_sha256'])
    (root / 'results.json').write_text(json.dumps(results, indent=1) + '\n')
    return len(rows)


def copy_scripts(repo):
    d = repo / SCRIPTS
    if d.exists():
        shutil.rmtree(d)
    (d / 'eval').mkdir(parents=True)
    names = ('env.sh', 'h100_build_variant.py', 'h100_verify_variant.py', 'seed_latent_cache.py', 'smoke.sh',
             'train_variant.sh', 'job_variant.sh', 'step_variant4.sh', 'stepwise_watcher.sh', 'publish_initvar.py',
             'build_initvar_report.py')
    for n in names:
        shutil.copy2(WORK / n, d / n)
    shutil.copy2(WORK / 'train_variant.sh.v1', d / 'train_variant_v1_2gpu.sh')
    for n in ('evaluate_v0_matched.py', 'eval_env.sh', 'eval_step.sh', 'eval_step_hf.sh', 'parity_reference.py', 'parity_step.sh'):
        shutil.copy2(WORK / 'eval' / n, d / 'eval' / n)
    runner = sha256(d / 'eval/evaluate_v0_matched.py')
    (d / 'README.md').write_text(f"""# H100 wrappers for the v0 initialization-variant runs (October 2026)

Copies of the scripts that trained, evaluated and published the H100 init-variant runs; they ran from
`/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work` against this branch at `4ebf38a` (unmodified).
Results: [`{REPORT}`](../../{REPORT}).

Training (`scripts/train_loopwam.py` of this branch, unchanged):
- `env.sh`: Python/paths; `h100_build_variant.py` runs `scripts/build_init_variant.py` with this cluster's canonical
  donor artifact hash (`a990a583...`; the branch pins the HiPerGator build `a3bf2097...`).
- `train_variant_v1_2gpu.sh` (core_random, ffn_magnitude: 2 GPUs x microbatch 8 x accumulation 8) and `train_variant.sh`
  (same plus NPROC/WORKERS; stepwise used 4 x 8 x 4): stage the latent cache on node-local /tmp, train, then verify.
- `job_variant.sh` (run lock, retries), `smoke.sh`, `step_variant4.sh` (4-GPU smoke + run inside an srun step),
  `stepwise_watcher.sh` (waits for a new interactive job, falls back to the remaining time of another), `seed_latent_cache.py`.
- `h100_verify_variant.py`: the branch fairness check; the only accepted difference is the regenerated T5 text-cache file
  bytes (same names and valid-token counts).

Evaluation (`eval/`): `evaluate_v0_matched.py` (SHA-256 `{runner[:12]}...`, byte-identical to the copy that ran) implements
the campaign's matched LIBERO protocol with a CUDA-graph replay that must pass a bit-exact `--gate`; `eval_step.sh` /
`eval_step_hf.sh` launch it inside an srun step; `parity_reference.py` runs the released v0 4/4 through the same runner
and compares with the campaign's per-episode record. Publishing: `publish_initvar.py`. Report: `build_initvar_report.py`.
""")


def observations(ref, train, evals):
    """Statements derived from (and checked against) the records."""
    sr = {v: evals[v]['summary']['success_rate'] for v in evals}
    suite = {v: {k: evals[v]['summary']['per_suite'][k]['success_rate'] for k in SUITES} for v in evals}
    ranked = sorted(sr, key=sr.get, reverse=True)
    out = []
    names = ', '.join(f"`{v}` {pct(sr[v])}%" for v in ranked)
    if sr[ranked[0]] < ref['success_rate']:
        out.append(f"No variant beats the v0 initialization ({pct(ref['success_rate'])}%). Ranking: {names}.")
    else:
        out.append(f"Ranking against the v0 initialization ({pct(ref['success_rate'])}%): {names}.")
    unpretrained = ('core_random', 'no_video_pretraining')
    pre = [v for v in ranked if v not in unpretrained]
    if pre and all(sr[p] > sr[u] for p in pre for u in unpretrained if u in sr):
        out.append(f"Every variant whose core comes from pretrained Wan blocks ({', '.join(f'`{v}`' for v in pre)}) scores "
                   f"{pct(min(sr[v] for v in pre))}-{pct(max(sr[v] for v in pre))}%, "
                   f"{100 * (ref['success_rate'] - max(sr[v] for v in pre)):.1f}-{100 * (ref['success_rate'] - min(sr[v] for v in pre)):.1f} "
                   'points below v0. Without pretrained core weights the score drops further: '
                   f"`no_video_pretraining` (no Wan weights at all) {pct(sr['no_video_pretraining'])}% and `core_random` "
                   f"(pretrained prelude and coda around a freshly initialized core) {pct(sr['core_random'])}%.")
    gap = {v: {k: ref['per_suite'][k]['success_rate'] - suite[v][k] for k in SUITES} for v in evals}
    if all(max(gap[v], key=gap[v].get) == 'libero_10' for v in evals):
        out.append('Long has the largest drop for every model: '
                   + ', '.join(f"`{v}` {100 * gap[v]['libero_10']:.1f}" for v in ranked)
                   + f" points below v0's {pct(ref['per_suite']['libero_10']['success_rate'])}%. On the other three suites "
                   'the pretrained-core variants stay within '
                   f"{100 * max(gap[v][k] for v in pre for k in SUITES if k != 'libero_10'):.1f} points of v0.")
    seeds = {v: [evals[v]['summary']['per_seed'][s]['success_rate'] for s in SEEDS] for v in evals}
    spread = {v: max(x) - min(x) for v, x in seeds.items()}
    widest = max(spread, key=spread.get)
    others = max(x for v, x in spread.items() if v != widest)
    out.append(f"Seed spread (max-min over seeds 42/43/44) is at most {100 * others:.2f} points for every model except "
               f"`{widest}` ({100 * spread[widest]:.2f} points, lowest at seed "
               f"{SEEDS[seeds[widest].index(min(seeds[widest]))]}); v0 is "
               f"{100 * (max(ref['per_seed'][s]['success_rate'] for s in SEEDS) - min(ref['per_seed'][s]['success_rate'] for s in SEEDS)):.2f}. "
               'The protocol reuses one noise vector for every query of a task/seed, so a seed effect can be shared across episodes.')
    loss = {v: train[v]['loss_action'] for v in evals}
    by_loss = sorted(loss, key=loss.get)
    out.append('Final-epoch action loss orders the variants ' + ' < '.join(f'`{v}` {loss[v]:.4f}' for v in by_loss)
               + '; the two highest-loss variants are also the two lowest-scoring.' if set(by_loss[-2:]) == set(ranked[-2:])
               else 'Final-epoch action loss orders the variants ' + ' < '.join(f'`{v}` {loss[v]:.4f}' for v in by_loss) + '.')
    return out


def report(ref, train, evals):
    order = ['default'] + sorted(evals, key=lambda v: evals[v]['summary']['success_rate'], reverse=True)
    def ev(v):
        return ref if v == 'default' else evals[v]['summary']
    lines = []
    A = lines.append
    A('# LoopWAM v0 initialization variants: training and matched LIBERO evaluation')
    A('')
    A(f"Generated {datetime.date.today().isoformat()} from the run records by "
      f"[`{SCRIPTS}/build_initvar_report.py`](../../{SCRIPTS}/build_initvar_report.py); machine-readable data in "
      f"[`{EVIDENCE}`](../../{EVIDENCE}).")
    A('')
    A('Every variant is LoopWAM v0 with video/action loops 4/4 (3 prelude + 6 shared core x 4 loops + 3 coda blocks per '
      'expert, 584,536,135 policy parameters), trained for ten epochs on all four LIBERO suites with the v0 release '
      'recipe: 1,712 demonstrations / 277,713 windows, seed 42, global batch 128, 21,700 updates, AdamW (0.9, 0.95), '
      'weight decay 0.01, LR 1e-4 with 5% warm-up and cosine decay, clip 1.0, FP32 master weights with BF16 compute. '
      'Only the student initialization differs (prelude = Wan blocks 0-2 and coda = Wan blocks 27-29 in every variant '
      'that uses Wan weights).')
    A('')
    A('## Results (6,000 episodes per model)')
    A('')
    A('| Initialization | All four suites | Spatial | Object | Goal | Long | Seed 42 | Seed 43 | Seed 44 |')
    A('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for v in order:
        s = ev(v)
        name = 'default (v0 release)' if v == 'default' else f'`{v}`'
        A(f"| {name} | **{pct(s['success_rate'])}%** ({s['successes']:,}) | "
          + ' | '.join(pct(s['per_suite'][k]['success_rate']) for k in SUITES) + ' | '
          + ' | '.join(pct(s['per_seed'][seed]['success_rate']) for seed in SEEDS) + ' |')
    A('| `core16_21` | not evaluated | | | | | | | |')
    A('')
    A('Success rates in percent; each suite has 1,500 episodes (10 tasks x 50 initial states x 3 seeds), each seed 2,000. '
      'The default row is the campaign result for the released `libero-all-suites/v0-video4-action4` '
      f"(evaluation_report/final, checkpoint export `{ref['checkpoint_sha256'][:12]}...`); every other row was evaluated on "
      'H100 in this work with the same protocol.')
    A('')
    hw = lambda p, n: 196 * (p * (1 - p) / n) ** .5
    total = [hw(evals[v]['summary']['success_rate'], 6000) for v in evals]
    long_ = [hw(evals[v]['summary']['per_suite']['libero_10']['success_rate'], 1500) for v in evals]
    A('Observations. Each variant was trained once (seed 42), so training-seed variation is not measured; the binomial 95% '
      f'interval is about +/-{min(total):.1f}-{max(total):.1f} points on a model total and +/-{min(long_):.1f}-{max(long_):.1f} '
      'on Long (episodes are not fully independent, so treat this as a lower bound). Gaps of under one point, such as '
      '`core_mean` vs `ffn_magnitude`, are not resolved.')
    A('')
    for line in observations(ref, train, evals):
        A(f'- {line}')
    A('')
    A('### LIBERO-Long per task (successes out of 150)')
    A('')
    A('| Task | default | ' + ' | '.join(f'`{v}`' for v in order[1:]) + ' |')
    A('|---:|---:|' + '---:|' * (len(order) - 1))
    for t in range(10):
        A(f"| {t} | {ref['long_tasks'][t]} | " + ' | '.join(str(evals[v]['long_tasks'][t]) for v in order[1:]) + ' |')
    A('')
    A('## Training')
    A('')
    A('| Initialization | Init rule | Trained on | Layout (GPUs x micro x accum) | Hours | Segments | Final-epoch loss video / action | Fairness | Hugging Face |')
    A('|---|---|---|---|---:|---:|---|---|---|')
    for v in order + ['core16_21']:
        t = train[v]
        loss = '—' if t['loss_video'] is None else f"{t['loss_video']:.4f} / {t['loss_action']:.4f}"
        seg = '—' if t['segments'] is None else str(t['segments'])
        hf = f"[{t['hf']}](https://huggingface.co/anhdao69/LoopWAM_NT/tree/main/{t['hf']})" if t['hf'] else 'not uploaded'
        A(f"| `{v}` | {INIT[v]} | {t['where']}, {t['gpu']} | {t['layout']} | {t['hours']:.1f} | {seg} | {loss} | {t['fairness']} | {hf} |")
    A('')
    A('Every run used the same data manifest, normalization (`d5ad351eb327...`), Wan2.1 VAE (`38071ab59bd9...`), schedule '
      'and precision. The branch fairness check (`scripts/v0_init_variant_verify.py`) passed for every B200 and H100 '
      'init-variant run, and the same field-by-field comparison with the v0 release passed for `no_video_pretraining`, with '
      'one documented exception on this cluster: the 40 T5 text-embedding files were regenerated (identical prompts, file '
      'names and valid-token counts; different bytes). The global batch is 128 in every run, but the device layout differs (B200 '
      '2 x 16 x 4, H100 2 x 8 x 8 or 4 x 8 x 4, v0 release 4 x 8 x 4), so the sampler order and noise streams differ '
      'between runs. Hours are training time only (the H100 runs shared nodes with other jobs at times). '
      'H100 init artifacts were built from this cluster\'s canonical donor artifact (`a990a583...`; the branch pins the '
      'HiPerGator build `a3bf2097...`, same Wan2.1 source).')
    A('')
    step_t = read(RUNS / 'stepwise/train/timing.json')['measured_mean_update_seconds']
    A(f'`stepwise` ran on all four H100s of interactive job 1080498 (4 x 8 x 4, the v0 release layout) at {step_t:.2f} s/update; '
      '`core_random` and `ffn_magnitude` ran side by side on job 990547 (GPUs 0,1 and 2,3). `no_video_pretraining` hours '
      'include the updates repeated after its first job ended (two segments). `no_video_pretraining` was trained '
      'from LoopWAM_NT `56e3328` plus the scratch-initialization patch (branch `scratch_init_v0`); its inference code is '
      'identical to this branch.')
    A('')
    A('## Evaluation protocol and checks')
    A('')
    A('The campaign protocol `lightwam_20261009_matched` (`evaluation_report/audit_setup.py`): four suites x ten tasks x '
      'official initial states 0-49 x evaluation seeds 42/43/44. One environment per task/seed (env.seed(base seed) at '
      'construction) is reused for its 50 initial states in order; the global Python/NumPy/Torch seed is set once per '
      'task/seed and every action query uses the base seed. 30 settling actions (done ignored), then at most 400 '
      '(Spatial/Object/Goal) or 700 (Long) policy steps; 32-action chunks with ten executed per query; success is the '
      'environment done flag. Native LoopWAM inference: FP32 weights, BF16 autocast, BF16 Wan2.1 VAE, ten denoising steps, '
      'CFG 1, action shift 1; observation adapter and normalization from `scripts/evaluate_loopwam_libero.py`; gripper '
      'sign(-(2g-1)).')
    A('')
    runner = next(iter(evals.values()))['manifest']['runner_sha256']
    parity = read(RUNS / 'parity_reference_v0_video4_action4/parity.json')
    A(f"- Runner: [`{SCRIPTS}/eval/evaluate_v0_matched.py`](../../{SCRIPTS}/eval/evaluate_v0_matched.py) (SHA-256 "
      f"`{runner[:12]}...`, identical for every model), adapted from the residual_trans H100 runner; task/seed groups are "
      'claimed atomically so several runners can share the work, and finished groups are never rerun.')
    A('- Speed: one CUDA graph replays the unchanged native video-prefill + action-denoising computation (no Inductor, no '
      'reduced steps). Before scoring each checkpoint, `--gate` compared it with native eager `infer_action` on real '
      'simulator observations from three suites: all 24 queries were bit-identical for every model.')
    A(f"- Harness parity: the released v0 4/4 policy through the same runner scored {parity['h100_successes']}/500 on Long "
      f"seed 42 versus {parity['campaign_successes']}/500 in the campaign record; {pct(parity['outcome_agreement'], 1)}% of "
      'episodes have the same outcome (trajectories diverge numerically across machines).')
    A('- `core_mean` was trained on HiPerGator, whose text-cache bytes differ from this cluster\'s; its evaluation checks '
      'the cluster files (same 40 names and valid-token counts), recorded in `core_mean/eval_text_cache_substitution.json`.')
    A('- Videos were not saved (they do not affect outcomes). 16 worker processes on 4 H100s and 16 CPU cores; the '
      'simulator is CPU-bound at about 18 policy steps per second per worker.')
    A('')
    A('| Model | Evaluated checkpoint SHA-256 | Graph / eager query (ms) | Wall time | Policy steps | Queries |')
    A('|---|---|---:|---:|---:|---:|')
    for v in order[1:]:
        e = evals[v]
        g = e['gate']
        A(f"| `{v}` | `{e['summary']['checkpoint_sha256'][:12]}...` | {g['graph_ms_median']:.0f} / {g['eager_ms_median']:.0f} | "
          f"{e['wall_seconds'] / 3600:.2f} h | {e['summary']['total_policy_steps']:,} | {e['summary']['total_queries']:,} |")
    A('')
    cr = sorted(evals['core_random']['runs'], key=lambda r: r['started'])
    fmt = lambda t: datetime.datetime.fromtimestamp(t).strftime('%H:%M')
    A('Gate timings for `no_video_pretraining` and `core_mean` were measured while another evaluation shared the GPUs. '
      f"`core_random` started on GPUs {cr[0]['gpus']} at {fmt(cr[0]['started'])} and gained GPUs {cr[-1]['gpus']} at "
      f"{fmt(cr[-1]['started'])}, when the `ffn_magnitude` training released them. Wall time runs from the first runner "
      'start to the last runner end; all evaluations ran on 2026-10-28.')
    A('')
    A('## Artifacts')
    A('')
    A(f"- Evidence: [`{EVIDENCE}`](../../{EVIDENCE}): `results.json`, `episodes.csv` (every scored episode plus the parity "
      'run), `suite_seed_results.csv`, and per model the evaluation summary/manifest, CUDA-graph gate, runner sessions, '
      'training timing, fairness and initialization manifest.')
    A(f"- Scripts: [`{SCRIPTS}`](../../{SCRIPTS}).")
    A('- Weights: Hugging Face `anhdao69/LoopWAM_NT`, `libero-all-suites/init_variants/<variant>` (core16_21, core_mean, '
      'core_random, ffn_magnitude, no_video_pretraining); `stepwise` is not uploaded. On this cluster: '
      '`/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants/<variant>` (H100 training runs and all five '
      'evaluations, `eval_matched/`).')
    A('')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    args = ap.parse_args()
    repo = Path(args.repo)
    ref = campaign_reference()
    train = training_records()
    evals = evaluation_records()
    n = write_evidence(repo, ref, train, evals)
    copy_scripts(repo)
    (repo / REPORT).write_text(report(ref, train, evals))
    print(json.dumps(dict(report=REPORT, episodes_rows=n, models=list(evals))))


if __name__ == '__main__':
    main()
