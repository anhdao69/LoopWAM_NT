"""Build matched_baselines/README.md from this directory's verified data and the branch report snapshots.

Inputs: verification/<run>.json (verify_run.py results), data/suite_results.csv (seed spread), runs/<run>/ (manifest,
summary, build reports) and ../branch_reports (snapshots of the other branches' reports, see sync_branch_reports.py).
"""
import csv, json, pathlib, re

HERE = pathlib.Path(__file__).resolve().parent
BR = HERE.parent / 'branch_reports'
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']

# run dir, label in data/*.csv, what was evaluated
RUNS = [
    ('fasterwam', 'FasterWAM', 'public release, HF `hustvl/FasterWAM` `libero/step_021700.pt` (one model for all suites)'),
    ('lightwam', 'Light-WAM', 'public release, HF `l1ziang/lightwam-checkpoints` (one checkpoint per suite)'),
    ('loopwam_dense_s30', 'LoopWAM Dense-S30', 'HF `libero-all-suites/dense-s30`'),
    ('loopwam_dense_s12', 'LoopWAM Dense-S12', 'HF `libero-all-suites/dense-s12`'),
    ('loopwam_v0_ka3', 'LoopWAM v0 Ka3', 'HF `libero-all-suites/v0-video4-action4`, test-time action depth 3 (core loops 2,3,4)'),
    ('loopwam_v0_ka5', 'LoopWAM v0 Ka5', 'same v0 export, test-time action depth 5 (core loops 1,1,2,3,4)'),
    ('loopwam_v0_ka6', 'LoopWAM v0 Ka6', 'same v0 export, test-time action depth 6 (core loops 1,1,1,2,3,4)'),
    ('loopwam_partition_p0c6s6', 'LoopWAM v0 partition 0/6/6', 'HF `partition-grid-centered-20261026/p0c6s6_k4`'),
    ('loopwam_partition_p1c6s5', 'LoopWAM v0 partition 1/6/5', 'HF `partition-grid-centered-20261026/p1c6s5_k4`'),
    ('loopwam_partition_p5c6s1', 'LoopWAM v0 partition 5/6/1', 'HF `partition-grid-centered-20261026/p5c6s1_k4`'),
    ('loopwam_partition_p6c6s0', 'LoopWAM v0 partition 6/6/0', 'HF `partition-grid-centered-20261026/p6c6s0_k4`'),
    ('loopwam_partition_p1c9s2_k3', 'LoopWAM v0 partition 1/9/2 k3', 'HF `partition-grid-centered-20261026/p1c9s2_k3` (3 loops)'),
    ('loopwam_partition_p4c3s5_k7', 'LoopWAM v0 partition 4/3/5 k7', 'HF `partition-grid-centered-20261026/p4c3s5_k7` (7 loops)'),
    ('loopwam_init_core16_21', 'LoopWAM v0 init core16_21', 'HF `libero-all-suites/init_variants/core16_21`'),
]


def pct(x):
    return f'{x:.2f}'


ver = {r: json.loads((HERE / 'verification' / f'{r}.json').read_text()) for r, _, _ in RUNS}
spread = {(row['model'], row['suite']): float(row['seed_sample_std'])
          for row in csv.DictReader(open(HERE / 'data' / 'suite_results.csv'))}

# Four-suite matched results reported on other branches: the all-models table of the elastic latency report.
all_models = (BR / 'elastic/plans/performance/LoopWAM_All_Models_Latency_20261030.md').read_text()
section = all_models.split('## 5. All models', 1)[1].split('\n## ', 1)[0]
external, not_evaluated = [], []
for line in section.splitlines():
    cells = [c.strip() for c in line.strip().strip('|').split('|')]
    if len(cells) != 13 or cells[0] in ('Group', '---'):
        continue
    group, model, _, config, sr = cells[:5]
    label = model if config == 'native' else f'{model} {config}'
    if sr == 'not evaluated':
        not_evaluated.append(label)
    elif re.fullmatch(r'\d+\.\d+', sr):
        external.append((float(sr), label, group))

rows = [(ver[r]['success_rate'], label, 'this evaluation (local RTX PRO 6000 Blackwell)') for r, label, _ in RUNS]
rows += [(sr, label, f'H100, branch `elastic` all-models report ({group})') for sr, label, group in external]
rows.sort(key=lambda x: -x[0])

L = []
w = L.append
w('# Matched LIBERO evaluation: public WAM baselines and LoopWAM exports (updated 2026-10-30)')
w('')
w('Every number here uses the campaign protocol `lightwam_20261009_matched`: 4 suites x 10 tasks x official initial '
  'states 0-49 x evaluation seeds 42/43/44 = **6,000 episodes per model** (1,500 per suite, 2,000 per seed).')
w('')
w('- **This evaluation:** 14 models evaluated on a local 2x RTX PRO 6000 Blackwell host (eager native inference), '
  'each re-verified on 2026-10-30 against its full 6,000-episode grid (Section 5). This is where FasterWAM, Light-WAM, '
  'Dense-S12/S30, the v0 test-time action-depth sweep, the six centered partition-grid exports and the `core16_21` '
  'initialization variant were scored.')
w('- **Other branches (pulled 2026-10-30):** every four-suite matched result reported on the other branches '
  '(Elastic, Residual Transport, ChronoLoop, init variants, campaign rows), as reported there; the '
  'documents themselves are in [`../branch_reports`](../branch_reports).')
w('- FastWAM was scored by the campaign (`evaluation_report/final`). LiLa-WAM has no matched result (Section 4).')
w('')
w('## 1. All four-suite matched results (6,000 episodes each), highest first')
w('')
w('| # | Model | Success % | Evaluated |')
w('|---:|---|---:|---|')
for i, (sr, label, src) in enumerate(rows, 1):
    bold = '**' if 'this evaluation' in src else ''
    w(f'| {i} | {bold}{label}{bold} | {bold}{pct(sr)}{bold} | {src} |')
w('')
w('Bold rows are this evaluation. With 6,000 episodes the binomial 95% interval is about +/-0.6 points for a model '
  'near 95% success (+/-0.8 near 90%); episodes within a task/seed group share the seed, so treat it as a lower '
  'bound. Rows from other branches are copied from their reports and were not re-verified here.')
w('')
w('## 2. This evaluation, per suite')
w('')
w('Success % with the sample standard deviation over the three seeds; per-seed columns average all four suites.')
w('')
w('| Model | Spatial | Object | Goal | Long | **All** | Seed 42 | Seed 43 | Seed 44 | What was evaluated |')
w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
for r, label, what in sorted(RUNS, key=lambda x: -ver[x[0]]['success_rate']):
    v = ver[r]
    cells = [f"{v['suites'][s]['rate']:.2f} ± {spread[(label, s)]:.2f}" for s in SUITES]
    seeds = [f"{v['seeds'][s]:.2f}" for s in ('42', '43', '44')]
    w(f"| {label} | {' | '.join(cells)} | **{pct(v['success_rate'])}** | {' | '.join(seeds)} | {what} |")
w('')
w('For reference, the campaign scored the same v0 export at action depth 4 (its trained depth, "LoopWAM full 4/4") '
  'at **94.92%** (96.93 / 98.73 / 96.13 / 87.87), and FastWAM at 96.63% (`evaluation_report/final`).')
w('')
w('Per-task results, McNemar paired tests against the campaign\'s LoopWAM full 4/4 and every episode are in '
  '[`data/`](data).')
w('')
w('## 3. Gaps these results fill')
w('')
w('The `elastic` all-models latency report (copied in `../branch_reports/elastic/plans/performance/`) lists these '
  'models as "not evaluated"; their matched results are now:')
w('')
fills = {'all-suites Dense-S12': 'loopwam_dense_s12', 'all-suites Dense-S30': 'loopwam_dense_s30',
         'partition p0c6s6_k4': 'loopwam_partition_p0c6s6', 'partition p1c6s5_k4': 'loopwam_partition_p1c6s5',
         'partition p5c6s1_k4': 'loopwam_partition_p5c6s1', 'partition p6c6s0_k4': 'loopwam_partition_p6c6s0',
         'partition p1c9s2_k3': 'loopwam_partition_p1c9s2_k3', 'partition p4c3s5_k7': 'loopwam_partition_p4c3s5_k7',
         'init variant core16_21': 'loopwam_init_core16_21'}
w('| Model (as named there) | Matched success % (this evaluation) | Long % |')
w('|---|---:|---:|')
for name in not_evaluated:
    r = fills.get(name)
    if r:
        w(f"| {name} | {pct(ver[r]['success_rate'])} | {ver[r]['suites']['libero_10']['rate']:.2f} |")
    else:
        w(f'| {name} | not evaluated | |')
w('')
w('`core16_21` is also the one variant marked "not evaluated" in `LoopWAM_v0_Init_Variants_Results_20261028.md` '
  '(branch `v0-init-variants`).')
w('')
w('## 4. LiLa-WAM')
w('')
w('LiLa-WAM has no matched LIBERO result. Its only public weights are a RoboTwin 2.0 model (one head camera, '
  'dual-arm state and actions), which cannot run LIBERO; its RoboTwin latency is in '
  '`../branch_reports/elastic/plans/performance/Public_WAM_Latency_20261030.md`. A LiLa-WAM trained on our LIBERO data '
  'with the paper recipe was evaluated on branch `elastic` and then withdrawn there (commit `e88defe`, 2026-10-30: '
  '"Remove LiLa-WAM LIBERO training and evaluation results"), so it is not reported here.')
w('')
w('## 5. Checkpoints and verification (this evaluation)')
w('')
w('`matched_eval/verify_run.py` (in `harness/matched_eval_source.tar.gz`) re-read every run on 2026-10-30: the full '
  'grid of 6,000 episode files (each file name vs its contents, initial state = episode index, checkpoint sha256), '
  'the 120 atomically written task/seed group files against the episode files, the manifest protocol, every model '
  'build report (checkpoint sha256, loop schedule, NFE 10, CFG 1, 30 settling steps, replan 10, nothing left unloaded, '
  'strict loads for LoopWAM), and the summary totals. Outputs: [`verification/`](verification).')
w('')
w('| Model | Checkpoint file (under the eval root) | sha256 | Action core schedule | Builds | Verified |')
w('|---|---|---|---|---:|---|')
for r, label, _ in RUNS:
    v = ver[r]
    builds = json.loads((HERE / 'runs' / r / 'build_reports.json').read_text())
    files = sorted({b['checkpoint'].replace('/data/anh/code/wam_baselines/', '') for b in builds.values()})
    shas = sorted(set(v['checkpoint_sha256'].values()))
    loops = '-' if v['action_core_video_loops'] is None else ','.join(map(str, v['action_core_video_loops']))
    status = f"passed ({v['episodes']} episodes)" if v['passed'] else 'FAILED: ' + '; '.join(v['issues'][:3])
    w(f"| {label} | {'<br>'.join(f'`{f}`' for f in files)} | {'<br>'.join(f'`{s[:16]}…`' for s in shas)} | {loops} "
      f"| {v['builds']} | {status} |")
w('')
w((HERE / 'README_body.md').read_text().rstrip())
w('')
(HERE / 'README.md').write_text('\n'.join(L) + '\n')
print(f'{len(rows)} rows, {len(not_evaluated)} not-evaluated names')
