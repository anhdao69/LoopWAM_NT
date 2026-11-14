"""Reconcile saved RT-10 epoch-10 results and render the requested report.

Only derived files in this directory are written. Source result records are read-only.
v0 4/4 is excluded from this update. No evaluation or GPU work is launched.
"""
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SPECS = [
    ('v0_4x2', 'v0 4/2', 'robotwin10_epoch10_queue_20261113/v0_4x2', 'Local RTX PRO 6000', True),
    ('v0_4x1', 'v0 4/1', 'robotwin10_v0_4x1_epoch10_20261108', 'Local RTX PRO 6000', True),
    ('v0_2x2', 'v0 2/2', 'robotwin10_epoch10_queue_20261113/v0_2x2', 'Local RTX PRO 6000', True),
    ('v0_1x4', 'v0 1/4', 'robotwin10_epoch10_remote189_20261113/v0_1x4', 'aLinh RTX PRO 5000', True),
    ('dense_s12_latest', 'Dense-12 — latest continuation', 'robotwin10_epoch10_remote189_20261113/dense_s12', '400 local + 1,600 aLinh episodes', True),
    ('dense_s12_previous', 'Dense-12 — earlier local run', 'robotwin10_dense_s12_epoch10_20261109', 'Local RTX PRO 6000', False),
]
TASKS = ['adjust_bottle', 'grab_roller', 'hanging_mug', 'move_stapler_pad', 'open_microwave',
         'press_stapler', 'scan_object', 'stack_blocks_two', 'stamp_seal', 'turn_switch']
SETTINGS = ['demo_clean', 'demo_randomized']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_read(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def csv_write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def reconcile(spec):
    ident, label, relative, host, latest = spec
    directory = ROOT / relative
    names = ['COMPLETE.json', 'final_audit.json', 'final_results.json', 'planned_protocol.json',
             'cpu_validation.json', 'launch.json', 'failed_attempts.json', 'episodes.csv', 'task_results.csv']
    complete, audit, final, protocol, cpu, launch, failures = [read(directory / n) for n in names[:7]]
    assert audit['status'] == 'passed' and audit['verified_pairs'] == 20 and audit['verified_videos'] == 2000
    assert audit['renderer_logs_clean'] and sha(directory / 'COMPLETE.json') == audit['complete_sha256']
    checkpoint = audit['checkpoint_sha256']
    assert complete['checkpoint_sha256'] == checkpoint
    assert len(complete['finished']) == 20 and not complete['running'] and not complete['failed'] and complete['pending'] == 0
    assert cpu['epoch'] == protocol['epoch'] == launch['epoch'] == 10
    assert cpu['step'] == protocol['expected_step'] == 90890 and cpu['strict_load']
    assert cpu['variant'] == protocol['variant'] == launch['variant']
    assert protocol['variant'] != 'v0-video4-action4' and ident != 'v0_4x4'
    assert protocol['tasks'] == TASKS and protocol['settings'] == SETTINGS
    assert protocol['seed'] == 42 and protocol['episodes'] == 100
    assert protocol['num_inference_steps'] == 10 and protocol['action_horizon'] == 32 and protocol['replan_steps'] == 24
    assert launch['status'] == 'complete'
    assert len(failures) == audit['failed_attempts']
    assert sum(x['episodes'] for x in failures) == audit['excluded_partial_episodes']
    expected_keys = {(t, s) for t in TASKS for s in SETTINGS}
    final_rows = {(x['task'], x['setting']): x for x in final['tasks']}
    assert len(final['tasks']) == 20 and set(final_rows) == expected_keys
    assert set(complete['finished']) == {f'{t}/{s}' for t, s in expected_keys}
    task_csv = csv_read(directory / 'task_results.csv')
    assert len(task_csv) == 20 and {(x['task'], x['setting']) for x in task_csv} == expected_keys
    for row in task_csv:
        assert row == {k: str(v) for k, v in final_rows[(row['task'], row['setting'])].items()}
    episodes_csv = csv_read(directory / 'episodes.csv')
    episodes = {(x['task'], x['setting'], int(x['episode_index'])): x for x in episodes_csv}
    assert len(episodes_csv) == len(episodes) == 2000
    for task, setting in expected_keys:
        cell = complete['finished'][f'{task}/{setting}']
        assert cell['complete'] and cell['exit_code'] == 0 and cell['checkpoint_sha256'] == checkpoint
        assert cell['task'] == task and cell['config'] == setting
        assert len(cell['episodes']) == len({e['seed'] for e in cell['episodes']}) == 100
        for i, ep in enumerate(cell['episodes']):
            assert type(ep['success']) is bool and ep['video_verified'] and ep['queries'] > 0
            assert ep['instruction'] and len(ep['video_sha256']) == len(ep['query_trace_sha256']) == 64
            record = dict(task=task, setting=setting, episode_index=i, **ep)
            assert episodes[(task, setting, i)] == {k: str(v) for k, v in record.items()}
        count = sum(e['success'] for e in cell['episodes'])
        row = final_rows[(task, setting)]
        assert count == cell['successes'] == row['successes'] == row['sr_percent']
        assert row['episodes'] == 100 and row['seconds'] == cell['seconds']
        assert row['accepted_attempt'] == Path(cell['directory']).name
    totals = {}
    for setting in SETTINGS + ['combined']:
        cells = [v for v in complete['finished'].values() if setting == 'combined' or v['config'] == setting]
        n = sum(len(v['episodes']) for v in cells)
        s = sum(sum(e['success'] for e in v['episodes']) for v in cells)
        totals[setting] = dict(successes=s, episodes=n, sr_percent=100 * s / n)
        assert totals[setting] == final['totals'][setting]
        if setting != 'combined':
            assert complete['success_rates'][setting] == s / n
    per_task = []
    for task in TASKS:
        clean, randomized = (final_rows[(task, s)]['successes'] for s in SETTINGS)
        per_task.append(dict(task=task, clean_successes=clean, clean_episodes=100, clean_sr_percent=float(clean),
                             randomized_successes=randomized, randomized_episodes=100, randomized_sr_percent=float(randomized),
                             combined_successes=clean + randomized, combined_episodes=200,
                             combined_sr_percent=(clean + randomized) / 2))
    return dict(id=ident, label=label, source_directory=relative, host=host, primary_run=latest,
                epoch=10, update=90890, checkpoint_sha256=checkpoint, protocol=protocol, launch=launch,
                audit=audit, totals=totals, tasks=per_task,
                source_sha256={n: sha(directory / n) for n in names})


def render(runs, checked):
    primary = [r for r in runs if r['primary_run']]
    parts = [f'''# RoboTwin-10 epoch 10: detailed task results

Updated **{checked}** (Vietnam time). Scope: **v0 4/2, v0 4/1, v0 2/2, v0 1/4 and Dense-12**. **v0 4/4 is excluded from this update; its existing results and files are unchanged.**

Five model versions and six completed evaluations are documented. The latest Dense-12 continuation is used in the five-model task comparison, and the earlier Dense-12 evaluation is retained separately. All six runs have passing saved final audits. This report freshly reconciles **12,000 accepted episode records**, 120 task/setting cells, both CSV exports, aggregate JSON results and completion-file hashes. No simulation or video-content audit was rerun for this documentation update.

## Overall scores

Every row uses seed **42**, **NFE 10**, and **2,000 accepted episodes**: 10 tasks × clean/randomized × 100 episodes. Clean and randomized denominators are 1,000 each.
''']
    summary = []
    for run in runs:
        t = run['totals']
        summary.append([f"[{run['label']}](../{run['source_directory']}/final_report.md)",
            f"{t['demo_clean']['successes']}/1000 ({t['demo_clean']['sr_percent']:.2f}%)",
            f"{t['demo_randomized']['successes']}/1000 ({t['demo_randomized']['sr_percent']:.2f}%)",
            f"{t['combined']['successes']}/2000", f"**{t['combined']['sr_percent']:.2f}%**", run['host']])
    parts.append(table(['Model / evaluation', 'Clean', 'Randomized', 'All successes', 'Overall SR', 'Execution'], summary))
    parts.append('''
The two Dense-12 rows use the same epoch-10 checkpoint SHA256. They are separate evaluations; their episodes are not pooled, and tasks are not selected from whichever run scored better. “Latest” follows execution history, not score ranking. No completed Dense-30 epoch-10 evaluation was found in the saved RT-10 result inventory; its epoch-5 result is not substituted here.

## Task comparison — latest completed run per included model

Each entry below is the task's combined SR across **200 episodes** (100 clean + 100 randomized). The final row is the full 2,000-episode SR. Equal task denominators make the mean of task SRs equal to overall SR.
''')
    rows = [[task] + [f"{r['tasks'][i]['combined_sr_percent']:.2f}%" for r in primary] for i, task in enumerate(TASKS)]
    rows.append(['**All 10 tasks**'] + [f"**{r['totals']['combined']['sr_percent']:.2f}%**" for r in primary])
    parts.append(table(['Task'] + [r['label'] for r in primary], rows))
    parts.append('''
## Detailed clean and randomized results

Setting SRs are shown explicitly with their denominators. Each model's combined task SR is `(clean successes + randomized successes) / 200 × 100`; it is not a score from a separate set of episodes.
''')
    for run in runs:
        parts.append(f"### {run['label']}\n\nEpoch 10 / update 90,890; {run['host']}. "
                     f"[Original final report](../{run['source_directory']}/final_report.md) · "
                     f"[Task CSV](../{run['source_directory']}/task_results.csv) · "
                     f"[Episode CSV](../{run['source_directory']}/episodes.csv) · "
                     f"[Passed audit](../{run['source_directory']}/final_audit.json).")
        rows = [[x['task'], f"{x['clean_successes']}/100", f"{x['clean_sr_percent']:.2f}%",
                 f"{x['randomized_successes']}/100", f"{x['randomized_sr_percent']:.2f}%",
                 f"{x['combined_successes']}/200", f"{x['combined_sr_percent']:.2f}%"] for x in run['tasks']]
        t = run['totals']
        rows.append(['**Total**', f"{t['demo_clean']['successes']}/1000", f"{t['demo_clean']['sr_percent']:.2f}%",
                     f"{t['demo_randomized']['successes']}/1000", f"{t['demo_randomized']['sr_percent']:.2f}%",
                     f"{t['combined']['successes']}/2000", f"**{t['combined']['sr_percent']:.2f}%**"])
        parts.append(table(['Task', 'Clean successes', 'Clean SR', 'Randomized successes', 'Randomized SR',
                            'Combined successes', 'Combined SR'], rows))
    parts.append('''
## Protocol, provenance and exclusions

All included exports are trained for **10 epochs / 90,890 optimizer updates** on the ten-task RoboTwin set. Core settings match: seed42; unseen instructions; NFE10; predict32 / execute24; original 14-D state/action normalization and task horizons; original scene admission; FP32 policy weights/BF16 compute; shared T5 and CUDA graphs; original camera preprocessing and rendering. The 4/2, 4/1, 2/2 and 1/4 labels describe video/action loop counts, not NFE or chunk length. Dense-12 is the dense 12-block architecture.

The accepted denominators exclude scene-admission failures, smoke/calibration episodes, and failed or interrupted attempts. The final audits retain those exclusions:
''')
    parts.append(table(['Evaluation', 'Saved audit', 'Accepted cells', 'Scored episodes', 'Excluded attempts',
                        'Excluded logged partial episodes', 'Workers in final controller'],
                       [[r['label'], 'Passed', '20/20', 2000, r['audit']['failed_attempts'],
                         r['audit']['excluded_partial_episodes'], r['launch']['workers']] for r in runs]))
    parts.append('''
- **Dense-12 latest:** 400 fully verified episodes (`adjust_bottle` and `grab_roller`, both settings) originated locally and were transferred to aLinh. The other 1,600 were evaluated on aLinh. Eight interrupted local attempts / 266 logged episodes remain excluded.
- **v0 1/4:** all 2,000 accepted episodes were generated on aLinh. On resume, 800 already completed episodes were retained, and the remaining 1,200 were evaluated. Two interrupted `open_microwave` attempts / 112 logged episodes remain excluded.
- **v0 4/1:** the accepted final result contains 2,000 episodes. Its saved exclusion ledger records 12 failed/interrupted attempts and 486 logged partial episodes following earlier resource failures. Those attempts do not contribute to the displayed SR.
- **4/2, 2/2 and the earlier Dense-12:** their saved audits record zero failed attempts and zero excluded partial episodes.

Final-controller worker counts do not describe every earlier attempt. Preserved launches may describe only the last execution segment, so their durations should not be treated as total time across stops, retries and migration.

The original Python instruction-template RNG is unseeded. Equal seed42 and matched benchmark settings do not guarantee identical scenes/instructions across independent runs, and hosts/concurrency differ. These are observed aggregate scores, not claims of perfectly paired rollouts or statistically significant differences.

### Checkpoint identity
''')
    parts.append(table(['Evaluation', 'Native variant', 'Epoch / update', 'Checkpoint SHA256'],
                       [[r['label'], r['protocol']['variant'], '10 / 90890', f"`{r['checkpoint_sha256']}`"] for r in runs]))
    parts.append('''
## Verification and exports

This update recalculated success counts from all 12,000 saved episode flags and checked them against task/episode CSVs and final JSON totals. It verified each COMPLETE.json SHA256 against its passed audit, checked 20 unique task/setting cells per run and 100 unique accepted scene seeds within each cell, and verified the epoch/update, native variant, seed, NFE, action horizon and exclusion counts. It also checked that the shared protocol fields match across the six records.

This is a fresh report/record reconciliation. Media bytes, simulator packages and GPU numerical gates were not re-audited here; their verification is the scope of each saved final audit. All original completion records, audits, per-episode CSVs and task CSVs remain unchanged. The v0 4/4 epoch-10 directory is hash-protected and is outside the updated result set.

- [Overall results CSV](summary.csv): six evaluation rows, with run identity and checkpoint hashes.
- [Detailed task CSV](task_results.csv): 60 rows, one row per task per evaluation, with separate clean/randomized/combined counts and SR.
- [Latest-run task comparison CSV](task_comparison.csv): ten rows covering the five included model versions.
- [Full JSON results and input hashes](results.json).
- [Reconciliation receipt](evidence/reconciliation.json) and [final report verification](evidence/report_validation.json).
- [Reconciliation/report generator](reconcile_and_render.py); it reads source records and writes only derived files in this report directory.

The main [results catalogue](../all_results.md) and [report index](../README.md) link this dated update. Their older exports and other benchmark sections retain their original scope. Prior Markdown files are backed up under `evidence/before_update/`.
''')
    (OUT / 'README.md').write_text('\n\n'.join(s.strip() for s in parts) + '\n')


def main():
    runs = [reconcile(spec) for spec in SPECS]
    common = ['tasks', 'settings', 'episodes', 'seed', 'instruction_type', 'num_inference_steps',
              'action_horizon', 'replan_steps', 'noise_seed', 'image_preprocessing', 'instruction_rng',
              'cuda_graph', 'compile_action_infer', 'renderer_shader', 'stats_sha256', 'vae_sha256']
    for key in common:
        assert all(r['protocol'][key] == runs[0]['protocol'][key] for r in runs), key
    assert runs[4]['checkpoint_sha256'] == runs[5]['checkpoint_sha256']
    checked = datetime.now(timezone(timedelta(hours=7))).isoformat(timespec='seconds')
    full = dict(updated=checked, benchmark='RoboTwin-10', excluded_model='v0 4/4', seed=42,
                nfe=10, episodes_per_run=2000, primary_run_ids=[r['id'] for r in runs if r['primary_run']], runs=runs)
    (OUT / 'results.json').write_text(json.dumps(full, indent=2) + '\n')
    summaries, task_rows = [], []
    for r in runs:
        row = dict(run_id=r['id'], model=r['label'], epoch=10, optimizer_update=90890,
                   primary_run=r['primary_run'], host=r['host'], seed=42, nfe=10)
        for setting, prefix in [('demo_clean', 'clean'), ('demo_randomized', 'randomized'), ('combined', 'combined')]:
            row.update({prefix + '_' + k: v for k, v in r['totals'][setting].items()})
        row.update(checkpoint_sha256=r['checkpoint_sha256'], audit_status='passed', source_directory=r['source_directory'])
        summaries.append(row)
        task_rows.extend(dict(run_id=r['id'], model=r['label'], primary_run=r['primary_run'], **task) for task in r['tasks'])
    comparisons = []
    for i, task in enumerate(TASKS):
        row = dict(task=task)
        row.update({r['id'] + '_sr_percent': r['tasks'][i]['combined_sr_percent'] for r in runs if r['primary_run']})
        comparisons.append(row)
    csv_write(OUT / 'summary.csv', summaries)
    csv_write(OUT / 'task_results.csv', task_rows)
    csv_write(OUT / 'task_comparison.csv', comparisons)
    evidence = dict(status='passed', checked=checked, runs=6, primary_models=5, task_setting_cells=120,
                    episode_records_reconciled=12000, common_protocol_fields_checked=common,
                    source_files={r['source_directory'] + '/' + n: digest for r in runs for n, digest in r['source_sha256'].items()},
                    original_video_audits_repeated=False, simulations_run=False, excluded_model='v0 4/4')
    (OUT / 'evidence/reconciliation.json').write_text(json.dumps(evidence, indent=2) + '\n')
    render(runs, checked)
    print(json.dumps({'status': 'passed', 'runs': 6, 'episodes': 12000,
                      'totals': {r['id']: r['totals']['combined'] for r in runs}}, indent=2))


if __name__ == '__main__':
    main()
