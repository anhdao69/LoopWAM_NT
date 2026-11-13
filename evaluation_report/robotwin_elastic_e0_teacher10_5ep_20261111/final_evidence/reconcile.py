"""Recompute final RT-50 E0 comparison from immutable saved episode records.

Only writes derived final_comparison.json / task_comparison.csv in this report.
Run after recheck_host.py has refreshed local_recheck.json and remote137_recheck.json.
"""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT.parent
CHECKPOINT = '07d24ff77410e516ac90d24d76d651e2102b09b0458d9d5d86ea18dff39e823e'
OLD_CHECKPOINT = 'abc3c0075a6e1f3ce380be8d6c7b6b9d8bb697ac21cf3f2d40a93efb821b539a'
PATHS = {
    'flat_k1_n2': ROOT / 'local/flat_k1_n2',
    'flat_k2_n2': ROOT / 'remote137/flat_k2_n2',
    'flat_k4_n2': INDEX / 'robotwin_elastic_e0_teacher10_5ep_44_20261112/flat_k4_n2',
}
OLD_PATHS = {
    'flat_k1_n2': INDEX / 'robotwin_elastic_e0_20261107/flat_k1_n2',
    'flat_k4_n2': INDEX / 'robotwin_elastic_e0_20261107/parallel137_20261108/remote_mirror/flat_k4_n2',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def reconcile(path, schedule, checkpoint):
    c, m, a, f = [read(path / name) for name in
                  ['COMPLETE.json', 'manifest.json', 'final_audit.json', 'final_results.json']]
    assert a['status'] == 'passed' and a['verified_pairs'] == 100 and a['verified_videos'] == 10000
    assert a['renderer_logs_clean'] and a['complete_sha256'] == sha(path / 'COMPLETE.json')
    assert c['checkpoint_sha256'] == m['checkpoint_sha256'] == a['checkpoint_sha256'] == checkpoint
    assert not c['running'] and not c['failed'] and c['pending'] == 0 and len(c['finished']) == 100
    assert m['elastic_schedule'] == schedule and m['num_inference_steps'] == 2
    assert m['action_depths'] == [int(schedule[6]), int(schedule[6])]
    assert m['seed'] == 42 and m['episodes'] == 100 and len(set(m['tasks'])) == 50
    assert m['settings'] == ['demo_clean', 'demo_randomized']
    assert m['action_horizon'] == 32 and m['replan_steps'] == 24 and m['video_loops'] == 4
    expected = {(t, s) for t in m['tasks'] for s in m['settings']}
    assert set(c['finished']) == {f'{t}/{s}' for t, s in expected}
    result_rows = {(r['task'], r['setting']): r for r in f['tasks']}
    assert len(f['tasks']) == 100 and set(result_rows) == expected
    with (path / 'task_results.csv').open() as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 100
    assert {tuple((r['task'], r['setting'])) for r in csv_rows} == expected
    for row in csv_rows:
        assert row == {k: str(v) for k, v in result_rows[(row['task'], row['setting'])].items()}
    with (path / 'episodes.csv').open() as handle:
        episode_csv = list(csv.DictReader(handle))
    assert len(episode_csv) == 10000
    by_key = {(r['task'], r['setting'], int(r['episode_index'])): r for r in episode_csv}
    assert len(by_key) == 10000
    for (task, setting) in expected:
        cell = c['finished'][f'{task}/{setting}']
        assert cell['complete'] and cell['exit_code'] == 0 and cell['checkpoint_sha256'] == checkpoint
        assert cell['task'] == task and cell['config'] == setting and len(cell['episodes']) == 100
        assert len({ep['seed'] for ep in cell['episodes']}) == 100
        for i, ep in enumerate(cell['episodes']):
            assert type(ep['success']) is bool and ep['video_verified'] is True
            assert ep['queries'] > 0 and ep['instruction'] and ep['seed'] >= 4300000
            assert re.fullmatch(r'[0-9a-f]{64}', ep['video_sha256'])
            assert re.fullmatch(r'[0-9a-f]{64}', ep['query_trace_sha256'])
            want = dict(task=task, setting=setting, episode_index=i, **ep)
            assert by_key[(task, setting, i)] == {k: str(v) for k, v in want.items()}
        success = sum(ep['success'] for ep in cell['episodes'])
        row = result_rows[(task, setting)]
        assert cell['successes'] == row['successes'] == row['sr_percent'] == success
        assert row['episodes'] == 100 and row['seconds'] == cell['seconds']
        assert row['accepted_attempt'] == Path(cell['directory']).name == 'attempt_01'
    totals = {}
    for setting in m['settings'] + ['combined']:
        cells = [v for v in c['finished'].values() if setting == 'combined' or v['config'] == setting]
        successes = sum(v['successes'] for v in cells)
        count = sum(len(v['episodes']) for v in cells)
        totals[setting] = dict(successes=successes, episodes=count, sr_percent=100 * successes / count)
        assert totals[setting] == f['totals'][setting]
        if setting != 'combined':
            assert c['success_rates'][setting] == successes / count
    assert read(path / 'failed_attempts.json') == []
    assert a['failed_attempts'] == a['excluded_partial_episodes'] == 0
    return dict(path=str(path), totals=totals, audit=a,
                files_sha256={n: sha(path / n) for n in ['COMPLETE.json', 'manifest.json',
                    'final_audit.json', 'final_results.json', 'task_results.csv', 'episodes.csv']},
                tasks=f['tasks'], manifest={k: v for k, v in m.items() if k != 'runtime_hashes'}), c


def main():
    data, records, old = {}, {}, {}
    for schedule, path in PATHS.items():
        data[schedule], records[schedule] = reconcile(path, schedule, CHECKPOINT)
    for schedule, path in OLD_PATHS.items():
        old[schedule], _ = reconcile(path, schedule, OLD_CHECKPOINT)
    fields = ['tasks', 'settings', 'seed', 'episodes', 'instruction_type', 'video_loops',
              'step_condition', 'num_inference_steps', 'action_horizon', 'replan_steps',
              'noise_seed', 'image_preprocessing', 'instruction_rng', 'cuda_graph',
              'compile_action_infer', 'renderer_shader', 'harness_sha256', 'policy_sha256',
              'stats_sha256', 'vae_sha256']
    for field in fields:
        assert all(r['manifest'][field] == data['flat_k1_n2']['manifest'][field] for r in data.values()), field
    host_checks = []
    for filename in ['local_recheck.json', 'remote137_recheck.json']:
        host = read(ROOT / 'final_evidence' / filename)
        assert host['status'] == 'passed'
        host_checks.append(dict(file=filename, sha256=sha(ROOT / 'final_evidence' / filename)))
        for check in host['runs']:
            run = data[check['schedule']]
            assert check['status'] == 'passed' and check['checkpoint_sha256'] == CHECKPOINT
            for original, digest in check['original_artifact_sha256'].items():
                assert run['files_sha256'][Path(original).name] == digest
            assert check['original_final_audit'] == run['audit']
            assert check['smoke']['passed'] and check['smoke']['excluded'] and check['smoke']['episodes'] == 8
            launch = check['launch']
            assert launch['status'] == 'complete' and launch['workers'] == 4
            assert check['active_process_receipt']['exit_code'] == 0
            seconds = launch['finished_unix'] - launch['started_unix']
            run['execution'] = dict(host=host['host'], launch=launch, scoring_wall_seconds=seconds,
                episodes_per_hour=10000 * 3600 / seconds,
                audit_after_process_seconds=run['audit']['audited_unix'] - launch['finished_unix'],
                queue_to_audit_seconds=run['audit']['audited_unix'] - check['queue_launch']['started_unix'],
                cumulative_worker_cell_seconds=sum(x['seconds'] for x in run['tasks']),
                policy_queries=check['policy_queries'], video_bytes=check['video_bytes'])
    tasks = data['flat_k1_n2']['manifest']['tasks']
    table = []
    for task in tasks:
        row = {'task': task}
        for label, runs in [('new', data), ('old', old)]:
            for schedule, run in runs.items():
                values = {r['setting']: r['successes'] for r in run['tasks'] if r['task'] == task}
                prefix = label + '_' + schedule
                row[prefix + '_clean'] = values['demo_clean']
                row[prefix + '_randomized'] = values['demo_randomized']
                row[prefix + '_sr_percent'] = sum(values.values()) / 2
        row['new_22_minus_11_pp'] = row['new_flat_k2_n2_sr_percent'] - row['new_flat_k1_n2_sr_percent']
        row['new_44_minus_11_pp'] = row['new_flat_k4_n2_sr_percent'] - row['new_flat_k1_n2_sr_percent']
        row['new_44_minus_22_pp'] = row['new_flat_k4_n2_sr_percent'] - row['new_flat_k2_n2_sr_percent']
        for k in [1, 4]:
            row[f'new_minus_old_k{k}_pp'] = row[f'new_flat_k{k}_n2_sr_percent'] - row[f'old_flat_k{k}_n2_sr_percent']
        table.append(row)
    comparison = {}
    for a, b in itertools.combinations(data, 2):
        deltas = [r[f'new_{b}_sr_percent'] - r[f'new_{a}_sr_percent'] for r in table]
        same_seed = same_instruction = same_both = identical_seed_cells = 0
        for cell_key in records[a]['finished']:
            ae = records[a]['finished'][cell_key]['episodes']; be = records[b]['finished'][cell_key]['episodes']
            identical_seed_cells += [x['seed'] for x in ae] == [x['seed'] for x in be]
            for x, y in zip(ae, be):
                same_seed += x['seed'] == y['seed']
                same_instruction += x['instruction'] == y['instruction']
                same_both += x['seed'] == y['seed'] and x['instruction'] == y['instruction']
        comparison[b + '_minus_' + a] = dict(task_wins=sum(d > 0 for d in deltas),
            task_ties=sum(d == 0 for d in deltas), task_losses=sum(d < 0 for d in deltas),
            same_episode_index_and_seed=same_seed, same_episode_index_and_instruction=same_instruction,
            same_episode_index_seed_and_instruction=same_both, identical_seed_cells=identical_seed_cells,
            total_compared_episode_positions=10000)
    out = dict(status='passed', checked_unix=time.time(), scope='saved records plus fresh host log/trace/source checks',
               checkpoint_sha256=CHECKPOINT, schedules=data, epoch1_baselines=old,
               invariant_manifest_fields=fields, host_evidence=host_checks,
               comparisons=comparison, task_comparison=table,
               videos_rehashed_or_ffprobed_this_update=False,
               saved_full_video_audits_verified_by_complete_hash=True)
    (ROOT / 'final_comparison.json').write_text(json.dumps(out, indent=2) + '\n')
    with (ROOT / 'task_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    print(json.dumps(dict(status=out['status'], new_episodes=30000, baseline_episodes=20000,
                         totals={k: v['totals'] for k, v in data.items()}, comparisons=comparison), indent=2))


if __name__ == '__main__':
    main()
