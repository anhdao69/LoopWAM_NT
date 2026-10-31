"""Read-only audit of frozen RoboTwin evidence; write report exports only."""
import collections
import concurrent.futures
import csv
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

B = Path('/data14t/anhdh35/loopwam_robotwin_v0_20261030')
ROOT = Path('/storage/anhdh35/LoopWAM-robotwin-eval')
REPO = Path('/storage/anhdh35/LoopWAM_NT')
R = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()

def main():
    started = time.time()
    spec = importlib.util.spec_from_file_location('frozen_evaluator', ROOT/'scripts/evaluate_loopwam_robotwin.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    protocol = json.loads((B/'results/manifest.json').read_text())
    complete = json.loads((B/'results/COMPLETE.json').read_text())
    summary = json.loads((B/'results/summary.json').read_text())
    launch = json.loads((B/'launch.json').read_text())
    original = json.loads((B/'recovery_20261031/launch.json').read_text())
    assert launch['status'] == 'complete'
    assert complete['finished'] == summary['finished']
    assert not complete['running'] and not complete['failed'] and complete['pending'] == 0
    evaluator.validate_eval_summary(complete, protocol['checkpoint_sha256'], protocol['tasks'], 100)
    assert sha(protocol['checkpoint']) == protocol['checkpoint_sha256']
    for path, expected in protocol['runtime_hashes'].items():
        assert sha(path) == expected, ('source drift', path)
    print('Checkpoint and frozen source hashes passed', flush=True)
    runtime = json.loads((B/'runtime_manifest.json').read_text())
    for path, info in runtime['files'].items():
        assert sha(path) == info['sha256'], ('runtime drift', path)
    preserved = json.loads((B/'recovery_20261031/preserved_result_hashes.json').read_text())
    assert len(preserved) == 95
    for path, expected in preserved.items():
        assert sha(path) == expected, ('preserved result changed', path)
    assert (B/'results/manifest.json').read_bytes() == (B/'recovery_20261031/manifest.json').read_bytes()
    print('Runtime assets and all 95 preserved results passed', flush=True)

    def verify(item):
        key, cell = item
        records = evaluator.completed_records(Path(cell['directory']), cell['config'], 100)
        assert records == cell['episodes'], ('episode trace/video mismatch', key)
        assert cell['successes'] == sum(x['success'] for x in records)
        result = json.loads((Path(cell['directory']).parent/'result.json').read_text())
        assert result == cell
        times = []
        for line in (Path(cell['directory'])/'eval.log').read_text(errors='replace').splitlines():
            if line.startswith('{"policy_query"'):
                times.append(json.loads(line)['seconds'])
        assert len(times) == sum(x['queries'] for x in records)
        return key, times

    query_times = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, (key, times) in enumerate(pool.map(verify, complete['finished'].items()), 1):
            query_times.extend(times)
            if index % 10 == 0:
                print(f'Validated {index}/100 pairs including videos and action traces', flush=True)

    attempts = []
    for path in sorted((B/'results').glob('*/attempt_*')):
        if not path.is_dir():
            continue
        failure = path/'failure.json'
        if failure.exists():
            text = (path/'eval.log').read_text(errors='replace')
            reason = ('cuda_out_of_memory' if 'OutOfMemoryError' in text else
                      'render_buffer_allocation' if 'cannot create buffer' in text else 'other')
            outcome = json.loads(failure.read_text())
            attempts.append(dict(path=str(path), reason=reason, **outcome))

    rows = []
    episodes = []
    for task in protocol['tasks']:
        for setting in protocol['settings']:
            cell = complete['finished'][f'{task}/{setting}']
            rows.append(dict(task=task, setting=setting, successes=cell['successes'], episodes=100,
                             sr_percent=cell['successes'], accepted_attempt=Path(cell['directory']).name,
                             attempt_seconds=cell['seconds'], directory=cell['directory']))
            for index, ep in enumerate(cell['episodes']):
                episodes.append(dict(task=task, setting=setting, episode_index=index,
                                     accepted_attempt=Path(cell['directory']).name, **ep))
    for name, data in [('task_results.csv', rows), ('episodes.csv', episodes), ('failed_attempts.csv', attempts)]:
        with (R/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    totals = {}
    for setting in protocol['settings']:
        cells = [v for v in rows if v['setting'] == setting]
        total = sum(v['successes'] for v in cells)
        totals[setting] = dict(successes=total, episodes=5000, sr_percent=total/50)
        assert total/5000 == complete['success_rates'][setting]
    total = sum(v['successes'] for v in rows)
    totals['combined'] = dict(successes=total, episodes=10000, sr_percent=total/100)
    assert total == 9173
    times = sorted(query_times)
    def percentile(p):
        return times[round((len(times)-1)*p)]
    audit = dict(status='passed', audited_unix=time.time(), duration_seconds=time.time()-started,
                 checkpoint_sha256=protocol['checkpoint_sha256'], complete_sha256=sha(B/'results/COMPLETE.json'),
                 protocol_sha256=sha(B/'results/manifest.json'), source_files_verified=len(protocol['runtime_hashes']),
                 runtime_files_verified=len(runtime['files']), completed_pairs=100, verified_videos=10000,
                 episode_records=10000, preserved_result_files=95, totals=totals,
                 failed_attempts=len(attempts), failure_reasons=dict(collections.Counter(x['reason'] for x in attempts)),
                 excluded_partial_episodes=sum(x['episodes'] for x in attempts),
                 accepted_attempt_distribution=dict(collections.Counter(x['accepted_attempt'] for x in rows)),
                 recovery_new_failures=sum(Path(x['path']).name in ('attempt_04','attempt_05') for x in attempts),
                 timing=dict(original_started=original['started_unix'], original_stopped=original['stopped_unix'],
                             recovery_started=launch['started_unix'], finished=launch['finished_unix'],
                             initial_seconds=original['stopped_unix']-original['started_unix'],
                             downtime_seconds=launch['started_unix']-original['stopped_unix'],
                             recovery_seconds=launch['finished_unix']-launch['started_unix'],
                             total_wall_seconds=launch['finished_unix']-original['started_unix']),
                 query_latency=dict(count=len(times), mean_seconds=sum(times)/len(times),
                                    median_seconds=percentile(.5), p95_seconds=percentile(.95),
                                    scope='Scored successful attempts, mixed worker concurrency; not isolated model latency'),
                 source=dict(main_commit=git('rev-parse','HEAD'), main_remote=git('rev-parse','origin/LoopWAM_NT'),
                             evaluated_commit=launch['code_commit'],
                             robotwin_upstream_commit=git('rev-parse','origin/robotwin_eval_continue_20261030')))
    (R/'final_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (R/'final_results.json').write_text(json.dumps(dict(totals=totals, tasks=rows), indent=2)+'\n')
    print(json.dumps(audit, indent=2), flush=True)

if __name__ == '__main__':
    main()
