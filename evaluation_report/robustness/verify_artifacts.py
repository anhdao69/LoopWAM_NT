"""Read-only audit of this frozen campaign; writes only its verification receipt."""
import csv
import hashlib
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    started = time.time()
    state = json.loads((ROOT / 'queue_status.json').read_text())
    plans = {b: json.loads((ROOT / f'{b}_tasks.json').read_text()) for b in ('plus', 'pro')}
    with (ROOT / 'episode_details.csv').open() as f:
        csv_rows = list(csv.DictReader(f))
    csv_index = {(r['model'], r['benchmark'], r['id'], int(r['episode'])): r for r in csv_rows}
    assert len(csv_index) == len(csv_rows) == 16142
    files = {}
    episodes = []
    jobs = []

    def register(path, sha, kind):
        path = str(path)
        if path in files:
            assert files[path][0] == sha, ('Conflicting hash', path)
        else:
            files[path] = (sha, kind)

    for job in state['jobs']:
        assert job['status'] == 'complete'
        out = Path(job['output'])
        manifest = json.loads((out / 'manifest.json').read_text())
        proto = manifest['protocol']
        assert proto['groups'] == plans[job['benchmark']]['groups']
        assert proto['mode'] == 'scored' and proto['workers'] == 4
        register(ROOT / f"{job['benchmark']}_tasks.json", proto['plan_sha256'], 'task_plan')
        for path, sha in proto['code'].items():
            register(path, sha, 'source')
        assert json.loads((out / 'summary.json').read_text()) == job['summary']
        found = []
        for group in proto['groups']:
            register(group['physical_bddl'], group['bddl_sha256'], 'bddl')
            register(group['initial_states_file'], group['initial_states_sha256'], 'initial_states')
            rr = json.loads((out / 'groups' / f"{group['id']}.json").read_text())
            assert len(rr) == group['episodes']
            assert sorted(r['episode'] for r in rr) == list(range(group['episodes']))
            for row in rr:
                assert row['id'] == group['id'] and row['seed'] == 42
                assert row['checkpoint_sha256'] == job['checkpoint_sha256']
                assert row['initial_state_index'] == row['episode']
                assert row['initial_state_sha256'] == group['selected_state_hashes'][row['episode']]
                assert row['description'] == group['description']
                assert row['suite'] == group['suite'] and row['category'] == group['category']
                assert type(row['success']) is bool
                assert 1 <= row['policy_steps'] <= group['max_steps']
                assert row['queries'] == math.ceil(row['policy_steps'] / 10)
                csv_row = csv_index[job['model'], job['benchmark'], row['id'], row['episode']]
                for key, value in row.items():
                    assert csv_row[key] == ('' if value is None else str(value)), (job['id'], row['id'], key)
                episode_file = out / 'episodes' / f"{row['id']}_{row['episode']}.json"
                assert json.loads(episode_file.read_text()) == row
                register(row['video'], row['video_sha256'], 'video')
                register(row['query_log'], row['query_log_sha256'], 'query_log')
                queries = json.loads(Path(row['query_log']).read_text())
                assert len(queries) == row['queries']
                assert [q['query'] for q in queries] == list(range(row['queries']))
                for q in queries:
                    assert math.isfinite(q['seconds']) and q['seconds'] >= 0
                    assert len(q['action_sha256']) == 64
                    int(q['action_sha256'], 16)
                assert math.isclose(sum(q['seconds'] for q in queries), row['query_seconds'])
            found.extend(rr)
        assert len(found) == job['expected_episodes'] == job['summary']['episodes']
        assert sum(r['success'] for r in found) == job['summary']['successes']
        assert sum(r['queries'] for r in found) == job['summary']['total_queries']
        jobs.append(dict(id=job['id'], episodes=len(found), successes=sum(r['success'] for r in found)))
        episodes.extend(found)
    text = json.loads((ROOT / 'text_provenance.json').read_text())
    for name, info in text['files'].items():
        register(Path('/data14t/anhdh35/loopwam_robustness/text') / name, info['sha256'], 'text_embedding')
    bundle = json.loads((ROOT / 'source_bundle.json').read_text())
    register(ROOT / bundle['path'], bundle['sha256'], 'source_bundle')

    def verify(item):
        path, (sha, kind) = item
        if kind != 'source':
            assert Path(path).stat().st_size > 0, ('Empty file', path)
        assert digest(path) == sha, ('Hash mismatch', path)
        return kind, Path(path).stat().st_size

    counts, sizes = Counter(), Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n, (kind, size) in enumerate(pool.map(verify, files.items()), 1):
            counts[kind] += 1
            sizes[kind] += size
            if n % 4000 == 0:
                print(f'Hashed {n}/{len(files)} artifacts', flush=True)
    result = dict(passed=True, started_unix=started, finished_unix=time.time(),
                  jobs=jobs, episodes=len(episodes), queries=sum(r['queries'] for r in episodes),
                  unique_hashed_files=len(files), hashed_files_by_kind=dict(counts),
                  hashed_bytes_by_kind=dict(sizes), seed=42,
                  scope='All scored episode/group/CSV records, query logs, videos, task plans, referenced source files, BDDL/state files, cached text, and source bundle. No simulations rerun; checkpoint weight tensors and selected state arrays not reloaded.',
                  audit_script_sha256=digest(__file__))
    (ROOT / 'final_artifact_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
