"""Read-only recheck of retained RT-50 records; never launches evaluation.

Run with 'local' on the local host, or '137' through SSH on server137.
This checks logs/traces, video presence/size, source and checkpoint hashes.
It does not rehash or ffprobe the videos; those checks belong to final_audit.json.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import socket
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_run(base, schedule):
    run = base / 'runs' / schedule
    complete = json.loads((run / 'results/COMPLETE.json').read_text())
    manifest = json.loads((run / 'results/manifest.json').read_text())
    audit = json.loads((run / 'final_audit.json').read_text())
    assert sha(run / 'results/COMPLETE.json') == audit['complete_sha256']
    assert sha(manifest['checkpoint']) == manifest['checkpoint_sha256'] == audit['checkpoint_sha256']
    source = json.loads((base / 'source_manifest.json').read_text())
    def check_source(item):
        path, record = item
        assert sha(base / 'repo' / path) == record['sha256'], path
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check_source, source.items()))
    pattern = re.compile(r'Success rate: (?:\x1b\[\d+m)?(\d+)/(\d+).*?current seed: (?:\x1b\[\d+m)?(\d+)')
    error_markers = ['oidn error:', 'cuda out of memory', 'cudaerrormemoryallocation',
                     'cannot create buffer', 'errordevicelost', 'vk::devicelosterror']
    def check_cell(item):
        key, cell = item
        directory = Path(cell['directory'])
        assert json.loads((directory.parent / 'result.json').read_text()) == cell, key
        content = (directory / 'eval.log').read_text(errors='replace')
        assert not any(m in content.lower() for m in error_markers), key
        triples = pattern.findall(content)
        assert len(triples) == len(cell['episodes']) == 100, key
        queries, instructions = {}, {}
        for line in content.splitlines():
            if line.startswith('{"policy_query"'):
                event = json.loads(line)
                queries.setdefault(event['episode'], []).append(event)
            elif line.startswith('{"loopwam_episode"'):
                event = json.loads(line)
                instructions.setdefault(event['loopwam_episode'], []).append(event['instruction'])
        previous = 0
        byte_count = query_count = 0
        for index, (triple, ep) in enumerate(zip(triples, cell['episodes'])):
            successes, count, seed = map(int, triple)
            assert count == index + 1 and successes - previous in (0, 1), key
            assert bool(successes - previous) == ep['success'] and seed == ep['seed'], key
            previous = successes
            q = queries.get(index, [])
            assert q and [x['policy_query'] for x in q] == list(range(len(q))), key
            assert len(q) == ep['queries'] and instructions.get(index) == [ep['instruction']], key
            trace = hashlib.sha256(json.dumps([x['action_sha256'] for x in q]).encode()).hexdigest()
            assert trace == ep['query_trace_sha256'], key
            video = directory / (f'episode{index}_randomized-{str(cell["config"] == "demo_randomized").lower()}'
                                 f'_success-{str(ep["success"]).lower()}.mp4')
            assert str(video) == ep['video'] and video.is_file() and video.stat().st_size >= 1024, key
            byte_count += video.stat().st_size
            query_count += len(q)
        assert previous == cell['successes'], key
        return byte_count, query_count
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(check_cell, complete['finished'].items()))
    assert len(checks) == 100
    return {
        'schedule': schedule, 'base': str(base), 'status': 'passed',
        'checkpoint_sha256': manifest['checkpoint_sha256'], 'source_files_rehashed': len(source),
        'cells_log_trace_verified': 100, 'episodes_log_trace_verified': 10000,
        'videos_present': 10000, 'video_bytes': sum(x[0] for x in checks),
        'policy_queries': sum(x[1] for x in checks),
        'video_content_hashes_and_ffprobe_rerun': False,
        'original_final_audit': audit,
        'launch': json.loads((run / 'launch.json').read_text()),
        'active_process_receipt': json.loads((base / 'active_process.json').read_text()),
        'queue_launch': json.loads((base / 'queue_launch.json').read_text()),
        'smoke': json.loads((base / f'smoke_{schedule}.json').read_text()),
        'original_artifact_sha256': {rel: sha(run / rel) for rel in
            ['results/COMPLETE.json', 'results/manifest.json', 'final_audit.json', 'final_results.json']},
    }


if __name__ == '__main__':
    started = time.time()
    host = sys.argv[1]
    if host == 'local':
        jobs = [(Path('/data14t/anhdh35/loopwam_robotwin_elastic_e0_teacher10_5ep_20261111'), 'flat_k1_n2')]
    elif host == '137':
        root = Path('/media/vmo-perception/disk_2/anhdh35')
        jobs = [(root / 'loopwam_robotwin_elastic_e0_teacher10_5ep_20261111', 'flat_k2_n2'),
                (root / 'loopwam_robotwin_elastic_e0_teacher10_5ep_44_20261112', 'flat_k4_n2')]
    else:
        raise ValueError(host)
    results = [check_run(*job) for job in jobs]
    print(json.dumps({'status': 'passed', 'hostname': socket.gethostname(), 'host': host,
                      'checked_unix_host_clock': time.time(), 'elapsed_seconds': time.time() - started,
                      'runs': results}, indent=2))
