"""Check the rendered tables, exports, links and explicit v0 4/4 exclusion."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def csv_rows(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def rows(text):
    lines = [line for line in text.splitlines() if line.startswith('| ')]
    return [[x.strip().replace('**', '') for x in line.strip('|').split('|')] for line in lines[2:]]


def main():
    data = read(OUT / 'results.json')
    evidence = read(OUT / 'evidence/reconciliation.json')
    assert evidence['status'] == 'passed' and evidence['episode_records_reconciled'] == 12000
    assert len(data['runs']) == 6 and len(data['primary_run_ids']) == 5
    assert not any(r['protocol']['variant'] == 'v0-video4-action4' for r in data['runs'])
    for relative, digest in evidence['source_files'].items():
        assert sha(ROOT / relative) == digest, relative
    protected = read(OUT / 'evidence/protected_v0_4x4.json')
    current_paths = {str(p.relative_to(ROOT)) for p in (ROOT / 'robotwin10_v0_4x4_epoch10_20261108').rglob('*') if p.is_file()}
    assert current_paths == set(protected['files'])
    for relative, digest in protected['files'].items():
        assert sha(ROOT / relative) == digest, relative
    for name, original_lines in protected['catalogue_lines'].items():
        now = [line for line in (ROOT / name).read_text().splitlines() if '4/4' in line or '4x4' in line]
        assert now == original_lines, name
    backup = OUT / 'evidence/before_update'
    for relative, digest in read(backup / 'sha256.json').items():
        assert sha(backup / relative) == digest
    text = (OUT / 'README.md').read_text()
    task_csv = csv_rows(OUT / 'task_results.csv')
    summary_csv = csv_rows(OUT / 'summary.csv')
    compare_csv = csv_rows(OUT / 'task_comparison.csv')
    assert len(task_csv) == 60 and len(summary_csv) == 6 and len(compare_csv) == 10
    assert len({(r['run_id'], r['task']) for r in task_csv}) == 60
    for run in data['runs']:
        source_rows = [r for r in task_csv if r['run_id'] == run['id']]
        section = text.split('### ' + run['label'] + '\n', 1)[1].split('\n##', 1)[0]
        rendered = rows(section)
        assert len(source_rows) == 10 and len(rendered) == 11
        for actual, task, csvrow in zip(rendered[:10], run['tasks'], source_rows):
            assert actual == [task['task'], f"{task['clean_successes']}/100", f"{task['clean_sr_percent']:.2f}%",
                              f"{task['randomized_successes']}/100", f"{task['randomized_sr_percent']:.2f}%",
                              f"{task['combined_successes']}/200", f"{task['combined_sr_percent']:.2f}%"]
            expected = dict(run_id=run['id'], model=run['label'], primary_run=run['primary_run'], **task)
            assert csvrow == {k: str(v) for k, v in expected.items()}
        summary = next(r for r in summary_csv if r['run_id'] == run['id'])
        for setting, prefix in [('demo_clean', 'clean'), ('demo_randomized', 'randomized'), ('combined', 'combined')]:
            total = run['totals'][setting]
            assert sum(int(r[prefix + '_successes']) for r in source_rows) == total['successes']
            assert sum(int(r[prefix + '_episodes']) for r in source_rows) == total['episodes']
            for key, value in total.items():
                assert float(summary[prefix + '_' + key]) == float(value)
        assert summary['checkpoint_sha256'] == run['checkpoint_sha256']
        t = run['totals']
        assert rendered[-1] == ['Total', f"{t['demo_clean']['successes']}/1000", f"{t['demo_clean']['sr_percent']:.2f}%",
                                f"{t['demo_randomized']['successes']}/1000", f"{t['demo_randomized']['sr_percent']:.2f}%",
                                f"{t['combined']['successes']}/2000", f"{t['combined']['sr_percent']:.2f}%"]
    section = text.split('## Task comparison — latest completed run per included model\n', 1)[1].split('\n## ', 1)[0]
    comparison = rows(section)
    primary = [r for r in data['runs'] if r['primary_run']]
    assert len(comparison) == 11
    for i, csvrow in enumerate(compare_csv):
        assert comparison[i] == [csvrow['task']] + [f"{r['tasks'][i]['combined_sr_percent']:.2f}%" for r in primary]
        for run in primary:
            assert float(csvrow[run['id'] + '_sr_percent']) == run['tasks'][i]['combined_sr_percent']
    catalogue = (ROOT / 'all_results.md').read_text()
    new_section = catalogue.split('## RoboTwin-10 — selected epoch-10 results, updated November 15\n', 1)[1].split('\n## ', 1)[0]
    for line in section.splitlines():
        if line.startswith('| '):
            assert line in new_section
    report_files = [OUT / 'README.md'] + [ROOT / run['source_directory'] / 'final_report.md' for run in data['runs']]
    link_count = 0
    receipt = OUT / 'evidence/report_validation.json'
    for p in report_files:
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if target.startswith(('http://', 'https://', '#')):
                continue
            path = (p.parent / target.split('#')[0]).resolve()
            assert path.exists() or path == receipt, (str(p), target)
            link_count += 1
    for p in OUT.glob('*.py'):
        compile(p.read_text(), str(p), 'exec')
    result = dict(status='passed', checked_utc=datetime.now(timezone.utc).isoformat(),
                  selected_runs=6, primary_models=5, reconciled_episode_records=12000,
                  detailed_task_rows_checked=60, task_comparison_rows_checked=10,
                  source_files_unchanged=len(evidence['source_files']),
                  protected_v0_4x4_files_unchanged=len(protected['files']),
                  existing_v0_4x4_catalogue_lines_unchanged=True, local_links_checked=link_count,
                  prior_report_backups_verified=8, simulations_run=False,
                  output_sha256={name: sha(OUT / name) for name in
                      ['README.md', 'results.json', 'summary.csv', 'task_results.csv', 'task_comparison.csv']})
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    assert receipt.exists()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
