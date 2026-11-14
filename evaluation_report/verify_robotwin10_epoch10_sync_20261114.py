"""Verify the merged epoch-10 snapshot without changing reports or rerunning inference."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'robotwin10_epoch10_results_20261115'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def csvrows(p):
    with p.open() as f:
        return list(csv.DictReader(f))

def main():
    manifest = read(OUT/'evidence/sync_source_manifest.json')
    for rel, info in manifest.items():
        rel = Path(rel).relative_to('evaluation_report')
        p = OUT/'evidence/server_catalogues'/rel if str(rel) in ('README.md','all_results.md') else ROOT/rel
        assert sha(p)==info['sha256'] and p.stat().st_size==info['bytes'], rel
    data=read(OUT/'results.json')
    assert len(data['runs'])==6 and len(data['primary_run_ids'])==5
    tasks=csvrows(OUT/'task_results.csv'); summaries=csvrows(OUT/'summary.csv')
    comparisons=csvrows(OUT/'task_comparison.csv')
    assert len(tasks)==60 and len(summaries)==6 and len(comparisons)==10
    count=0
    for run in data['runs']:
        d=ROOT/run['source_directory']
        complete=read(d/'COMPLETE.json'); audit=read(d/'final_audit.json'); final=read(d/'final_results.json')
        assert audit['status']=='passed' and audit['complete_sha256']==sha(d/'COMPLETE.json')
        assert audit['checkpoint_sha256']==complete['checkpoint_sha256']==run['checkpoint_sha256']
        assert len(complete['finished'])==20 and not complete['running'] and not complete['failed'] and complete['pending']==0
        protocol=read(d/'planned_protocol.json')
        assert protocol==run['protocol'] and protocol['epoch']==10 and protocol['seed']==42
        assert protocol['expected_step']==90890 and protocol['num_inference_steps']==10
        assert protocol['variant']!='v0-video4-action4'
        records=csvrows(d/'episodes.csv'); assert len(records)==2000
        indexed={(r['task'],r['setting'],int(r['episode_index'])):r for r in records}
        assert len(indexed)==2000
        totals=collections.defaultdict(lambda:[0,0]); cells={}
        for cell in complete['finished'].values():
            assert cell['complete'] and cell['exit_code']==0
            task,setting=cell['task'],cell['config']; episodes=cell['episodes']
            assert len(episodes)==len({e['seed'] for e in episodes})==100
            success=0
            for i,ep in enumerate(episodes):
                assert type(ep['success']) is bool
                expected=dict(task=task,setting=setting,episode_index=i,**ep)
                assert indexed[(task,setting,i)]=={k:str(v) for k,v in expected.items()}
                success+=ep['success']
            assert success==cell['successes'];cells[(task,setting)]=success
            totals[setting][0]+=success;totals[setting][1]+=100
        totals['combined']=[sum(v[0] for v in totals.values()),sum(v[1] for v in totals.values())]
        summary=next(r for r in summaries if r['run_id']==run['id'])
        for setting,(success,n) in totals.items():
            expected=dict(successes=success,episodes=n,sr_percent=100*success/n)
            assert expected==run['totals'][setting]==final['totals'][setting]
            prefix={'demo_clean':'clean','demo_randomized':'randomized','combined':'combined'}[setting]
            for k,v in expected.items():assert float(summary[prefix+'_'+k])==float(v)
        for task in run['tasks']:
            a,b=(cells[(task['task'],s)] for s in ('demo_clean','demo_randomized'))
            assert a==task['clean_successes'] and b==task['randomized_successes']
            assert a+b==task['combined_successes'] and (a+b)/2==task['combined_sr_percent']
            row=next(r for r in tasks if r['run_id']==run['id'] and r['task']==task['task'])
            assert row=={k:str(v) for k,v in dict(run_id=run['id'],model=run['label'],primary_run=run['primary_run'],**task).items()}
            rendered=f"| {task['task']} | {a}/100 | {a:.2f}% | {b}/100 | {b:.2f}% | {a+b}/200 | {(a+b)/2:.2f}% |"
            section=(OUT/'README.md').read_text().split('### '+run['label']+'\n',1)[1].split('\n##',1)[0]
            assert rendered in section
        excluded=read(d/'failed_attempts.json')
        assert len(excluded)==audit['failed_attempts'] and sum(r['episodes'] for r in excluded)==audit['excluded_partial_episodes']
        count+=len(records)
    assert count==12000
    for row in comparisons:
        for run in data['runs']:
            if run['primary_run']:
                task=next(t for t in run['tasks'] if t['task']==row['task'])
                assert float(row[run['id']+'_sr_percent'])==task['combined_sr_percent']
    sync=read(ROOT/'sync_manifest_20261114d.json')
    for item in sync['preserved_versions']: assert sha(ROOT/item['preserved'])==item['sha256']
    old=next(x for x in sync['preserved_versions'] if x['file']=='all_results.md')
    section=(OUT/'catalogue_section.md').read_text()
    assert (ROOT/'all_results.md').read_text().replace(section+'\n','',1)==(ROOT/old['preserved']).read_text()
    for p in [OUT/'README.md',ROOT/'README.md',ROOT/'all_results.md']:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('http:','https:','#','/')):continue
            assert (p.parent/target.split('#')[0]).exists(), (p.name,target)
    for p in [ROOT/'all_results/build_snapshot.py', OUT/'reconcile_and_render.py',OUT/'verify_report.py']:
        compile(p.read_text(),str(p),'exec')
    print(f'PASS: {len(manifest)} source-file hashes, 6 runs, {count} episode records, 120 cells, 60 task rows, CSV/Markdown agreement, preserved catalogue and links.')
    print('Scope: saved-record reconciliation; no new simulation, media-byte or GPU numerical audit.')

if __name__=='__main__':main()
