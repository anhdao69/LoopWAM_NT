"""CPU-only reconstruction of scored results; --final requires every queue/grid complete."""
import argparse,collections,csv,datetime,hashlib,html,itertools,json,pathlib,statistics
ROOT=pathlib.Path(__file__).resolve().parent
LIGHT=pathlib.Path('/storage/anhdh35/Light-WAM')
LOOP=pathlib.Path('/storage/anhdh35/LoopWAM_NT-eval')
FAST=pathlib.Path('/storage/anhdh35/FastWAM-eval')
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
STAGES=['01_full_4_4','02_long_4_4_original','03_long_4_4_repeat','04_long_mix','05_long_4_1','06_long_concat']
LABELS=['LoopWAM full 4/4','LoopWAM Long 4/4 original','LoopWAM Long 4/4 repeat','LoopWAM Long KV mix 4/1','LoopWAM Long aligned 4/1','LoopWAM Long KV concat 4/1']
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default

def validate_grid(rows,suites,sha=None):
    expected=set(itertools.product(suites,[42,43,44],range(10),range(50)))
    seen=set()
    for r in rows:
        key=(r['suite'],r['seed'],r['task'],r['episode'])
        if key in seen:raise ValueError('duplicate episode: '+str(key))
        seen.add(key)
        if type(r['success']) is not bool:raise ValueError('non-boolean success')
        if r['initial_state_index']!=r['episode']:raise ValueError('initial-state mismatch')
        if sha and r.get('checkpoint_sha256')!=sha:raise ValueError('checkpoint identity mismatch')
    if seen!=expected:raise ValueError(f'incomplete/unexpected grid: missing={len(expected-seen)}, extra={len(seen-expected)}')

def aggregate(rows,keys):
    groups=collections.defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    out=[]
    for key,rs in sorted(groups.items()):
        n=len(rs);s=sum(r['success'] for r in rs)
        out.append(dict(zip(keys,key),successes=s,episodes=n,success_rate_percent=100*s/n))
    return out

def load_light():
    rows=[];base=LIGHT/'evaluate_results/libero_long_seeds'
    if 'exit_status=0' not in (base/'run_manifest.txt').read_text():raise ValueError('Light-WAM did not exit successfully')
    for seed in [42,43,44]:
        folder=base/f'seed_{seed}_step_080000';summary=read(folder/'summary.json')
        for f in sorted((folder/'libero_10').glob('*_results.json')):
            x=read(f);ok=x['success_episodes'];bad=x['failure_episodes']
            if len(set(ok+bad))!=50 or len(ok+bad)!=50 or set(ok+bad)!=set(range(50)):raise ValueError('invalid baseline partition')
            if x['successes']!=len(ok) or x['total_episodes']!=50:raise ValueError('baseline count mismatch')
            for ep in range(50):rows.append(dict(model='Light-WAM Long',suite='libero_10',seed=seed,task=x['task_id'],episode=ep,initial_state_index=ep,success=ep in ok,task_description=x['task_description'],record_source='historical binary reconstruction'))
        if sum(r['success'] for r in rows if r['seed']==seed)!=summary['suite_stats']['libero_10']['total_successes']:raise ValueError('baseline summary mismatch')
    validate_grid(rows,['libero_10'])
    manifest=dict(line.split('=',1) for line in (base/'run_manifest.txt').read_text().splitlines() if '=' in line)
    wall=(datetime.datetime.fromisoformat(manifest['finished'])-datetime.datetime.fromisoformat(manifest['started'])).total_seconds()
    return rows,{'label':'Light-WAM Long','status':'complete','path':str(base),'wall_seconds':wall,'manifest':manifest,'parameters':read(ROOT/'lightwam_parameter_count.json')}

def load_native(label,folder,suites,final):
    summary=read(folder/'summary.json',{})
    if summary.get('mode')!='final_rollout':
        if final:raise ValueError(label+': no final summary')
        return [],{'label':label,'status':'pending','path':str(folder),'progress':read(folder/'progress.json',{})}
    manifest=read(folder/'manifest.json')
    sha=summary['checkpoint_sha256']
    if manifest['verified']['checkpoint_sha256']!=sha:raise ValueError('manifest checkpoint mismatch')
    a=manifest['arguments']
    if set(a['suites'])!=set(suites) or a['seeds']!=[42,43,44] or set(a['tasks'])!=set(range(10)) or a['episodes']!=50:raise ValueError('wrong manifest protocol')
    rows=[dict(read(f),model=label,record_source=str(f)) for f in sorted((folder/'episodes').glob('*.json'))]
    validate_grid(rows,suites,sha)
    if summary['episodes']!=len(rows) or summary['successes']!=sum(r['success'] for r in rows):raise ValueError('summary totals disagree')
    if abs(summary['success_rate']-sum(r['success'] for r in rows)/len(rows))>1e-12:raise ValueError('summary SR disagrees')
    for x in aggregate(rows,['suite']):
        persisted=summary['per_suite'][x['suite']]
        if persisted['episodes']!=x['episodes'] or persisted['successes']!=x['successes']:raise ValueError('suite summary disagrees')
    for x in aggregate(rows,['suite','seed']):
        persisted=summary['per_suite'][x['suite']]['seeds'][str(x['seed'])]
        if persisted['episodes']!=x['episodes'] or persisted['successes']!=x['successes']:raise ValueError('seed summary disagrees')
    queries=sum(r['queries'] for r in rows);steps=sum(r['policy_steps'] for r in rows);qt=sum(r['query_seconds'] for r in rows)
    for key,value in [('total_queries',queries),('total_policy_steps',steps),('total_query_seconds',qt)]:
        if key in summary and abs(summary[key]-value)>max(1e-6,abs(value)*1e-10):raise ValueError('telemetry summary disagrees: '+key)
    caps={s:700 if s=='libero_10' else 400 for s in suites}
    if any(not (0<=r['policy_steps']<=caps[r['suite']]) or r['queries']<0 or r['query_seconds']<0 for r in rows):raise ValueError('invalid rollout telemetry')
    cp=pathlib.Path(a.get('checkpoint_dir',''))
    if a.get('backend')=='cuda-graph':
        gate_path=(FAST/'setup_logs/verified_graph_fastwam.json') if label.startswith('Fast') else LOOP/'setup_logs'/('verified_graph_'+cp.name+'.json')
        gate=read(gate_path,{})
        if not gate.get('passed') or gate.get('max_abs_difference')!=0 or gate.get('checkpoint_sha256')!=sha:raise ValueError('graph gate absent/failed/mismatched')
    metadata={f:read(cp/f) for f in ['export.json','training_manifest.json','training_timing.json','fairness.json'] if (cp/f).exists()}
    data=read(cp/'data_manifest.json',{})
    metadata['data_manifest_summary']={k:(len(v) if isinstance(v,list) else v) for k,v in data.items()}
    wall=summary['elapsed_seconds']
    averages=[1000*r['query_seconds']/r['queries'] for r in rows if r['queries']]
    info={'label':label,'status':'complete','path':str(folder),'wall_seconds':wall,'summary':summary,'manifest':{'arguments':a,'verified':{k:v for k,v in manifest['verified'].items() if k!='text_provenance'},'other':{k:v for k,v in manifest.items() if k not in ['arguments','verified','groups']}},'release_metadata':metadata,'performance':{'policy_steps':steps,'queries':queries,'summed_overlapping_query_seconds':qt,'mean_query_ms':1000*qt/queries if queries else None,'median_episode_mean_query_ms':statistics.median(averages) if averages else None,'policy_steps_per_wall_second':steps/wall,'queries_per_wall_second':queries/wall,'mean_episode_policy_steps':steps/len(rows),'mean_rollout_seconds':statistics.mean(r['rollout_seconds'] for r in rows),'timing_boundary':'whole native prediction incl preprocessing/action conversion' if label.startswith('Fast') else 'model inference excludes preprocessing/action conversion'}}
    return rows,info

def csv_write(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def table(rows,keys=None):
    if not rows:return '<p>No completed results available.</p>'
    keys=keys or list(rows[0])
    def val(v):return f'{v:.4f}' if isinstance(v,float) else str(v if v is not None else 'unavailable')
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(val(r.get(k)))+'</td>' for k in keys)+'</tr>' for r in rows)+'</tbody></table></div>'

def detail(title,obj):return '<details><summary>'+html.escape(title)+'</summary><pre>'+html.escape(json.dumps(obj,indent=2))+'</pre></details>'
def completion():
    paths=[LOOP/'evaluate_results/matched_lightwam/campaign_status.json',FAST/'evaluate_results/matched_lightwam/queue_status.json',LOOP/'evaluate_results/matched_v1a4/queue_status.json']
    return [{'path':str(p),'state':read(p,{})} for p in paths]

def build(final=False):
    queues=completion()
    if final and any(x['state'].get('status')!='complete' for x in queues):raise ValueError('evaluation queues have not all completed')
    if not (ROOT/'branch_review.json').exists():raise ValueError('branch review absent')
    rows,light=load_light();models=[light]
    specs=[(label,LOOP/'evaluate_results/matched_lightwam'/stage,SUITES if i==0 else ['libero_10']) for i,(stage,label) in enumerate(zip(STAGES,LABELS))]
    specs += [('Fast-WAM original all suites',FAST/'evaluate_results/matched_lightwam/rollouts',SUITES),('LoopWAM full video1/action4',LOOP/'evaluate_results/matched_v1a4/rollouts',SUITES)]
    for label,folder,suites in specs:
        rs,m=load_native(label,folder,suites,final);rows.extend(rs);models.append(m)
    if final and len(rows)!=27000:raise ValueError('wrong campaign total')
    suite=aggregate(rows,['model','suite']);seeds=aggregate(rows,['model','suite','seed']);tasks=aggregate(rows,['model','suite','task']);taskseeds=aggregate(rows,['model','suite','task','seed'])
    baseline_rate=100*sum(r['success'] for r in rows if r['model']=='Light-WAM Long')/1500
    for r in tasks+taskseeds:
        r['task_description']=next(x.get('task_description','') for x in rows if x['model']==r['model'] and x['suite']==r['suite'] and x['task']==r['task'])
    for r in suite:
        values=[s['success_rate_percent'] for s in seeds if s['model']==r['model'] and s['suite']==r['suite']]
        r['seed_sample_std_percent']=statistics.stdev(values)
        if r['suite']=='libero_10':r['difference_pp_vs_lightwam']=r['success_rate_percent']-baseline_rate
    baseline={(r['seed'],r['task'],r['episode']):r['success'] for r in rows if r['model']=='Light-WAM Long'}
    paired=[]
    for m in models[1:]:
        rs=[r for r in rows if r['model']==m['label'] and r['suite']=='libero_10']
        if not rs:continue
        counts=collections.Counter((baseline[(r['seed'],r['task'],r['episode'])],r['success']) for r in rs)
        paired.append({'model':m['label'],'both_success':counts[True,True],'both_failure':counts[False,False],'light_only_success':counts[True,False],'model_only_success':counts[False,True],'paired_episodes':len(rs)})
    evidence=[];artifacts={}
    for base in [LOOP/'setup_logs',FAST/'setup_logs',LOOP/'evaluate_results/matched_lightwam',LOOP/'evaluate_results/matched_v1a4',FAST/'evaluate_results/matched_lightwam',ROOT]:
        for p in sorted(list(base.glob('*.json'))+list(base.glob('*tests.log'))+list(base.glob('regression*.log'))+list(base.glob('environment.lock.txt'))+list(base.glob('*.py'))):
            if p.name in ['results.json','evidence_index.json','report_status.json']:continue
            b=p.read_bytes();evidence.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
            if ('verified_graph' in p.name or p.name in ['rollout_graph_parity.json','cpu_checkpoint_load.json','campaign_plan.json','runtime_estimates.json','v1a4_setup_status.json']):artifacts[str(p)]=read(p)
    for m in models:
        base=pathlib.Path(m['path'])
        for p in [base/'summary.json',base/'manifest.json']:
            if p.is_file():b=p.read_bytes();evidence.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)})
    report={'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'final':final,'scored_episodes':len(rows),'expected_episodes':27000,'models':models,'suite_results':suite,'seed_results':seeds,'task_results':tasks,'paired_long':paired,'queues':queues,'audit':read(ROOT/'audit_findings.json'),'branch_inventory':read(ROOT/'branch_inventory.json'),'branch_review':read(ROOT/'branch_review.json'),'hardware':read(ROOT/'hardware.json'),'environment_and_releases':read(ROOT/'release_environment_provenance.json'),'source_environment_hashes':read(ROOT/'source_environment_evidence.json'),'speed_and_setup_evidence':artifacts}
    out=ROOT/('final' if final else 'draft');out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(report,indent=2));(out/'evidence_index.json').write_text(json.dumps(evidence,indent=2))
    for name,rs in [('suite_results',suite),('seed_results',seeds),('task_results',tasks),('task_seed_results',taskseeds),('paired_long',paired),('episodes',rows)]:csv_write(out/(name+'.csv'),rs)
    parts=['<!doctype html><html><head><meta charset="utf-8"><title>WAM evaluation report</title><style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 24px;line-height:1.55;color:#172330}h1,h2{color:#143e65}table{border-collapse:collapse;font-size:14px}td,th{padding:8px;border:1px solid #ccd6e0;text-align:left}th{background:#eaf1f7}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;background:#f4f7fa;padding:16px}.scroll{overflow:auto}details{margin:16px 0}summary{cursor:pointer;font-weight:600}li{margin:7px 0}</style></head><body>', '<h1>WAM evaluation: detailed setup, branch audit and results</h1>',f'<p><strong>{"FINAL — all 27,000 scored episodes verified" if final else "DRAFT — incomplete campaign; pending models have no final score"}</strong>. Generated {report["generated_at"]}. Reconstructed scored episodes: {len(rows):,}/27,000.</p>', '<p>Seeds 42, 43, 44; 50 official initial states per task. The report distinguishes common rollout rules from native policy settings and timing boundaries. Every reported score below was independently reconstructed from complete episode records.</p>','<h2>Results by suite</h2>',table(suite),'<h2>Results by seed</h2>',table(seeds),'<h2>Production performance</h2>']
    perf=[]
    for m in models:
        p=m.get('performance',{});perf.append(dict(model=m['label'],status=m['status'],production_wall_seconds=m.get('wall_seconds'),**p))
    parts.append(table(perf,list(dict.fromkeys(k for r in perf for k in r))))
    parts+=['<h2>Paired Long outcomes against Light-WAM</h2>',table(paired)]
    for title,paras in report['audit']['sections'].items():parts.append('<h2>'+html.escape(title)+'</h2>'+''.join('<p>'+html.escape(p)+'</p>' for p in paras))
    parts += ['<h2>Hardware and simulator</h2>',detail('Hardware snapshot',report['hardware']),detail('Software versions and pinned releases',report['environment_and_releases']),detail('Evaluated source hashes and environment locks',report['source_environment_hashes']),'<h2>All branches: revisions, changes and implications</h2>',detail('Reviewed branch interpretations',report['branch_review']),detail('Branch inventory and source hashes',report['branch_inventory']),'<h2>Checkpoint and evaluation manifests</h2>']
    for m in models:parts.append(detail(m['label'],m))
    parts += ['<h2>Graph parity, benchmark choices and setup evidence</h2>',detail('Recorded gates and plans',artifacts),'<h2>Per-task scores</h2>',table(tasks),'<h2>Full task-by-seed appendix</h2>',table(taskseeds),'<h2>Data downloads and evidence</h2>','<ul>'+''.join(f'<li><a href="{name}">{name}</a></li>' for name in ['results.json','suite_results.csv','seed_results.csv','task_results.csv','task_seed_results.csv','episodes.csv','paired_long.csv','evidence_index.json'])+'</ul>',detail('Source evidence hashes',evidence),'</body></html>']
    (out/'report.html').write_text(''.join(parts))
    return {'output':str(out/'report.html'),'scored_episodes':len(rows),'final':final}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args();print(json.dumps(build(args.final)))
