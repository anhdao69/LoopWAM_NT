"""Reconcile existing metric evidence into a dated report; never rerun evaluation."""
from pathlib import Path
import json,csv,hashlib,datetime,math,shutil
R=Path(__file__).resolve().parents[1];OUT=R/'all_results';sources={};checks=[]
STAMP=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7))).isoformat(timespec='seconds')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);sources[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size);return json.loads(p.read_text())
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
def csvout(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def rate(k,n):return 100*k/n if n else None
def result_totals(rows):
 totals={}
 for setting in ['demo_clean','demo_randomized','combined']:
  a=[x for x in rows if setting=='combined' or x['setting']==setting];k=sum(x['successes'] for x in a);n=sum(x['episodes'] for x in a)
  totals[setting]=dict(successes=k,episodes=n,sr_percent=rate(k,n))
 return totals
LABELS={
'robotwin10_dense_s30':'Dense-S30 — local',
'robotwin10_dense_s12_20261105':'Dense-S12 — local, valid four-worker rerun',
'robotwin10_v0_4x4':'v0 4/4 — original local',
'robotwin10_v0_4x4_remote189':'v0 4/4 — rerun server189',
'robotwin10_v0_4x4_remote137_20261105':'v0 4/4 — rerun server137',
'robotwin10_v0_4x2':'v0 4/2 — original local',
'robotwin10_v0_2x2':'v0 2/2 — local',
'robotwin10_v0_4x1':'v0 4/1 — server189',
'robotwin10_v0_4x1_local_20261105':'v0 4/1 — local rerun',
'robotwin10_v0_1x4_remote137_20261105':'v0 1/4 — server137, valid four-worker rerun',
'robotwin10_h100_v0_4x4_20261106':'v0 4/4 — H100 retrain, local evaluation',
'robotwin10_h100_v0_4x2_remote189_20261106':'v0 4/2 — H100 retrain, server189 evaluation',
'robotwin_v0':'v0 4/4 — epoch5, full50 training',
'robotwin_v0_epoch10':'v0 4/4 — epoch10, full50 training',
'robotwin10_seed43_two_tasks/dense_s30':'Dense-S30 — seed43, two tasks only',
'robotwin10_seed43_two_tasks/v0_4x4':'v0 4/4 — seed43, two tasks only'}
robot=[];task_rows=[]
for relative,label in LABELS.items():
 d=R/relative;f=read(d/'final_results.json');a=read(d/'final_audit.json');c=read(d/'COMPLETE.json')
 assert a['status']=='passed',relative
 assert a['complete_sha256']==sha(d/'COMPLETE.json'),relative
 assert not c.get('running') and not c.get('failed'),relative
 rows=[]
 for key,cell in c['finished'].items():
  episodes=cell['episodes'];assert len(episodes)==100 and cell['complete'] and cell['exit_code']==0
  assert sum(x['success'] for x in episodes)==cell['successes']
  assert len({e['seed'] for e in episodes})==100
  rows.append(dict(task=cell['task'],setting=cell['config'],episodes=100,successes=cell['successes']))
 stored={(x['task'],x['setting']):(x['episodes'],x['successes']) for x in f['tasks']}
 assert stored=={(x['task'],x['setting']):(x['episodes'],x['successes']) for x in rows},relative
 totals=result_totals(rows)
 for key,t in totals.items():
  old=f['totals'][key];assert (old['successes'],old['episodes'])==(t['successes'],t['episodes'])
  assert math.isclose(old['sr_percent'],t['sr_percent'],abs_tol=1e-8)
 benchmark='RoboTwin50' if relative.startswith('robotwin_v0') else 'RoboTwin10_seed43_subset' if 'seed43' in relative else 'RoboTwin10'
 seed=43 if 'seed43' in relative else 42
 report=d/('final_report.md' if (d/'final_report.md').exists() else 'report.md')
 record=dict(run=relative,label=label,benchmark=benchmark,seed=seed,nfe=10,status='complete',audit='saved final audit passed; COMPLETE digest and score arithmetic rechecked',renderer_guard=a.get('renderer_logs_clean','not explicitly recorded in historical audit'),checkpoint_sha256=c['checkpoint_sha256'],report=str(report.relative_to(R)),totals=totals)
 robot.append(record)
 for x in rows:task_rows.append(dict(run=relative,benchmark=benchmark,seed=seed,**x,sr_percent=rate(x['successes'],x['episodes'])))
 checks.append(dict(run=relative,checks='COMPLETE SHA256 matches saved audit; complete records/task coverage; per-episode success sums; unique scene seeds; final JSON arithmetic',passed=True))
# Epoch6 lacks a saved full audit. Reconcile records without upgrading its audit claim.
b=Path('/data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031');c=read(b/'results/COMPLETE.json');p=read(b/'results/manifest.json');rows=[]
assert not c['running'] and not c['failed'] and c['pending']==0 and len(c['finished'])==100
assert set(c['finished'])=={f'{t}/{s}' for t in p['tasks'] for s in p['settings']}
for key,v in c['finished'].items():
 assert v['complete'] and v['exit_code']==0 and len(v['episodes'])==100 and sum(e['success'] for e in v['episodes'])==v['successes']
 assert len({e['seed'] for e in v['episodes']})==100
 rows.append(dict(task=v['task'],setting=v['config'],episodes=100,successes=v['successes']))
totals=result_totals(rows)
for setting in p['settings']:assert math.isclose(totals[setting]['sr_percent']/100,c['success_rates'][setting])
d=R/'robotwin_v0_epoch06';dump(d/'reconciled_results.json',dict(totals=totals,tasks=rows,audit_status='completion records reconciled; no final_audit.json found',source_complete=str(b/'results/COMPLETE.json'),source_complete_sha256=sha(b/'results/COMPLETE.json'),reconciled_at=STAMP))
robot.append(dict(run='robotwin_v0_epoch06',label='v0 4/4 — epoch6, full50 training',benchmark='RoboTwin50',seed=42,nfe=10,status='complete',audit='completion-record reconciliation only; no saved final artifact audit found',renderer_guard='not independently re-audited',checkpoint_sha256=c['checkpoint_sha256'],report='robotwin_v0_epoch06/report.md',totals=totals))
for x in rows:task_rows.append(dict(run='robotwin_v0_epoch06',benchmark='RoboTwin50',seed=42,**x,sr_percent=rate(x['successes'],x['episodes'])))
checks.append(dict(run='robotwin_v0_epoch06',checks='100 complete cells;100 unique episode seeds each; per-episode success arithmetic; no failed/pending cells; no video hash re-audit',passed=True))
flat=[]
for x in robot:
 for setting,t in x['totals'].items():flat.append({k:v for k,v in x.items() if k!='totals'}|dict(setting=setting,**t))
csvout('robotwin_results.csv',flat);csvout('robotwin_tasks.csv',task_rows)
# Historical matched LIBERO ledger: preserve origin and validate every suite sum.
matched=read(R/'matched_baselines/data/results.json')['models'];libero=[]
for m in matched:
 assert sum(x['episodes'] for x in m['suites'].values())==m['episodes']
 assert sum(x['successes'] for x in m['suites'].values())==m['successes']
 libero.append(dict(model=m['model'],episodes=m['episodes'],successes=m['successes'],sr_percent=rate(m['successes'],m['episodes']),source=m['source'],checkpoint_sha256=json.dumps(m.get('checkpoint_sha256')),**{k:rate(v['successes'],v['episodes']) for k,v in m['suites'].items()}))
csvout('libero_matched.csv',libero)
# CL/RT detailed JSON is the snapshot authority; cross-check existing CSV.
campaign=read(R/'campaign/detailed_results.json')['conditions'];campaign_rows=[];unpublished=[]
current=read(R/'campaign/current_results.json')['runs']
with (R/'campaign/run_summary.csv').open() as f:old_csv=list(csv.DictReader(f))
sources[str(R/'campaign/run_summary.csv')]=dict(sha256=sha(R/'campaign/run_summary.csv'))
for v in campaign:
 if v['status'] not in ['complete','external_complete']:
  unpublished.append(dict(model=v['run'],schedule=v.get('schedule'),status_at_snapshot=v['status']));continue
 evidence=next(x for x in current if x['checkpoint_sha256']==v['checkpoint_sha256'] and x.get('schedule')==v.get('schedule'))
 s=v.get('summary',dict(episodes=evidence['completed'],successes=evidence['successes']))
 assert s['episodes']==6000 and s['successes']==evidence['successes']
 if s.get('per_suite'):
  assert sum(t['episodes'] for t in s['per_suite'].values())==s['episodes'];assert sum(t['successes'] for t in s['per_suite'].values())==s['successes']
 match=[z for z in old_csv if z['checkpoint_sha256']==v['checkpoint_sha256'] and (z['schedule'] or None)==v.get('schedule')]
 assert len(match)==1 and int(match[0]['successes'])==s['successes'] and int(match[0]['episodes'])==s['episodes']
 campaign_rows.append(dict(model=match[0]['model'],family=v['family'],status=v['status'],episodes=s['episodes'],successes=s['successes'],sr_percent=rate(s['successes'],s['episodes']),checkpoint_sha256=v['checkpoint_sha256']))
csvout('cl_rt_results.csv',campaign_rows)
rob=read(R/'robustness/results.json');robrows=[]
for j in rob['jobs']:
 assert j['status']=='complete' and j['episodes']==j['planned'];assert j['successes']==j['summary']['successes']
 suite=[x for x in rob['suites'] if (x['model'],x['benchmark'])==(j['model'],j['benchmark'])]
 assert sum(x['episodes'] for x in suite)==j['episodes'] and sum(x['successes'] for x in suite)==j['successes']
 cats=[x for x in rob['categories'] if (x['model'],x['benchmark'])==(j['model'],j['benchmark'])]
 assert sum(x['episodes'] for x in cats)==j['episodes'] and sum(x['successes'] for x in cats)==j['successes']
 robrows.append(dict(model=j['model'],benchmark=j['benchmark'],seed=42,episodes=j['episodes'],successes=j['successes'],sr_percent=rate(j['successes'],j['episodes'])))
csvout('robustness_results.csv',robrows)
ka=read(R/'v0_ka_sweep/queue_status.json');karows=[]
for j in ka['jobs']:
 assert j['status']=='complete';v=j['summary'];assert v['episodes']==6000 and v['successes']==j['progress']['successes']
 karows.append(dict(action_loops=j['ka'],episodes=v['episodes'],successes=v['successes'],sr_percent=rate(v['successes'],v['episodes']),checkpoint_sha256=v['checkpoint_sha256']))
csvout('v0_ka_results.csv',karows)
# Build current master report.
lines=['# All recorded evaluation results', '',f'Updated **{STAMP}** (Vietnam time).', '',
'This report reconciles the saved evaluation records in this checkout. It includes the latest H100 RoboTwin-10 retrains, all retained RoboTwin reruns, full RoboTwin epoch comparisons, matched LIBERO, CL/RT, Plus/Pro and action-depth results. No evaluations were rerun during this update. Original logs, immutable scored artifacts and historical reports are retained.', '',
'RoboTwin score arithmetic was recomputed from accepted episode records and checked against final tables. Where a final audit exists, its COMPLETE.json digest was checked. Existing video audits are cited; this update did not rehash every historical video or re-run historical simulations. Epoch6 has complete records but lacks a saved final_audit.json. Historical LIBERO/Plus/Pro counts were cross-checked at their saved aggregation levels, not freshly re-audited at video level.', '',
'No pooled total is calculated across sections: historical campaign and branch catalogues overlap. Different benchmarks, training task sets, seeds, reruns and hosts remain separate. Unseeded RoboTwin instruction-template sampling prevents asserting exact trajectory pairing.', '',
'## RoboTwin-10 — full ten-task runs, seed42', '',
'Each row is 10 tasks × clean/randomized ×100 accepted episodes =2,000 episodes, NFE10. All12 runs have saved passing final audits. Reruns are separate observations; no best-seed or best-task selection. H100 means training hardware, not evaluation hardware.', '',
'| Model / execution | Clean | Randomized | Combined | Successes | Evidence |','| --- | ---: | ---: | ---: | ---: | --- |']
r10=sorted([x for x in robot if x['benchmark']=='RoboTwin10'],key=lambda x:-x['totals']['combined']['sr_percent'])
for x in r10:
 t=x['totals'];lines.append(f"| {x['label']} | {t['demo_clean']['sr_percent']:.2f}% | {t['demo_randomized']['sr_percent']:.2f}% | **{t['combined']['sr_percent']:.2f}%** | {t['combined']['successes']}/2000 | [report]({x['report']}) |")
lines+=['','### H100 retrains versus the original local variants','','| Variant | Original | H100 retrain | Difference | Success-count change |','| --- | ---: | ---: | ---: | ---: |']
lookup={x['run']:x for x in robot}
for action,new in [(4,'robotwin10_h100_v0_4x4_20261106'),(2,'robotwin10_h100_v0_4x2_remote189_20261106')]:
 a=lookup[f'robotwin10_v0_4x{action}']['totals']['combined'];b=lookup[new]['totals']['combined'];lines.append(f"| v0 4/{action} | {a['sr_percent']:.2f}% | {b['sr_percent']:.2f}% | {b['sr_percent']-a['sr_percent']:+.2f} pp | {b['successes']-a['successes']:+d} |")
lines+=['','The retrains declare matching model/trainer source hashes and normalization values. Their donor-artifact and data-manifest hashes differ from the earlier releases; see their setup reports. Evaluation hosts differ. The differences above are observed single-seed outcomes, not a significance claim.', '',
'## Full RoboTwin — all50 tasks, seed42','','Each epoch row uses 10,000 episodes:100 per task per setting;NFE10. These are the full50-task-trained checkpoint family, separate from RoboTwin-10 training.','','| Epoch | Clean | Randomized | Combined | Successes | Evidence status |','| --- | ---: | ---: | ---: | ---: | --- |']
for epoch,key in [(5,'robotwin_v0'),(6,'robotwin_v0_epoch06'),(10,'robotwin_v0_epoch10')]:
 x=lookup[key];t=x['totals'];label='Saved final audit passed' if epoch!=6 else 'Complete records reconciled; no saved final artifact audit'
 lines.append(f"| {epoch} | {t['demo_clean']['sr_percent']:.2f}% | {t['demo_randomized']['sr_percent']:.2f}% | **{t['combined']['sr_percent']:.2f}%** | {t['combined']['successes']}/10000 | [{label}]({x['report']}) |")
lines+=['','Epoch10 versus epoch5: +0.46pp clean, −0.18pp randomized, **+0.14pp combined (+14 successes)**. Epoch10 and the recent ten-task reruns include an explicit renderer-log guard; older historical audits did not all record that same check.', '',
'## Seed43 two-task experiments — separate from the main table','','Only move_stapler_pad and open_microwave;100 episodes each, clean/randomized;400 episodes/model. Never substituted into seed42 results.','','| Model | Clean | Randomized | Combined | Successes |','| --- | ---: | ---: | ---: | ---: |']
for x in robot:
 if x['benchmark']=='RoboTwin10_seed43_subset':
  t=x['totals'];lines.append(f"| [{x['label']}]({x['report']}) | {t['demo_clean']['sr_percent']:.2f}% | {t['demo_randomized']['sr_percent']:.2f}% | {t['combined']['sr_percent']:.2f}% | {t['combined']['successes']}/400 |")
lines+=['','## Matched standard LIBERO and Long-only results','','Source: [matched-baselines records](matched_baselines/README.md). Four-suite entries use6,000 episodes, seeds42/43/44,50 episodes/task/seed. Long-only entries use1,500 episodes and are explicitly marked by their denominator. The first14 entries are the later matched campaign; the remaining entries retain the earlier campaign provenance.','','| Model | Spatial | Object | Goal | Long | Overall / evaluated scope | Successes |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for x in libero:
 vals=[f"{x[k]:.2f}%" if k in x else '—' for k in ['libero_spatial','libero_object','libero_goal','libero_10']]
 lines.append('| '+x['model']+' | '+' | '.join(vals)+f" | {x['sr_percent']:.2f}% | {x['successes']}/{x['episodes']} |")
lines+=['','## ChronoLoop and Residual Transport — historical campaign','','Seeds42/43/44;6,000 episodes per completed condition. `external_complete` retains its source provenance and does not mean a new local execution. Counts agree between detailed JSON and the existing run-summary CSV. [Full campaign evidence](campaign/detailed_report.html).','','| Condition | Source status | Successes | SR |','| --- | --- | ---: | ---: |']
for x in campaign_rows:lines.append(f"| {x['model']} | {x['status']} | {x['successes']}/{x['episodes']} | {x['sr_percent']:.2f}% |")
lines+=['','## LIBERO-Plus and LIBERO-Pro','','Seven CL checkpoints, seed42 only. Plus follows the released FasterWAM15% selection, one episode per selected task,1,506 episodes/model. Pro uses Position and Task separately, four suites,10 episodes/task:400 episodes/perturbation,800/model. All14 jobs/16,142 episodes completed in the saved campaign. Low Pro Position SR is retained as measured; completion alone does not establish its cause. [Detailed protocol and results](robustness/report.md).','','| Model | Plus /1506 | Pro Position /400 | Pro Task /400 | Pro combined /800 |','| --- | ---: | ---: | ---: | ---: |']
for model in dict.fromkeys(x['model'] for x in robrows):
 plus=next(x for x in robrows if x['model']==model and x['benchmark']=='plus');pro=next(x for x in robrows if x['model']==model and x['benchmark']=='pro');cats={x['group']:x for x in rob['categories'] if x['model']==model and x['benchmark']=='pro'}
 fmt=lambda x:f"{x['successes']}/{x['episodes']} ({rate(x['successes'],x['episodes']):.2f}%)"
 lines.append(f"| {model} | {fmt(plus)} | {fmt(cats['Position'])} | {fmt(cats['Task'])} | {fmt(pro)} |")
lines+=['','## Original local v0 action-depth queue','','Kv4, NFE10, seeds42/43/44,6,000 episodes per condition. Ka1/Ka2 completed;Ka3/5/6 were removed from this queue at user request. Separate later matched-baseline Ka3/5/6 results above remain valid records of that campaign. These two histories must not be conflated. [Original queue](v0_ka_sweep/report.md). Per-suite columns added 2026-11-14 from the saved per-episode records of the same two runs (nothing re-evaluated): [breakdown by suite, seed and task](v0_ka_sweep/suite_breakdown_20261114/README.md).','','| Ka | Spatial | Object | Goal | Long | Successes | SR |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
# Per-suite columns (2026-11-14): read from the breakdown of the same two runs; totals must agree with the queue record.
kb=read(R/'v0_ka_sweep/suite_breakdown_20261114/breakdown.json')['runs'];KS=['libero_spatial','libero_object','libero_goal','libero_10']
for x in karows:
 b=kb[str(x['action_loops'])];assert b['passed'] and (b['episodes'],b['successes'],b['checkpoint_sha256'])==(x['episodes'],x['successes'],x['checkpoint_sha256'])
 assert sum(b['suites'][s]['successes'] for s in KS)==x['successes'] and all(b['suites'][s]['episodes']==1500 for s in KS)
 lines.append(f"| {x['action_loops']} | "+' | '.join(f"{b['suites'][s]['rate']:.2f}%" for s in KS)+f" | {x['successes']}/{x['episodes']} | {x['sr_percent']:.2f}% |")
lines+=['','## Exclusions and missing-result history','','- Dense-S12 and v0 1/4 ten-worker RoboTwin attempts were invalidated for GPU/renderer errors. Only their fresh four-worker audited reruns appear above; archived attempts are not additional model results.','- The waiting duplicate RoboTwin4/4 queue on server137 was cancelled; the separate later completed rerun is listed by its dated run identity.','- The original `dense_full` waiting queue was cancelled without scored episodes. Separate matched-baseline Dense-S12/S30 results are listed above; RoboTwin dense models are another benchmark.','- The historical CL/RT discovery snapshot has29 `not_published` conditions. This is its saved status, not a fresh assertion about Hugging Face availability today. They have no numeric result in that snapshot and are retained in [the unscored inventory](all_results/unscored_inventory.json).','- No seed43 best-task substitutions, smoke/calibration results, partial failed attempts or archived invalid runs are included in final SR.', '',
'## Branch-reported matched result catalogue','','The branch snapshot also records Elastic/RT/Chrono/init-variant results beyond the local campaign. The following catalogue preserves the original source labels from the October30 matched report. It overlaps the tables above; it is not another set of independently rerun models and must not be summed with them. [Branch evidence index](branch_reports/index.json).','']
src=R/'matched_baselines/README.md';text=src.read_text();sources[str(src)]=dict(sha256=sha(src));section=text.split('## 1. ',1)[1].split('\n## 2.',1)[0];table='\n'.join(x for x in section.splitlines() if x.startswith('|'))
lines+=[table,'','## Downloads and provenance','','- [All RoboTwin summary rows](all_results/robotwin_results.csv) and [every RoboTwin task/setting result](all_results/robotwin_tasks.csv), including seed and run identity.','- [Matched LIBERO](all_results/libero_matched.csv), [CL/RT](all_results/cl_rt_results.csv), [Plus/Pro](all_results/robustness_results.csv), [Ka sweep](all_results/v0_ka_results.csv).','- [Machine-readable consolidated snapshot](all_results/results.json) and [source hashes / reconciliation record](all_results/reconciliation.json).','', 'This is a local report update. It does not push commits or synchronize every repository branch. Historical branch documents retain their original dates and provenance.','']
(R/'all_results.md').write_text('\n'.join(lines))
dump(OUT/'results.json',dict(updated_at=STAMP,robotwin=robot,libero_matched=libero,cl_rt=campaign_rows,robustness=robrows,ka_sweep=karows,note='Sections overlap; do not sum episode counts across ledgers. No new evaluation performed.'))
dump(OUT/'unscored_inventory.json',dict(historical_unpublished_conditions=unpublished,original_ka_removed=[3,5,6],original_dense_queue_removed=['dense-s12','dense-s30'],invalidated_robotwin_attempts=['robotwin10_dense_s12_20261105/invalid_10worker_run','robotwin10_v0_1x4_remote137_20261105/invalid_10worker_run']))
dump(OUT/'reconciliation.json',dict(updated_at=STAMP,status='passed',robotwin_checks=checks,counts=dict(robotwin10_runs=len(r10),robotwin50_runs=3,seed43_subset_runs=2,libero_matched_rows=len(libero),cl_rt_completed_rows=len(campaign_rows),historical_unpublished=len(unpublished),robustness_jobs=len(robrows),ka_conditions=len(karows)),source_files=sources,scope='Saved COMPLETE digests, episode outcome arithmetic and aggregate consistency; not a fresh video audit.'))
print(json.dumps(dict(status='passed',robotwin10=len(r10),robotwin_total=len(robot),libero_matched=len(libero),cl_rt=len(campaign_rows),robustness=len(robrows),ka=len(karows),master=str(R/'all_results.md')),indent=2))
