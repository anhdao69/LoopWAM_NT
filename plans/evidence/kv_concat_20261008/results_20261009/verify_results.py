"""Independently verify the downloaded Long results using the Python standard library."""
import hashlib,json,math,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
e=read(R/'job4770/long_analysis/evidence.json');x=read(R/'job4770/expanded_analysis/evidence.json')
def key(r):return (r['base_seed'],r['suite'],r['task'],r['episode'])
def wilson(k,n):
 z=1.959963984540054;p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [center-half,center+half]
def paired(a,b):
 aa={key(r):r['success'] for r in a};bb={key(r):r['success'] for r in b};assert len(aa)==len(a) and len(bb)==len(b) and aa.keys()==bb.keys()
 w=sum(bool(aa[k]) and not bb[k] for k in aa);l=sum(bool(bb[k]) and not aa[k] for k in aa);n=w+l
 p=min(1.,2*sum(math.comb(n,i) for i in range(min(w,l)+1))/2**n) if n else 1.
 return dict(n=len(a),candidate_only=w,baseline_only=l,p_exact=p)
def same(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:same(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for aa,bb in zip(a,b):same(aa,bb)
 elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)
 else:assert a==b,(a,b)
def evaluation(path,n,mode):
 s=read(path/'summary.json');m=read(path/'manifest.json');rows=s['episodes']
 assert s['mode']=='final_rollout' and s['checkpoint_step']==7250 and s['action_kv_mode']==mode
 assert len(rows)==s['total_episodes']==n and sum(bool(r['success']) for r in rows)==s['successes']
 assert len({key(r) for r in rows})==n
 assert {key(r) for r in rows}=={(m['arguments']['seeds'][0],'libero_10',t,i) for t in range(10) for i in range(n//10)}
 assert m['protocol']==dict(max_policy_steps=700,settling_steps=30,replan_steps=10,denoising_steps=10,cfg=1,action_chunk=32)
 for r in rows:
  assert r['checkpoint_sha256']==s['checkpoint_sha256']==m['checkpoint_sha256']
  assert read(path/'episodes'/f"{r['id']}.json")==r
 return rows
baseline=read(R.parent/'accepted_controls/kv_campaign_job4771/control_aligned_v4a1_accepted/summary.json')['episodes']
assert baseline==e['episodes']['aligned']
for mode,job in [('concat',4770),('mix',4771)]:
 run=R/f'job{job}/{mode}_long';rows=[]
 for seed in [42,43,44]:rows+=evaluation(run/f'eval_seed{seed}',100,mode)
 assert rows==e['episodes'][mode]
 m=read(run/'train/manifest.json');timing=read(run/'train/timing.json');state=read(run/'train/trainer_state.json')
 assert m['resume'] is None and m['seed']==42 and m['global_batch']==128 and m['world_size']==2
 assert m['microbatch']==8 and m['gradient_accumulation']==8 and m['policy_parameters']==584536135+(288 if mode=='mix' else 0)
 assert timing['status']=='complete' and timing['completed_updates']==state['update']==7250 and timing['windows_seen']==state['windows_seen']==926780
 assert read(run/'train/fairness.json')['passed']
 records=[json.loads(s) for s in (run/'train/metrics.jsonl').read_text().splitlines()];updates=[r for r in records if 'loss_video' in r]
 assert [r['update'] for r in updates]==list(range(1,7251))
 assert all(math.isfinite(r[k]) for r in updates for k in ['loss_video','loss_action','grad_norm'])
 assert sum(r['windows'] for r in updates)==926780
 for label,part in [('first100',updates[:100]),('last100',updates[-100:])]:same({k:statistics.mean(r[k] for r in part) for k in ['loss_video','loss_action','grad_norm']},e['training'][mode][label])
 for epoch in range(1,11):same({k:statistics.mean(r[k] for r in updates if r['epoch']==epoch) for k in ['loss_video','loss_action','grad_norm']},e['training'][mode]['epochs'][str(epoch)])
 assert len(e['training'][mode]['diagnostics'])==10
 latency=read(run/'latency_eager.json')['models']['loopwam_v0'];assert not latency['compiled'] and latency['checkpoint_sha256']==rows[0]['checkpoint_sha256'] and len(latency['samples_ms'])==100 and len(latency['trials'])==2
 same(statistics.mean(latency['samples_ms']),latency['mean_ms'])
 for seed in [42,43,44]:same(paired([r for r in rows if r['base_seed']==seed],[r for r in baseline if r['base_seed']==seed]),e['tests'][mode][str(seed)])
 same(paired(rows,baseline),e['tests'][mode]['pooled'])
for mode,rows in e['episodes'].items():
 d=e['results'][mode];k=sum(bool(r['success']) for r in rows);assert k==d['successes'];same(k/300,d['pooled_sr']);same(wilson(k,300),d['wilson95'])
 assert {str(t):sum(bool(r['success']) for r in rows if r['task']==t) for t in range(10)}==d['per_task']
expanded={mode:evaluation(R/f'job4770/expanded_{mode}_seed42',500,mode) for mode in ['concat','aligned']}
for mode,rows in expanded.items():
 k=sum(bool(r['success']) for r in rows);assert k==x['models'][mode]['successes'];same(wilson(k,500),x['models'][mode]['wilson95'])
same(paired(expanded['concat'],expanded['aligned']),x['paired'])
assert e['decision']==x['original_decision']=='inconclusive' and x['new_decision'] is None
files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(R.rglob('*')) if p.is_file() and p.name!='verification.json'}
result=dict(passed=True,training_updates_checked=14500,new_final_episodes_checked=1600,baseline_control_episodes_checked=300,checks=['episode identity and raw records','checkpoint binding','protocol','fresh training budgets and finite metrics','loss aggregates','Wilson intervals','paired exact McNemar','latency samples','preregistered decision'],sha256=files)
(R/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='sha256'}))
