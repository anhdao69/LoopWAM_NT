#!/usr/bin/env python3
"""Supplemental 500-state and final six-model full-suite campaign reports."""
import argparse,json
from pathlib import Path
import numpy as np
from report_kv_campaign import identity,paired,wilson,read
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')


def publish(out,evidence,lines):
 partial=out.with_name(out.name+'.partial')
 if out.exists():raise ValueError('Refusing to overwrite final report')
 partial.mkdir(parents=True,exist_ok=False)
 (partial/'evidence.json').write_text(json.dumps(evidence,indent=2))
 (partial/'report.md').write_text('\n'.join(lines)+'\n');partial.rename(out)


def expanded_report(root):
 root=Path(root);rows={};models={}
 expected={(42,'libero_10',t,e) for t in range(10) for e in range(50)}
 for label in ('concat','aligned'):
  d=read(root/f'expanded_{label}_seed42/summary.json');r=d['episodes']
  if len(r)!=500 or {identity(x) for x in r}!=expected or d['mode']!='final_rollout':raise ValueError('Incomplete expanded grid')
  rows[label]=r;k=sum(bool(x['success']) for x in r);models[label]=dict(successes=k,episodes=500,sr=k/500,wilson95=wilson(k,500))
 evidence=dict(original_decision='inconclusive',new_decision=None,models=models,paired=paired(rows['concat'],rows['aligned']),episodes=rows,note='Expanded seed42 grid includes the original first10states/task; do not pool these executions as independent evidence. No follow-up threshold was preregistered.')
 lines=['# Expanded Long KV follow-up','','Original 300-episode decision remains **inconclusive**. No additional decision threshold was preregistered.','','| Model | Successes/500 | SR | Wilson95 |','|---|---|---|---|']
 for name,r in models.items():lines.append(f"| {name} | {r['successes']}/500 | {r['sr']:.2%} | {r['wilson95'][0]:.2%}–{r['wilson95'][1]:.2%} |")
 lines+=['',f"Paired exact McNemar: `{json.dumps(evidence['paired'])}`",'',evidence['note']]
 publish(root/'expanded_analysis',evidence,lines)
 report=root/'long_analysis/report.md'
 if report.exists():report.write_text(report.read_text()+'\n## Supplemental evaluation\n\n[Expanded 500-episode comparison](../expanded_analysis/report.md). The original decision is retained.\n')
 return evidence


def summarize_full_rows(rows):
 expected={(seed,s,t,e) for seed in (42,43,44) for s in SUITES for t in range(10) for e in range(10)}
 if len(rows)!=1200 or {identity(r) for r in rows}!=expected:raise ValueError('Incomplete or duplicate full evaluation identities')
 k=sum(bool(r['success']) for r in rows)
 return dict(successes=k,episodes=1200,sr=k/1200,wilson95=wilson(k,1200),
   seeds={str(seed):sum(bool(r['success']) for r in rows if r['base_seed']==seed) for seed in (42,43,44)},
   suites={s:dict(successes=sum(bool(r['success']) for r in rows if r['suite']==s),episodes=300) for s in SUITES},
   per_task={f'{s}:{t}':sum(bool(r['success']) for r in rows if r['suite']==s and identity(r)[2]==t) for s in SUITES for t in range(10)})


def full_report(cr,mr):
 roots=[Path(cr),Path(mr)];models={}
 for root,labels in zip(roots,[('full_41','full_33','full_dense12'),('full_14','full_22','full_dense30')]):
  for label in labels:
   run=root/label;rows=[]
   for seed in (42,43,44):
    d=read(run/f'eval_seed{seed}/summary.json')
    if d['mode']!='final_rollout' or d['checkpoint_step']!=21700:raise ValueError('Nonfinal full evaluation')
    rows.extend(d['episodes'])
   result=summarize_full_rows(rows);train=run/'train';timing=read(train/'timing.json');manifest=read(train/'manifest.json')
   metrics=[json.loads(x) for x in (train/'metrics.jsonl').read_text().splitlines()];metrics=[r for r in metrics if 'loss_video' in r]
   if timing['status']!='complete' or len(metrics)!=21700 or [r['update'] for r in metrics]!=list(range(1,21701)) or timing['windows_seen']!=2777130:raise ValueError('Incomplete full training evidence')
   keys=('loss_video','loss_action','grad_norm');mean=lambda rr:{k:float(np.mean([r[k] for r in rr])) for k in keys}
   result.update(timing=timing,training_gpu_hours=2*timing['elapsed_training_seconds']/3600,manifest=manifest,first100=mean(metrics[:100]),last100=mean(metrics[-100:]),epochs={str(e):mean([r for r in metrics if r['epoch']==e]) for e in range(1,11)},episodes=rows)
   models[label]=result
 evidence=dict(models=models,training_seed=42,evaluation_seeds=[42,43,44],limitations='Single training seed. Episode Wilson intervals do not account for task/state clustering.')
 lines=['# Full LIBERO campaign results','','All models: fresh canonical initialization, global batch128, training seed42,10epochs,21700updates,2777130real windows. Three evaluation seeds42/43/44;400episodes each.','','| Model | Seed42/43/44 successes out of400 | Pooled SR | Training h | Training GPUh |','|---|---|---|---|---|']
 for label,r in models.items():lines.append(f"| {label} | {'/'.join(str(r['seeds'][str(s)]) for s in (42,43,44))} | {r['sr']:.2%} | {r['timing']['elapsed_training_seconds']/3600:.2f} | {r['training_gpu_hours']:.2f} |")
 lines+=['','## Per-suite successes out of300','','| Model | Spatial | Object | Goal | Long |','|---|---|---|---|---|']
 for label,r in models.items():lines.append('| '+label+' | '+' | '.join(str(r['suites'][s]['successes']) for s in SUITES)+' |')
 lines+=['','All per-task outcomes, epoch/first100/last100 losses, manifests, timing, and raw episode identities are in evidence.json.','',evidence['limitations']]
 publish(roots[0]/'full_analysis',evidence,lines)
 return evidence


def main():
 p=argparse.ArgumentParser();p.add_argument('--concat-root',required=True);p.add_argument('--mix-root');p.add_argument('--expanded',action='store_true');a=p.parse_args()
 if a.expanded:expanded_report(a.concat_root)
 else:full_report(a.concat_root,a.mix_root)
if __name__=='__main__':main()
