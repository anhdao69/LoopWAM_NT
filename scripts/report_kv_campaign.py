#!/usr/bin/env python3
"""Generate measured Long KV comparison; never turn missing runs into results."""
import argparse,json,math
from pathlib import Path
from scipy.stats import binomtest,norm
import numpy as np


def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n
 center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [0.0 if k==0 else center-half,1.0 if k==n else center+half]


def identity(row):
 return (row['base_seed'],row['suite'],row.get('task',row.get('task_id')),row.get('episode',row.get('episode_index')))


def paired(a,b):
 aa={identity(r):bool(r['success']) for r in a};bb={identity(r):bool(r['success']) for r in b}
 if len(aa)!=len(a) or len(bb)!=len(b) or aa.keys()!=bb.keys():raise ValueError('Paired comparison requires identical unique episode identities')
 wins=sum(aa[k] and not bb[k] for k in aa);losses=sum(bb[k] and not aa[k] for k in aa)
 return dict(n=len(aa),candidate_only=wins,baseline_only=losses,p_exact=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.)


def two_proportion(k1,n1,k2,n2):
 p=(k1+k2)/(n1+n2);se=math.sqrt(p*(1-p)*(1/n1+1/n2));z=(k1/n1-k2/n2)/se if se else 0.
 return dict(z=z,p_two_sided=float(2*norm.sf(abs(z))),assumption='Unpaired episode-level approximation; ignores task/state clustering')


def decision(k,n):
 if n!=300:raise ValueError('Preregistered decision requires 300 episodes')
 return 'access hypothesis supported' if k>=267 else 'action-depth hypothesis supported' if k<=252 else 'inconclusive'


def read(p):return json.loads(Path(p).read_text())
def new_rows(root,label):
 rows=[]
 for seed in (42,43,44):
  d=read(root/label/f'eval_seed{seed}/summary.json')
  if d['mode']!='final_rollout' or d['total_episodes']!=100:raise ValueError('Incomplete new evaluation')
  rows.extend(d['episodes'])
 return rows


def summarize(rows):
 if len(rows)!=300 or {r['base_seed'] for r in rows}!={42,43,44} or len({identity(r) for r in rows})!=300:raise ValueError('Require three complete unique evaluation seeds')
 k=sum(bool(r['success']) for r in rows)
 return dict(successes=k,episodes=300,pooled_sr=k/300,wilson95=wilson(k,300),
     seed_successes={str(s):sum(bool(r['success']) for r in rows if r['base_seed']==s) for s in (42,43,44)},
     per_task={str(t):sum(bool(r['success']) for r in rows if identity(r)[2]==t) for t in range(10)})


def training_summary(path):
 records=[json.loads(x) for x in (path/'metrics.jsonl').read_text().splitlines()];updates=[r for r in records if 'update' in r and 'loss_video' in r]
 if len(updates)!=7250 or [r['update'] for r in updates]!=list(range(1,7251)):raise ValueError('Incomplete training update evidence')
 keys=['loss_video','loss_action','grad_norm']
 mean=lambda rows:{k:float(np.mean([r[k] for r in rows])) for k in keys}
 timing=read(path/'timing.json')
 return dict(timing=timing,training_gpu_hours=2*timing['elapsed_training_seconds']/3600,first100=mean(updates[:100]),last100=mean(updates[-100:]),epochs={str(e):mean([r for r in updates if r['epoch']==e]) for e in range(1,11)},diagnostics=[r for r in records if r.get('event')=='kv_diagnostics'])


def main():
 p=argparse.ArgumentParser();p.add_argument('--concat-root',required=True);p.add_argument('--mix-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 cr,mr,final=Path(a.concat_root),Path(a.mix_root),Path(a.output)
 if final.exists():raise ValueError('Refusing report overwrite')
 out=final.with_name(final.name+'.partial');out.mkdir(parents=True,exist_ok=False)
 baseline=read(mr.parent/'control_aligned_v4a1/summary.json')['episodes']
 concat=new_rows(cr,'concat_long');mix=new_rows(mr,'mix_long')
 rows={'aligned':baseline,'concat':concat,'mix':mix};results={k:summarize(v) for k,v in rows.items()}
 if results['aligned']['seed_successes']['42']!=81:raise ValueError('Aligned seed42 did not reproduce historical baseline')
 tests={mode:{**{str(s):paired([r for r in rows[mode] if r['base_seed']==s],[r for r in baseline if r['base_seed']==s]) for s in (42,43,44)},'pooled':paired(rows[mode],baseline),'unpaired_vs_original44':two_proportion(results[mode]['successes'],300,275,300)} for mode in ('concat','mix')}
 training={mode:training_summary(root/f'{mode}_long/train') for mode,root in [('concat',cr),('mix',mr)]}
 latency={mode:{flavor:read(root/f'{mode}_long/latency_{flavor}.json') for flavor in ('eager','compiled') if (root/f'{mode}_long/latency_{flavor}.json').exists()} for mode,root in [('concat',cr),('mix',mr)]}
 for label in ('aligned41','original44'):
  latency[label]={flavor:read(cr.parent/f'latency_{label}_{flavor}.json') for flavor in ('eager','compiled') if (cr.parent/f'latency_{label}_{flavor}.json').exists()}
 repeat=read(cr.parent/'control_repeat_v4a4/summary.json');repeat_seeds={str(v['seed']):v['successes'] for v in repeat['rounds'].values()};repeat_seeds['42']=91
 evidence=dict(results=results,tests=tests,training=training,latency=latency,episodes=rows,repeat44=repeat_seeds,original44=dict(seed_successes={'42':96,'43':88,'44':91},pooled_sr=275/300,wilson95=wilson(275,300)),decision=decision(results['concat']['successes'],300),limitations=['One training seed42; training-seed variance not measured.','Wilson and two-proportion calculations treat episodes as independent; shared tasks/states create clustering.','Threshold rule is preregistered heuristic, not proof of mechanism.'])
 (out/'evidence.json').write_text(json.dumps(evidence,indent=2))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 for mode in ('concat','mix'):
  diagnostics=training[mode]['diagnostics']
  if len(diagnostics)!=10:raise ValueError('Missing epoch diagnostics')
  fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
  for layer,ax in enumerate(axes.flat):
   values=np.array([d['attention_mass'][str(layer)] if mode=='concat' else np.array(d['weights'])[layer].mean(0) for d in diagnostics])
   for column in range(values.shape[1]):ax.plot(range(1,11),values[:,column],label=f'Loop {column+1}' if column<4 else 'Action')
   ax.set_title(f'Core layer {layer+1}');ax.set_xlabel('Epoch');ax.set_ylim(0,1)
  axes.flat[0].legend();fig.tight_layout();fig.savefig(out/f'{mode}_weights.png');plt.close(fig)
 lines=['# LoopWAM 4/1 all-loop KV measured results','',f"Decision: **{evidence['decision']}**.",'','Preregistered: ≥89% supports access; ≤84% supports action depth; otherwise inconclusive and expand to 500 initial-state rollouts. One training seed42.','','| Model | KV | Video/action depth | Block calls/chunk | Parameters | Seed42/43/44 | Pooled SR (Wilson95) |','|---|---|---|---|---|---|---|']
 for mode,r in results.items():
  ci=r['wilson95'];seeds='/'.join(str(r['seed_successes'][str(s)]) for s in (42,43,44))
  lines.append(f"| 4/1 | {mode} | 30/12 | 150 | {584536135+(288 if mode=='mix' else 0):,} | {seeds} | {r['pooled_sr']:.2%} ({ci[0]:.2%}–{ci[1]:.2%}) |")
 for name,seeds in [('4/4 original',{'42':96,'43':88,'44':91}),('4/4 repeat',repeat_seeds)]:
  k=sum(seeds.values());ci=wilson(k,300);lines.append(f"| {name} | aligned | 30/30 | 330 | 584,536,135 | {'/'.join(str(seeds[str(s)]) for s in (42,43,44))} | {k/300:.2%} ({ci[0]:.2%}–{ci[1]:.2%}) |")
 lines+=['','## Paired exact McNemar','','| Candidate | Seed/grid | Candidate-only wins | Baseline-only wins | Exact p |','|---|---|---|---|---|']
 for mode,values in tests.items():
  for key in ('42','43','44','pooled'):
   t=values[key];lines.append(f"| {mode} | {key} | {t['candidate_only']} | {t['baseline_only']} | {t['p_exact']:.6g} |")
 lines+=['','## Training and diagnostics','']
 for mode,t in training.items():lines += [f"### {mode}",'',f"Training: {t['timing']['elapsed_training_seconds']/3600:.2f} h; {t['training_gpu_hours']:.2f} GPU-hours.",'',f"First/last100 losses and per-epoch losses: `{json.dumps({k:t[k] for k in ('first100','last100','epochs')})}`",'',f'![{mode} diagnostics]({mode}_weights.png)','']
 lines+=['## Per-task success out of30','','| Task | Aligned | Concat | Mix |','|---|---|---|---|']
 for task in range(10):lines.append('| '+str(task)+' | '+' | '.join(str(results[m]['per_task'][str(task)]) for m in ('aligned','concat','mix'))+' |')
 lines+=['','## Latency','','| Model | Mode | Total ms | Video prefill ms | Action denoising ms |','|---|---|---|---|---|']
 for label,flavors in latency.items():
  for flavor,value in flavors.items():
   r=value['models']['loopwam_v0'];stage=r['stage_mean_ms']
   lines.append(f"| {label} | {flavor} | {r['mean_ms']:.3f} | {stage['video_prefill']:.3f} | {stage['action_denoising']:.3f} |")
 lines+=['','## Unpaired comparison against original4/4','']
 for mode,t in tests.items():lines.append(f"- {mode}: {json.dumps(t['unpaired_vs_original44'])}")
 lines+=['','## Statistical limitations','',*['- '+x for x in evidence['limitations']]]
 (out/'report.md').write_text('\n'.join(lines)+'\n')
 out.rename(final)
if __name__=='__main__':main()
