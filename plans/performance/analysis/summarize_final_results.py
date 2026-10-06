"""Rebuild numeric tables and figures from the archived production evidence."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'plans/evidence/final_results_20261006'
RAW=OUT/'raw'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
RUNS={
 'v0':('v0_scratch_fast_bs128_20261005','v0_auto_eval_job4689/inference'),
 'v1':('v1_bs128_job4691/train','v1_bs128_job4691/inference'),
 'dense_s12':('dense_v2_job4659_20261006/dense_s12_train','dense_v2_job4659_20261006/dense_s12_evaluation'),
 'v2':('dense_v2_job4659_20261006/v2_train','dense_v2_job4659_20261006/v2_evaluation')}
LABELS={'v0':'v0','v1':'v1','dense_s12':'Dense-S12','v2':'v2'}
COLORS={'v0':'#147d64','v1':'#bd6a18','dense_s12':'#3466b3','v2':'#a13c8d'}
read=lambda p:json.loads(p.read_text())

def weighted(rows,key):
 return sum(r[key]*r['windows'] for r in rows)/sum(r['windows'] for r in rows)

def wilson(k,n=100,z=1.959963984540054):
 p=k/n;denom=1+z*z/n;c=(p+z*z/(2*n))/denom
 h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
 return [c-h,c+h]

def csv_write(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

report={};epochs=[];episodes=[];summaries={};all_records={};validation={}
for version,(tr,ev) in RUNS.items():
 p,q=RAW/tr,RAW/ev
 m,t,s,em=read(p/'manifest.json'),read(p/'timing.json'),read(q/'summary.json'),read(q/'manifest.json')
 entries=[json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines()]
 rows=[r for r in entries if 'loss_video' in r and 'update' in r]
 val=[r for r in entries if r.get('event')=='validation']
 assert len(rows)==7250 and [r['update'] for r in rows]==list(range(1,7251))
 assert sum(r['windows'] for r in rows)==926780 and len(val)==10
 assert all(math.isfinite(r[k]) for r in rows for k in ['loss_video','loss_action','grad_norm','seconds'])
 assert sum(e['success'] for e in s['episodes'])==s['successes']
 warm=[r for r in rows if r['epoch']>1 and r['windows']==128]
 final=rows[-100:];initial=rows[:100]
 report[version]=dict(label=LABELS[version],successes=s['successes'],episodes=100,success_rate=s['success_rate'],
  wilson_95=wilson(s['successes']),train_seconds=t['elapsed_training_seconds'],gpu_count=m['world_size'],
  training_gpu_hours=t['elapsed_training_seconds']*m['world_size']/3600,
  real_windows_per_second=926780/t['elapsed_training_seconds'],
  backend=m.get('backend','ddp'),microbatch=m['microbatch'],accumulation=m['gradient_accumulation'],
  loops=em['loops'],parameters=m['policy_parameters'],seed=m['seed'],fresh=m['resume'] is None,
  warm_full_update_mean_seconds=statistics.mean(r['seconds'] for r in warm),
  warm_full_update_median_seconds=statistics.median(r['seconds'] for r in warm),
  peak_allocated_decimal_gb=max(r['peak_allocated_gb'] for r in rows),
  peak_reserved_decimal_gb=max(r['peak_reserved_gb'] for r in rows),
  first100_video=weighted(initial,'loss_video'),first100_action=weighted(initial,'loss_action'),
  last100_video=weighted(final,'loss_video'),last100_action=weighted(final,'loss_action'),
  first_validation_mse=val[0]['heldout_action_mse'],final_validation_mse=val[-1]['heldout_action_mse'],
  final_exit_video=weighted(final,f"exit{em['loops']}/video"),
  final_exit_action=weighted(final,f"exit{em['loops']}/action"),
  mean_episode_seconds=statistics.mean(e['duration_seconds'] for e in s['episodes']),
  mean_episode_policy_steps=statistics.mean(e['steps'] for e in s['episodes']),
  mean_replans=statistics.mean(e['replans'] for e in s['episodes']),
  failures_timeout=sum(not e['success'] and e['timeout'] for e in s['episodes']),
  successes_during_settling=sum(e['success'] and e['steps']==0 for e in s['episodes']),
  training_revision=m['git_revision'],evaluation_revision=em['git_revision'],checkpoint_sha256=s['checkpoint_sha256'])
 for epoch in range(1,11):
  rr=[r for r in rows if r['epoch']==epoch]
  assert len(rr)==725 and sum(r['windows'] for r in rr)==92678
  assert rr[-1]['windows']==6
  epochs.append(dict(version=version,epoch=epoch,updates=len(rr),real_windows=92678,
   update_compute_seconds=sum(r['seconds'] for r in rr),video_loss=weighted(rr,'loss_video'),action_loss=weighted(rr,'loss_action'),
   final_exit_video_loss=weighted(rr,f"exit{em['loops']}/video"),final_exit_action_loss=weighted(rr,f"exit{em['loops']}/action"),
   heldout_action_mse=val[epoch-1]['heldout_action_mse']))
 for e in s['episodes']:
  local_video=q/'videos'/Path(e['video']).name
  episodes.append(dict(version=version,task_id=e['task_id'],task=e['task_description'],episode_index=e['episode_index'],
   initial_state_index=e['initial_state_index'],seed=e['seed'],success=e['success'],timeout=e['timeout'],
   policy_steps=e['steps'],replans=e['replans'],duration_seconds=e['duration_seconds'],
   video_server=e['video']))
 summaries[version]=s;all_records[version]=rows;validation[version]=val

tasks=[]
for task in range(10):
 name=next(e['task'] for e in episodes if e['task_id']==task)
 tasks.append(dict(task_id=task,task=name,**{v:summaries[v]['per_task'][str(task)]['successes'] for v in RUNS}))
paired=[]
for a,b in [('v0','dense_s12'),('v0','v1'),('v1','v2'),('dense_s12','v2')]:
 aa={(e['task_id'],e['episode_index']):e for e in summaries[a]['episodes']}
 bb={(e['task_id'],e['episode_index']):e for e in summaries[b]['episodes']}
 assert aa.keys()==bb.keys()
 assert all((e['seed'],e['initial_state_index'])==(bb[k]['seed'],bb[k]['initial_state_index']) for k,e in aa.items())
 paired.append(dict(model_a=a,model_b=b,both_success=sum(e['success'] and bb[k]['success'] for k,e in aa.items()),
  only_a_success=sum(e['success'] and not bb[k]['success'] for k,e in aa.items()),
  only_b_success=sum(not e['success'] and bb[k]['success'] for k,e in aa.items()),
  both_fail=sum(not e['success'] and not bb[k]['success'] for k,e in aa.items())))

csv_write('models.csv',list(report.values()))
csv_write('epochs.csv',epochs);csv_write('tasks.csv',tasks);csv_write('episodes.csv',episodes);csv_write('paired_outcomes.csv',paired)
csv_write('failed_episodes.csv',[e for e in episodes if not e['success']])
(OUT/'analysis.json').write_text(json.dumps(dict(models=report,epochs=epochs,tasks=tasks,paired_outcomes=paired),indent=2))

plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,4))
for i,(v,r) in enumerate(report.items()):
 lo,hi=r['wilson_95'];p=r['success_rate']
 axes[0].bar(i,p*100,color=COLORS[v]);axes[0].errorbar(i,p*100,yerr=[[(p-lo)*100],[(hi-p)*100]],color='black',capsize=4)
 axes[0].text(i,p*100-10,f"{r['successes']}/100",ha='center',fontsize=10,color='white',fontweight='bold')
 axes[1].bar(i,r['train_seconds']/3600,color=COLORS[v]);axes[1].text(i,r['train_seconds']/3600+.1,f"{r['gpu_count']} GPUs",ha='center',fontsize=9)
for ax in axes:ax.set_xticks(range(4),[LABELS[v] for v in RUNS])
axes[0].set(title='LIBERO-Long success (95% Wilson intervals)',ylabel='Success (%)',ylim=(0,110))
axes[1].set(title='Actual training wall time',ylabel='Hours',ylim=(0,12))
fig.tight_layout();fig.savefig(FIG/'results_and_runtime.png');plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(14,4))
for v in RUNS:
 rows=all_records[v]
 x=[];ys=[[],[]]
 for start in range(0,len(rows),100):
  rr=rows[start:start+100];x.append(rr[-1]['update'])
  ys[0].append(weighted(rr,f"exit{report[v]['loops']}/video"));ys[1].append(weighted(rr,f"exit{report[v]['loops']}/action"))
 for j in (0,1):axes[j].plot(x,ys[j],label=LABELS[v],color=COLORS[v])
 axes[2].plot(range(1,11),[r['heldout_action_mse'] for r in validation[v]],marker='.',label=LABELS[v],color=COLORS[v])
for ax,title in zip(axes,['Final-exit video training loss','Final-exit action training loss','Held-out action MSE (8 fixed windows)']):
 ax.set_title(title);ax.set_yscale('log');ax.grid(alpha=.2)
axes[0].set_xlabel('Update (100-update bins)');axes[1].set_xlabel('Update (100-update bins)');axes[2].set_xlabel('Epoch');axes[0].legend()
fig.tight_layout();fig.savefig(FIG/'learning_curves.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
mat=[[t[v] for v in RUNS] for t in tasks];im=ax.imshow(mat,vmin=0,vmax=10,cmap='YlGn')
ax.set_xticks(range(4),[LABELS[v] for v in RUNS]);ax.set_yticks(range(10),[f'Task {i}' for i in range(10)])
for i,row in enumerate(mat):
 for j,value in enumerate(row):ax.text(j,i,f'{value}/10',ha='center',va='center',color='white' if value>=9 else 'black')
ax.set_title('Per-task success');fig.colorbar(im,ax=ax,label='Successful episodes');fig.tight_layout();fig.savefig(FIG/'per_task_success.png');plt.close(fig)
print(json.dumps(report,indent=2));print('paired',paired)
