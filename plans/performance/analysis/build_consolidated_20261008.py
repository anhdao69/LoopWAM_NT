"""Validate the frozen audit and render consolidated report, CSVs and figures."""
import csv,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
E=ROOT/'plans/evidence/consolidated_20261008'
d=json.loads((E/'audit.json').read_text());sup=json.loads((E/'robustness_audit.json').read_text())
old=json.loads((ROOT/'plans/evidence/final_results_20261006/runtime_summary.json').read_text())
lat=json.loads((ROOT/'plans/evidence/robustness_20261007/latency_aggregate.json').read_text())
labels={'v0_original':'v0 4/4 original','v1':'v1 4/4','dense_s12':'Dense-S12','v2':'v2 4/4','dense_s30':'Dense-S30','v0_4_1':'v0 4/1','v0_4_2':'v0 4/2','v0_2_2':'v0 2/2','v0_1_4':'v0 1/4','v0_full':'v0 four-suite','v0_4_4_repeat':'v0 4/4 fresh repeat'}
order=['dense_s12','dense_s30','v0_original','v1','v2','v0_4_1','v0_4_2','v0_2_2','v0_1_4']
def duration(s):
 if s is None:return '—'
 s=round(s);return f'{s//3600}h {(s%3600)//60:02d}m {s%60:02d}s'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def evtime(n):
 if n in ('v0_original','v1','dense_s12','v2'):return old['v0' if n=='v0_original' else n]['evaluation_wall_seconds']
 pipe=d['training'][n]['pipeline'] or {};st=pipe.get('stage_durations_seconds',{})
 return st.get('evaluation',st.get('dense_s30_evaluation'))
def wilson(k,n):
 z=1.96;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;e=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return f'{100*(c-e):.1f}–{100*(c+e):.1f}%'
for n,r in d['training'].items():
 assert r['finite'] and r['ordered_updates'],n
 t=r['timing'];m=r['manifest']
 if t['status']=='complete':
  assert r['metrics_count']==t['completed_updates']==m['planned_updates'],n
  assert t['windows_seen']==m['planned_windows'] and r['checkpoint_bytes']>0,n
baseline=d['training']['v0_original']['manifest']
for n,r in d['training'].items():
 if n=='v0_full':continue
 m=r['manifest']
 for k in ('seed','global_batch','epochs','train_windows','planned_updates','planned_windows','asset_sha256'):
  assert m[k]==baseline[k],(n,k)
 for k in ('train_episodes','validation_episodes','normalization_sha256','camera_order','video_offsets','action_offsets','content_files'):
  assert m['data'][k]==baseline['data'][k],(n,k)
for n,r in d['evaluation'].items():
 s=r['summary']
 if not s:continue
 assert r['records']==r['unique_keys']==r['nonempty_videos']==s['total_episodes'],n
 assert r['recounted_successes']==s['successes'] and abs(s['success_rate']-s['successes']/s['total_episodes'])<1e-12,n
 assert all(x['checkpoint_sha256']==s['checkpoint_sha256'] for x in s['episodes']),n
 if r['manifest']:assert s['checkpoint_sha256']==r['manifest']['checkpoint_sha256'],n
 if n in order:assert s['checkpoint_step']==7250,n
models=[]
for n in order:
 r=d['training'][n];m=r['manifest'];t=r['timing'];s=d['evaluation'][n]['summary']
 models.append(dict(run=n,label=labels[n],successes=s['successes'],episodes=s['total_episodes'],sr=s['success_rate'],parameters=m['policy_parameters'],gpus=m['world_size'],training_seconds=t['elapsed_training_seconds'],evaluation_seconds=evtime(n),gpu_hours=t['elapsed_training_seconds']*m['world_size']/3600,backend=m.get('backend',old.get('v0' if n=='v0_original' else n,{}).get('backend')),microbatch=m['microbatch'],accumulation=m['gradient_accumulation'],checkpoint_sha256=s['checkpoint_sha256']))
def csvwrite(name,rows):
 with (E/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csvwrite('models.csv',models)
episodes=[]
for n,r in d['evaluation'].items():
 for x in (r['summary'] or {}).get('episodes',[]):
  episodes.append(dict(run=n,suite=x.get('suite'),category=x.get('category',''),task_id=x['task_id'],episode_index=x['episode_index'],seed=x.get('seed'),success=x['success'],steps=x.get('steps'),video=x['video'],checkpoint_sha256=x['checkpoint_sha256']))
csvwrite('episodes.csv',episodes)
csvwrite('losses.csv',[dict(run=n,**{k+'_'+part:v[part] for k,v in r['loss'].items() for part in ('first100','last100')}) for n,r in d['training'].items()])
# Figures show complete runs only; GPU-hours make differing allocations visible.
fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
y=list(range(len(models)));names=[x['label'] for x in models]
axes[0].barh(y,[x['sr']*100 for x in models],color=['#64748b' if 'dense' in x['run'] else '#2563eb' for x in models]);axes[0].set_yticks(y,names);axes[0].invert_yaxis();axes[0].set_xlim(0,105);axes[0].set_xlabel('LIBERO-Long success (%) — 100 episodes/checkpoint')
for i,x in enumerate(models):axes[0].text(x['sr']*100+1,i,f"{x['sr']*100:.0f}%",va='center',fontsize=9)
axes[1].barh(y,[x['gpu_hours'] for x in models],color='#0d9488');axes[1].set_yticks(y,names);axes[1].invert_yaxis();axes[1].set_xlabel('Training H100 GPU-hours (excludes evaluation)')
fig.suptitle('Completed matched-data LIBERO-Long runs — one training seed')
fig.savefig(E/'long_results.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for n in order:
 es=d['training'][n]['epochs']
 for ax,k in zip(axes,('video','action')):ax.plot([e['epoch'] for e in es],[e[k] for e in es],label=labels[n]);ax.set_xlabel('Epoch');ax.set_ylabel(f'Mean {k} objective');ax.set_yscale('log')
axes[1].legend(fontsize=8,ncol=2);fig.suptitle('Real-window-weighted epoch losses; v1/v2 objectives combine exits')
fig.savefig(E/'loss_curves.png',dpi=170);plt.close(fig)
lines=[]
def add(s):lines.append(s.strip()+'\n')
add(f'''# LoopWAM — consolidated experimental results

**Audit snapshot: {d['audit_utc']} (October 8, 2026, approximately 08:53 EDT).**

This is the current consolidated report. It supersedes pending-result statements in older launch reports while preserving those documents as historical records. Results come from server production manifests, timing files, all logged training updates, evaluation summaries and episode records; supporting evidence and reproducible analysis are linked below. The fresh 4/4 repeat remains in progress at this snapshot.

## 1. Findings

- Original Long-only **v0 4/4: 96/100**, the best completed single evaluation among the matched Long runs. Evaluation seeds 42/43/44 give **96%, 88%, 91%**, averaging **91.67% ± 4.04 percentage points** (sample standard deviation) for that one trained checkpoint.
- **Dense-S30: 86%** with 1.416B policy parameters; original v0 uses 0.585B, about **58.7% fewer**. This is an encouraging checkpoint comparison, with no repeated-training-seed evidence yet.
- Recent **4/2 and 1/4 both score 86%**; **2/2 and 4/1 both score 81%**. The 1/4 model trained **36.8% faster than 4/2** on two H100s with the same fully populated VAE cache, but with different microbatches and accumulation.
- A separately trained **single four-suite v0 checkpoint achieves 388/400 (97%)** across Spatial/Object/Goal/Long. It trained on all 1,712 locally available demonstrations, so it is a distinct experiment from the 344/44 Long split.
- The four-suite checkpoint achieves **720/2,000 (36%) on LIBERO-Pro**. **LIBERO-Plus has no valid result** because its preflight failed.
- Measured policy-query latency: released FastWAM **279.90 ms** versus four-suite v0 **260.26 ms** mean, about **7.0% lower** for v0. This is a local timing comparison, with no matched FastWAM simulator SR in the audited results.

## 2. Architecture and training contract

LoopWAM stores 12 block pairs: three prelude, six shared core, three coda. Video/action loop counts specify core repetitions; effective expert depth is `3 + 6K + 3`. Dense-S12 executes 12 independent pairs once. Dense-S30 executes 30 independent pairs once, loading all 30 compact donor layers in order. All models use the native Wan VAE and the compact video/action widths described in the technical plan.

| Variant | Parameters | Video/action effective depth | Supervision |
|---|---:|---|---|
| Dense-S12 | 584,536,135 | 12 / 12 | Final only |
| Dense-S30 | 1,416,114,247 | 30 / 30 | Final only |
| v0 4/4 | 584,536,135 | 30 / 30 | Final only |
| v1 4/4 | 584,536,135 | 30 / 30 | All exits |
| v2 4/4 | 584,536,135 | 30 / 30 | All exits + core XSA |
| v0 4/1 | 584,536,135 | 30 / 12 | Final only |
| v0 4/2 | 584,536,135 | 30 / 18 | Final only |
| v0 2/2 | 584,536,135 | 18 / 18 | Final only |
| v0 1/4 | 584,536,135 | 12 / 30 | Final only |

For v1/v2, the last exit has weight 0.5; each of the first three exits has weight 1/6. For asymmetric v0, action core iteration `r` reads video iteration `max(1, Kv - Ka + r)`. Thus 4/1 reads video loop 4; 4/2 reads loops 3 and 4; 1/4 reuses loop 1 four times. In 1/4, video features are computed once with differentiable KV reuse, so action gradients propagate into the video graph. Actions see observation frames only. Tests cover cached/joint output and gradient agreement, future-frame isolation and checkpoint depth restoration.

Matched Long training: 388 local demonstrations split deterministically into **344 train / 44 validation**, 92,678 training windows/epoch, 11,602 validation windows, ten epochs, **7,250 updates and 926,780 real windows**, global batch 128, seed 42. Each epoch ends with six real windows; padded windows have zero contribution. Initialization uses the same canonical pretrained Wan compact donor artifact with a fresh optimizer; no production run resumes a smoke checkpoint. Changing architecture changes donor-layer usage and computation, even when initialization provenance is shared.

Precision: FP32 policy/master weights and Adam moments with BF16 compute. AdamW: learning rate 1e-4, betas (0.9, 0.95), epsilon 1e-8, weight decay 0.01, 5% warmup and cosine decay, gradient clipping at 1. Data normalization, cameras, video/action offsets and source data hashes were matched by launch gates. “Full dataset” here means the local export; the Long export contains 388 rather than the nominal 500 demonstrations.

## 3. Completed LIBERO-Long results

Every row below uses 100 rollouts of the final 10-epoch checkpoint, evaluation seed 42. Each of ten tasks uses initial states 0–9, a 700-policy-step cap and 30 settling steps, 32-action predictions, replanning every ten actions, ten denoising steps and CFG 1. The recorded episode seed is `42 + task_id*100000 + episode_index*1000`, with the replan index added for policy sampling. Actions use the checkpoint’s training normalizer. The interval is an approximate episode-level 95% Wilson interval, not a training-seed confidence interval.''')
add(table(['Model','SR','95% interval','H100s','Training','Evaluation','Train GPU-h'],[[m['label'],f"{m['successes']}/100 ({m['sr']:.0%})",wilson(m['successes'],100),m['gpus'],duration(m['training_seconds']),duration(m['evaluation_seconds']),f"{m['gpu_hours']:.2f}"] for m in models]))
add('''![Long success and GPU-hours](../evidence/consolidated_20261008/long_results.png)

Training time is `elapsed_training_seconds`, including validation/checkpoint work inside the training timer and excluding initial process/model setup. Evaluation time is the full recorded rollout stage including setup and video output. GPU-hours multiply training time by allocated GPU count. Setup, benchmarks, failed sizing trials, smoke checks and evaluation are excluded from training GPU-hours.

Raw wall times are not architecture-only comparisons: v0/v1/Dense-S30 and depth ablations use two H100s; Dense-S12/v2 use four. Earlier runs populated preprocessing caches during epoch 1, whereas the recent 4/1, 4/2, 2/2, 1/4 and fresh 4/4 runs reuse a fully populated frozen VAE cache. Different microbatch layouts also change random-number consumption despite the same seed. The new 4/4 repeat is intended to provide a closer runtime/control comparison under the recent setup.

### Backend and accumulation''')
add(table(['Model','Backend','Microbatch/GPU','Accumulation','GPUs'],[[m['label'],m['backend'],m['microbatch'],m['accumulation'],m['gpus']] for m in models]))
add('''The recent 4/2, 2/2, 1/4 and fresh 4/4 searches compared DDP, ZeRO-1 and ZeRO-2 while preserving parameter/optimizer precision and global batch. DDP won the tested configurations. Larger batches that exceeded GPU memory were rejected before production. Short benchmark rankings and full-run timing can differ because of storage and system variation; these measurements do not establish a universal optimum.

### Per-task Long success (out of ten)''')
add(table(['Task']+[labels[n] for n in order],[[str(i)]+[d['evaluation'][n]['summary']['per_task'][str(i)]['successes'] for n in order] for i in range(10)]))
tasks={x['task_id']:x.get('task_description',x.get('task_name','')) for x in d['evaluation']['v0_original']['summary']['episodes']}
add('\n'.join(f'{i}. {tasks[i]}' for i in range(10)))
add('''## 4. Loss behavior and numerical checks

The following objective values are **weighted by real windows over the first and last 100 updates** of each completed run. Last-update-only values are misleading because each Long epoch ends in a six-window batch. v1/v2 objectives aggregate multiple exits and are not identical to final-exit-only losses.''')
add(table(['Run','Video first100 → last100','Action first100 → last100'],[[labels[n]]+[f"{r['loss'][k]['first100']:.5f} → {r['loss'][k]['last100']:.5f}" for k in ('loss_video','loss_action')] for n,r in d['training'].items() if r['timing']['status']=='complete']))
add('''![Epoch loss curves](../evidence/consolidated_20261008/loss_curves.png)

All **86,950 updates across the ten completed production training runs** are present in order, without missing/duplicate update indices; video/action losses and gradient norms are finite. Both objectives decrease substantially. Loss improvement does not guarantee better closed-loop performance: Dense-S30 has lower final loss than v0 but scores 86% versus the original v0’s 96% at seed 42.

The original four-model audit also measured held-out normalized action MSE on only eight fixed validation windows (final v0 0.06274, v1 0.06754, Dense-S12 0.06977, v2 0.06086). This small open-loop diagnostic is not a substitute for simulator evaluation. Tail-batch hidden-state RMS logging is inflated by aggregation over padded physical microbatches; it should not be interpreted as activation explosion. See the [original detailed audit](LoopWAM_Final_Results_20261006.md) for final-exit losses, paired episode outcomes and that diagnostic caveat.

## 5. Evaluation-seed repeatability

One fixed original v0 checkpoint was evaluated again with seeds 42, 43 and 44. The task/initial-state grid is unchanged; diffusion/simulator seeds change. These are **evaluation seeds, not three training seeds**.''')
ms=d['ancillary']['v0_multiseed_job4728_worker1_20261007/pipeline_status.json']
add(table(['Evaluation seed','SR','Evaluation runtime'],[[seed,f"{d['evaluation'][f'v0_eval_seed_{seed}']['summary']['successes']}/100",duration(ms['stage_durations_seconds'][f'evaluation_seed_{seed}'])] for seed in (42,43,44)]))
add('''Combined: **275/300 (91.67%)**, sample SD across three seed-level scores **4.04 percentage points**. Seed 42 reproduced the original 96/100 aggregate result. The repeated seed-42 execution is not an independent new seed and should not be pooled with the original seed-42 result as if it doubled independent evidence. This variability is a reason to avoid treating small single-seed differences as conclusive.

## 6. One checkpoint trained on all four suites

The full-mixture v0 uses 4/4 loops, 584.5M parameters and one global action/state normalizer. It trains on every locally available demonstration: Spatial 434, Object 457, Goal 433, Long 388 (**1,712 total**). There is no held-out demonstration split in this full-mixture training configuration. One epoch has 277,713 windows; ten epochs complete **21,700 updates / 2,777,130 windows** at global batch 128 on four H100s.

This data volume and training scope differ from the Long-only controls. Its Long score must not be attributed solely to architecture or interpreted as an independent matched 344/44 training result.''')
full=d['training']['v0_full'];fp=full['pipeline']['stage_durations_seconds']
add(table(['Suite','Success','Evaluation runtime'],[[suite,f"{d['evaluation']['v0_full_'+key]['summary']['successes']}/100",duration(fp['evaluation_libero_'+key])] for suite,key in (('Spatial','spatial'),('Object','object'),('Goal','goal'),('Long','10'))]))
add(f"Overall **388/400 (97%)**. Training: **{duration(full['timing']['elapsed_training_seconds'])}**, **{full['timing']['elapsed_training_seconds']*4/3600:.2f} H100 GPU-hours**. Four-suite evaluation total: **{duration(sum(v for k,v in fp.items() if k.startswith('evaluation_')))}**. These standard-suite evaluations use the custom common 700-policy-step horizon, including the easier suites; protocol differences matter when comparing to external publications.")
add('''## 7. LIBERO-Pro and LIBERO-Plus

### LIBERO-Pro: completed

The **same single four-suite v0 checkpoint** is used across all four base suites and five perturbation categories. It is not four separately trained suite-specific models. Each category contains 40 tasks × 10 episodes = 400 rollouts; 2,000 total.''')
pro=d['evaluation']['libero_pro']['summary'];cats=pro['categories']
add(table(['Category','Successes / 400 episodes','SR'],[[name,sum(x['success'] for x in pro['episodes'] if x['category']==key),f"{cats[key]['success_rate']*100:.2f}%"] for name,key in [('Language','lan'),('Object','object'),('Environment','env'),('Task','task'),('Swap','swap')]]))
add('Per-base-suite Pro success rates, pooling 100 rollouts per category within each suite:')
def pro_rate(base,cat):
 rs=[x for x in pro['episodes'] if x['base_suite']==base and (cat is None or x['category']==cat)]
 return f"{100*sum(x['success'] for x in rs)/len(rs):.1f}%"
add(table(['Base suite','Language','Object','Environment','Task','Swap','All categories'],[[base]+[pro_rate(base,cat) for cat in ('lan','object','env','task','swap',None)] for base in ('libero_spatial','libero_object','libero_goal','libero_10')]))
pr=d['ancillary']['libero_pro_job4728_20261007/pipeline_status.json']
add(f'''Overall **720/2,000 = 36.00%**. Evaluation wall time: **{duration(pr['stage_durations_seconds']['evaluation'])}** on two H100s. All 2,000 unique suite/task/episode records and nonempty videos were verified. Category counts are 314/400 language, 252/400 object, 101/400 environment, 53/400 task and 0/400 swap.

Protocol: suite-specific caps of **220 Spatial / 280 Object / 300 Goal / 520 Long policy steps**, 30 settling steps, 10 diffusion steps, 32-action predictions and replanning every ten actions. Thus the 36% Pro result and 97% standard-suite result differ in both perturbations and horizons; their gap cannot be assigned entirely to robustness degradation. The **13.25% task-category score is not the overall Pro score**. No matched FastWAM Pro score or verified paper comparison is established by these artifacts.

The 0/400 swap result warrants inspection of category assets, task semantics and failure videos before making a causal claim about the model. Completed execution alone does not explain the failure mechanism.

### LIBERO-Plus: no valid score

The latest attempt failed during preflight with `wand.exceptions.MissingDelegateError: NoDecodeDelegateForThisImageFormat PNG`. Earlier attempts encountered a missing MagickWand library or were cancelled. There is no completed Plus rollout benchmark and no valid Plus success rate. This is an environment/preflight failure, not a measured 0% policy score. The failed queue is not running.

## 8. FastWAM versus LoopWAM latency

Timing uses one H100 80GB, batch one, two 224×224 cameras, 32-action horizon, ten diffusion steps, FP32 weights with BF16 compute, and cached text embeddings with padding masks. Each of two trials has five warmup and 50 timed queries; the table aggregates 100 measured queries/model. Timers synchronize CUDA and include observation preprocessing/transfers, online VAE, visual prefill and action denoising; simulator, text encoding and model loading are excluded.''')
add(table(['Checkpoint','Mean ms','Median ms','P95 ms','Trial means ms'],[[name]+[f"{lat[key][k]:.2f}" for k in ('mean_ms','p50_ms','p95_ms')]+[', '.join(f'{x:.2f}' for x in lat[key]['trial_means_ms'])] for name,key in [('Released FastWAM','fastwam'),('Four-suite LoopWAM v0 4/4','loopwam_v0')]]))
add('''LoopWAM has **19.63 ms lower mean latency (7.01%; approximately 1.075× throughput)** in this local measurement. The trial means vary, especially for FastWAM; this is a modest measured difference, not an architecture-independent guarantee. These milliseconds describe a policy query/action chunk, not one executed robot action. Peak allocated memory in those inference trials was approximately 39.07 decimal GB for FastWAM and 3.98 GB for v0, including the loaded model/runtime state in each trial.

The measured v0 checkpoint is the four-suite checkpoint, not the newer asymmetric variants. No isolated latency measurements are available for 4/1, 4/2, 2/2, 1/4 or Dense-S30. Their rollout wall times also reflect episode lengths, simulator rendering and video encoding, so they should not be substituted for query latency. The audited artifacts contain no matched local FastWAM success-rate evaluation; external published scores are intentionally not combined with these local results.

## 9. Fresh 4/4 repeat: ongoing at audit''')
repeat=d['training']['v0_4_4_repeat'];rt=repeat['timing']
add(f'''Allocation **4728.20**, worker-1, two H100s; source `8a29ffdce1537409e2b7e975e838d990ebbe068a`, the same source as the 4/2, 2/2 and 1/4 grid. Same seed 42, Long split, ten epochs, global batch 128, DDP microbatch 8 and accumulation 8. It initializes from the canonical donors and a fresh optimizer, with no checkpoint resume. The full frozen VAE cache is reused.

At the snapshot it had **{rt['completed_updates']}/7,250 updates**, with finite losses/gradients. Both native ten-update training checks and two simulator smoke episodes passed. Automatic final evaluation is queued for the final checkpoint, using the same 100-episode Long protocol. Estimated training remains roughly **7½ hours**, then **45–90 minutes** evaluation; the launch estimate was **17:00–17:45 EDT October 8**. This estimate is not a completed runtime or success rate. The repeat uses the same training seed and therefore does not establish across-training-seed variability.

## 10. Interpretation and remaining questions

1. **Weight sharing is promising:** original v0 achieved the highest observed Long seed-42 SR with 58.7% fewer policy parameters than Dense-S30. The original v0’s evaluation-seed range is 88–96%; broader claims require multiple fresh training seeds and matched systems settings.
2. **Additional objectives did not improve the measured final-exit score:** v1=82%, v2=81%, versus original v0=96%. Different GPU layouts and random-number consumption limit a strict attribution to all-exit loss or XSA.
3. **Video/action depth changes the accuracy–cost trade-off:** 1/4 ties 4/2 at 86% with lower measured training time; 2/2 also trains quickly but scores 81%. No dedicated inference-latency measurements support claims about runtime deployment benefits yet.
4. **Standard-suite success does not establish robustness:** the full-mixture checkpoint scores 97% on the standard evaluation and 36% on Pro under different horizons. Investigate swap failures and evaluate matched protocols before assigning causes.
5. **Next evidence to collect:** finish the same-setup 4/4 repeat, repeat training across seeds, measure per-variant query latency, inspect Pro failures, and repair Plus dependencies before reporting Plus results. This report does not launch extra work.

## 11. Audit scope, artifacts and reproducibility

The snapshot includes **ten completed training runs plus one ongoing repeat**, **17 completed evaluation executions**, **3,600 episode records**, and **3,600 nonempty server videos**. The original seed-42 rollout grid appears in two separate executions; these 3,600 records are an execution inventory, not 3,600 independent experimental conditions. Success totals were independently recounted from episode records. Unique keys include suite/category/task/episode, which is essential for Pro because task IDs repeat between base suites.

Every completed run’s update sequence, real-window count, final checkpoint existence and finite losses/gradients were checked. Evaluation episode hashes match their summary checkpoint hash; available evaluation manifests match summaries. Previous production pipelines validate the checkpoint contracts. This audit **does not recompute all multi-gigabyte checkpoint hashes or decode every video again**. The original 400-video audit included ffprobe validation; the new consolidated video check establishes presence and nonzero size only. No claims are made about the visual quality of every rollout.

Videos and model/optimizer checkpoints remain on the H100 server. The report, figures, all episode outcomes and paths, training/evaluation manifests, source/asset hashes, timing/loss summaries and analysis scripts are saved locally and pushed to GitHub. Older launch estimates remain historical; use completed measured times in this report for finished runs.

- [Full server audit snapshot](../evidence/consolidated_20261008/audit.json)
- [Robustness protocol and Plus failure evidence](../evidence/consolidated_20261008/robustness_audit.json)
- [Model comparison CSV](../evidence/consolidated_20261008/models.csv)
- [All 3,600 episodes and server video paths](../evidence/consolidated_20261008/episodes.csv)
- [Weighted loss summary CSV](../evidence/consolidated_20261008/losses.csv)
- [Latency raw trials and aggregate](../evidence/robustness_20261007/latency_aggregate.json)
- [Original four-model audit](LoopWAM_Final_Results_20261006.md)
- [Loop grid launch and speed tests](LoopWAM_v0_loop_grid_20261008.md)
- [Fresh 4/4 launch](LoopWAM_v0_V4A4_repeat_20261008.md)

Rebuild derived tables, checks and figures from the frozen evidence:

```bash
python plans/performance/analysis/build_consolidated_20261008.py
```

To capture a new server snapshot, run `audit_consolidated_20261008.py` on the H100 login host, archive its JSON output, and update the report timestamp. The frozen snapshot is intentionally kept reproducible.

### Checkpoint and source inventory
''')
add(table(['Run','Evaluated checkpoint SHA-256','Training source'],[[m['label'],'`'+m['checkpoint_sha256']+'`','`'+str((d['training'][m['run']]['pipeline'] or {}).get('source_revision',d['training'][m['run']]['manifest'].get('git_revision','see manifest')))+'`'] for m in models]))
add(f"Four-suite/Pro checkpoint: `{pro['checkpoint_sha256']}`. Server root: `{d['root']}`. Relative run paths and all manifests are included in the JSON audit.")
(ROOT/'plans/performance/LoopWAM_Consolidated_Results_20261008.md').write_text('\n'.join(lines))
assert len(episodes)==3600
assert sum(r['metrics_count'] for r in d['training'].values() if r['timing']['status']=='complete')==86950
print('Validated 86,950 completed training updates and 3,600 evaluated episodes; report/CSVs/plots written.')
