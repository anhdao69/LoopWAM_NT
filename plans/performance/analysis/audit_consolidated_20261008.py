"""Read-only export of production results; run from the server FastWAM root."""
import datetime,json,math,statistics
from pathlib import Path
ROOT=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt')
train_paths={
'v0_original':'v0_scratch_fast_bs128_20261005',
'v1':'v1_bs128_job4691/train',
'dense_s12':'dense_v2_job4659_20261006/dense_s12_train',
'v2':'dense_v2_job4659_20261006/v2_train',
'dense_s30':'dense_s30_job4719_20261006/dense_s30_train',
'v0_full':'v0_full_libero_job4659_20261006/train',
'v0_4_1':'v0_v4a1_job4728_20261007/train',
'v0_4_2':'loop_grid_job4728_20261008/v4a2/train',
'v0_2_2':'loop_grid_job4659_20261008/v2a2/train',
'v0_1_4':'loop_grid_job4659_20261008/v1a4/train',
'v0_4_4_repeat':'loop_v4a4_repeat_job4728_20261008/v4a4/train'}
eval_paths={
'v0_original':'v0_auto_eval_job4689/inference','v1':'v1_bs128_job4691/inference',
'dense_s12':'dense_v2_job4659_20261006/dense_s12_evaluation','v2':'dense_v2_job4659_20261006/v2_evaluation',
'dense_s30':'dense_s30_job4719_20261006/dense_s30_evaluation',
'v0_4_1':'v0_v4a1_job4728_20261007/evaluation',
'v0_4_2':'loop_grid_job4728_20261008/v4a2/evaluation',
'v0_2_2':'loop_grid_job4659_20261008/v2a2/evaluation',
'v0_1_4':'loop_grid_job4659_20261008/v1a4/evaluation',
'v0_4_4_repeat':'loop_v4a4_repeat_job4728_20261008/v4a4/evaluation',
'libero_pro':'libero_pro_job4728_20261007'}
for seed in (42,43,44):eval_paths[f'v0_eval_seed_{seed}']=f'v0_multiseed_job4728_worker1_20261007/seed_{seed}'
for suite in ('spatial','object','goal','10'):eval_paths[f'v0_full_{suite}']=f'v0_full_libero_job4659_20261006/evaluation_libero_{suite}'
def read(p):return json.loads(p.read_text()) if p.exists() else None
out=dict(audit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(ROOT),training={},evaluation={},ancillary={})
for name,rel in train_paths.items():
 p=ROOT/rel;m=read(p/'manifest.json');t=read(p/'timing.json')
 rows=[]
 metrics=p/'metrics.jsonl'
 if metrics.exists():
  for line in metrics.read_text().splitlines():
   try:x=json.loads(line)
   except ValueError:continue
   if 'loss_video' in x and 'update' in x:rows.append(x)
 def avg(rs,k):return sum(r[k]*r.get('windows',128) for r in rs)/sum(r.get('windows',128) for r in rs) if rs else None
 loss={k:{'first100':avg(rows[:100],k),'last100':avg(rows[-100:],k)} for k in ('loss_video','loss_action')} if rows else {}
 out['training'][name]=dict(path=rel,manifest=m,timing=t,trainer_state=read(p/'trainer_state.json'),
  pipeline=read(p.parent/'pipeline_status.json'),metrics_count=len(rows),
  ordered_updates=bool(rows) and [r['update'] for r in rows]==list(range(1,len(rows)+1)),
  finite=bool(rows) and all(math.isfinite(r[k]) for r in rows for k in ('loss_video','loss_action','grad_norm')),
  loss=loss,checkpoint_bytes=(p/'latest.pt').stat().st_size if (p/'latest.pt').exists() else None,
  epochs=[{'epoch':e,'updates':len(rs),'video':avg(rs,'loss_video'),'action':avg(rs,'loss_action')} for e in range(1,11) if (rs:=[r for r in rows if r.get('epoch')==e])])
for name,rel in eval_paths.items():
 p=ROOT/rel;s=read(p/'summary.json');m=read(p/'manifest.json') or read(p/'evaluation/manifest.json');rows=s.get('episodes',[]) if s else []
 keys=[(r.get('suite'),r.get('category'),r.get('task_id'),r.get('episode_index'),r.get('variant_id')) for r in rows]
 out['evaluation'][name]=dict(path=rel,summary=s,manifest=m,records=len(rows),unique_keys=len(set(keys)),
  recounted_successes=sum(bool(r.get('success')) for r in rows),
  nonempty_videos=sum(Path(r.get('video','')).is_file() and Path(r['video']).stat().st_size>0 for r in rows),
  summary_mtime_utc=datetime.datetime.fromtimestamp((p/'summary.json').stat().st_mtime,datetime.timezone.utc).isoformat() if s else None)
for rel in ('v0_multiseed_job4728_worker1_20261007/results.json','v0_multiseed_job4728_worker1_20261007/pipeline_status.json','v0_full_libero_job4659_20261006/summary.json','libero_pro_job4728_20261007/pipeline_status.json'):
 out['ancillary'][rel]=read(ROOT/rel)
out['robustness_files']=[str(p.relative_to(ROOT)) for p in ROOT.glob('*') if any(s in p.name.lower() for s in ('plus','pro_','latency'))]
print(json.dumps(out,indent=2,allow_nan=False))
