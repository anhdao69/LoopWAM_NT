import hashlib,json,math,os,sys,time
from pathlib import Path
sys.path.insert(0,'scripts')
from run_full_loop_queue import verify_train,source_hashes,read,write,SUITES
root=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt')
base=read(root/'v0_full_libero_job4659_20261006/train/manifest.json')
for job,pairs in ((4768,[[4,1],[1,4]]),(4769,[[3,3],[2,2]])):
 out=root/f'full_loop_job{job}'
 if (out/'release.json').exists():print(job,'already released');continue
 if not (out/'prepared.json').exists():print(job,'still preparing');continue
 prepared=read(out/'prepared.json');status=read(out/'status.json')
 assert status['stage']=='prepared_awaiting_release'
 assert status['source_revision']=='d166e8f7b89a4fe9fe807c1d1896e447a7963a88'
 assert prepared['source_hashes']==source_hashes()
 evals=read(out/'evaluation_benchmarks.json')
 assert len(evals)==4 and {e['workers_per_gpu'] for e in evals}=={1,2,4,8}
 assert all(e['max_action_difference']==0 for e in evals)
 assert prepared['evaluation']['workers_per_gpu']==8
 audits={}
 for pair in pairs:
  label=''.join(map(str,pair));config=prepared['configs'][label];directory=out/f'smoke_{label}'
  t=verify_train(directory,pair,True);m=read(directory/'manifest.json')
  assert config['pair']==pair and config['backend']==m['backend'] and config['microbatch']==m['microbatch']
  assert m['asset_sha256']==base['asset_sha256']
  assert set(m['data'])==set(base['data'])
  differences=[key for key in m['data'] if m['data'][key]!=base['data'][key] and key!='normalization_path']
  assert not differences,differences
  metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
  updates=[x for x in metrics if 'grad_norm' in x]
  assert len(updates)==10 and all(math.isfinite(x['grad_norm']) for x in updates)
  events=[]
  for line in (out/f'train_smoke_{label}.log').read_text().splitlines():
   try:events.append(json.loads(line))
   except json.JSONDecodeError:pass
  initial=[e for e in events if 'optimizer_state_entries_before_training' in e]
  assert len(initial)==1 and initial[0]['optimizer_state_entries_before_training']==0
  assert any(e.get('event')=='vae_anchor_pass' for e in events)
  evaluation=read(out/f'smoke_eval_{label}'/'summary.json')
  assert evaluation['total_episodes']==8 and evaluation['video_loops']==pair[0] and evaluation['action_loops']==pair[1]
  assert {r['suite'] for r in evaluation['episodes']}==set(SUITES)
  assert all(Path(r['video']).is_file() and Path(r['video']).stat().st_size>0 for r in evaluation['episodes'])
  audits[label]=dict(native_seconds=t['measured_mean_update_seconds'],data_matches_original_full_run=True,fresh_optimizer=True,vae_anchor=True,finite_gradients=True,smoke_episodes=8)
 release=dict(source_revision=status['source_revision'],approved_pairs=pairs,audited_unix=time.time(),audits=audits,evaluation_workers_per_gpu=8)
 write(out/'release.json',release)
 print(job,'RELEASED',json.dumps(release))
