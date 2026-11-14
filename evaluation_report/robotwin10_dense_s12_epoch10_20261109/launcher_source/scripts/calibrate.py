"""Validate the proven four-worker path with16unscored episodes and memory headroom."""
import json,time,importlib.util,shutil
from pathlib import Path
TASKS=['adjust_bottle','grab_roller','press_stapler','scan_object']
def calibrate(B,W,python,run,status,gpu,write,checkpoint_sha):
 from fastwam.robotwin_pipeline import validate_eval_summary
 spec=importlib.util.spec_from_file_location('audit',B/'finalize_results.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
 def validate(profile):
  smoke=json.loads((B/profile['name']/'COMPLETE.json').read_text());validate_eval_summary(smoke,checkpoint_sha,TASKS,2)
  for cell in smoke['finished'].values():audit.assert_clean_render_log((Path(cell['directory'])/'eval.log').read_text(errors='replace'))
  return smoke
 if (B/'calibration.json').exists():
  c=json.loads((B/'calibration.json').read_text());assert c['checkpoint_sha256']==checkpoint_sha
  assert c['selected']['workers']==4 and len(c['profiles'])==1 and all(p['workers']==4 for p in c['profiles']),'Only four workers are approved for sustained evaluation'
  for p in c['profiles']:
   if p['passed']:validate(p)
  assert c['selected']['passed'] and c['selected']['memory_margin_mib']>=10000
  return c
 profiles=[]
 for workers in [4]:
  name=f'smoke_graph_w{workers}';profile_path=B/(name+'_profile.json')
  if profile_path.exists():
   profile=json.loads(profile_path.read_text())
   if profile['passed']:validate(profile);profiles.append(profile);continue
  # Never time the native evaluator's cached-cell resume as a fresh throughput trial.
  if (B/name).exists():
   archive=B/'calibration_abandoned'/f'{name}_{time.time_ns()}';archive.mkdir(parents=True)
   shutil.move(str(B/name),str(archive/name))
   if profile_path.exists():shutil.move(str(profile_path),str(archive/profile_path.name))
   write(archive/'archive_receipt.json',dict(original_directory=str(B/name),archived_directory=str(archive/name),reason='No valid passed timing receipt; preserve old trial and measure fresh episodes',archived_unix=time.time()))
  args=[python,str(W/'scripts/evaluate_loopwam_robotwin.py'),'--checkpoint',str(B/'checkpoint/policy.pt'),'--output-dir',str(B/name),'--robotwin-root',str(W/'third_party/RoboTwin'),'--data-dir',str(B/'checkpoint'),'--gpus','0','--per-gpu',str(workers),'--episodes','2','--seed','42','--expected-step','90890','--tasks',','.join(TASKS),'--max-attempts','1','--no-progress-timeout','600']
  profile=dict(name=name,workers=workers,backend='cuda-graph',passed=False,episodes=0,tasks=TASKS,episodes_per_cell=2)
  try:
   timing=run(name,args,reserve_mib=8000);profile.update(timing)
   smoke=validate(profile);profile.update(passed=True,episodes=16,successes=sum(c['successes'] for c in smoke['finished'].values()))
  except (RuntimeError,ValueError,AssertionError,FileNotFoundError) as e:
   profile['error']=str(e)
   if workers==4:write(profile_path,profile);raise
  write(profile_path,profile);profiles.append(profile);write(B/'calibration_progress.json',dict(profiles=profiles))
 good=[p for p in profiles if p['passed'] and p['memory_margin_mib']>=10000]
 if not good:raise RuntimeError('No passed profile retained10000MiB memory margin')
 selected=min(good,key=lambda p:p['seconds']/p['episodes'])
 c=dict(checkpoint_sha256=checkpoint_sha,profiles=profiles,selected=selected,finished_unix=time.time(),basis='Fixed four-worker configuration from prior successful sustained runs; sixteen unscored smoke episodes and at least10000MiB memory headroom. Higher concurrency is disabled after prior long-run resource failures. All smoke episodes excluded.')
 write(B/'calibration.json',c);return c
