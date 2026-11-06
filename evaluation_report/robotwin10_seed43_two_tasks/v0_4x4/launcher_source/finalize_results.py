"""Wait for successful evaluation, audit scored artifacts, and publish final tables."""
import concurrent.futures,csv,hashlib,importlib.util,json,shutil,time,datetime
from pathlib import Path
B=Path('/data14t/anhdh35/loopwam_robotwin10_seed43_two_tasks_20261103/v0_4x4')
W=Path('/storage/anhdh35/LoopWAM-robotwin10-dense-eval')
R=Path('/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin10_seed43_two_tasks/v0_4x4')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 while True:
  l=json.loads((B/'launch.json').read_text())
  if l['status']=='complete':break
  if l['status']=='error':raise RuntimeError('Evaluation stopped with an error; final scored report is withheld')
  time.sleep(30)
 c=json.loads((B/'results/COMPLETE.json').read_text());p=json.loads((B/'results/manifest.json').read_text())
 spec=importlib.util.spec_from_file_location('frozen_eval',W/'scripts/evaluate_loopwam_robotwin.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 mod.validate_eval_summary(c,p['checkpoint_sha256'],p['tasks'],100)
 assert p['seed']==43 and p['tasks']==['open_microwave','move_stapler_pad'] and len(c['finished'])==4 and not c['failed'] and not c['running']
 assert sha(p['checkpoint'])==p['checkpoint_sha256']
 for path,expected in p['runtime_hashes'].items():assert sha(path)==expected,path
 runtime=json.loads((B/'runtime_manifest.json').read_text())
 for path,info in runtime['files'].items():assert sha(path)==info['sha256'],path
 def verify(item):
  key,cell=item;records=mod.completed_records(Path(cell['directory']),cell['config'],100)
  assert records==cell['episodes'],key
  assert cell['successes']==sum(e['success'] for e in records)
  assert json.loads((Path(cell['directory']).parent/'result.json').read_text())==cell
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(verify,c['finished'].items()))
 rows=[];episodes=[];totals={}
 for task in p['tasks']:
  for setting in p['settings']:
   cell=c['finished'][f'{task}/{setting}'];rows.append(dict(task=task,setting=setting,successes=cell['successes'],episodes=100,sr_percent=cell['successes'],accepted_attempt=Path(cell['directory']).name,seconds=cell['seconds']))
   for i,e in enumerate(cell['episodes']):episodes.append(dict(task=task,setting=setting,episode_index=i,**e))
 for setting in p['settings']:
  successes=sum(x['successes'] for x in rows if x['setting']==setting);totals[setting]=dict(successes=successes,episodes=200,sr_percent=successes/2)
  assert successes/200==c['success_rates'][setting]
 totals['combined']=dict(successes=sum(x['successes'] for x in rows),episodes=400,sr_percent=sum(x['successes'] for x in rows)/4)
 assert len(episodes)==400
 for name,data in [('task_results.csv',rows),('episodes.csv',episodes)]:
  with (R/name).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
 failures=[]
 for path in sorted((B/'results').glob('*/attempt_*/failure.json')):failures.append(dict(path=str(path),**json.loads(path.read_text())))
 audit=dict(status='passed',audited_unix=time.time(),verified_pairs=4,verified_videos=400,verified_source_files=len(p['runtime_hashes']),verified_runtime_files=len(runtime['files']),checkpoint_sha256=p['checkpoint_sha256'],complete_sha256=sha(B/'results/COMPLETE.json'),failed_attempts=len(failures),excluded_partial_episodes=sum(x['episodes'] for x in failures))
 (R/'final_audit.json').write_text(json.dumps(audit,indent=2));(R/'final_results.json').write_text(json.dumps(dict(totals=totals,tasks=rows),indent=2));(R/'failed_attempts.json').write_text(json.dumps(failures,indent=2));shutil.copy2(B/'results/COMPLETE.json',R/'COMPLETE.json')
 end=datetime.datetime.fromtimestamp(l['finished_unix'],datetime.timezone(datetime.timedelta(hours=7))).isoformat()
 lines=['# v0-video4-action4 — two-task seed-43 evaluation','',f'Completed {end}. All 4 task/setting pairs and 400 scored episodes passed artifact verification.','', '| Setting | Successes / episodes | SR |','| --- | ---: | ---: |']
 for setting in ['demo_clean','demo_randomized','combined']:
  t=totals[setting];lines.append(f"| {setting} | {t['successes']}/{t['episodes']} | {t['sr_percent']:.2f}% |")
 lines+=['','| Task | Clean successes / 100 | Randomized successes / 100 | Combined SR |','| --- | ---: | ---: | ---: |']
 for task in p['tasks']:
  clean=c['finished'][f'{task}/demo_clean']['successes'];random=c['finished'][f'{task}/demo_randomized']['successes'];lines.append(f'| {task} | {clean} | {random} | {(clean+random)/2:.2f}% |')
 lines+=['',f"Controller elapsed time: {(l['finished_unix']-l['started_unix'])/3600:.2f} hours, excluding download and preflight. This is a controller-segment duration; any later manual resumes must be accounted for separately.",f"Interrupted attempts: {len(failures)}; logged partial episodes excluded from scoring: {audit['excluded_partial_episodes']}.",'','Protocol: seed 43; clean/randomized; 100 accepted episodes per task/setting; NFE10; predict32/execute24; unseen instructions; native v0 (12 stored block pairs, video/action core loops 4/4). All unscored calibration episodes are excluded. The reused exact-output graph gate included seed 43. Actual scene seeds and instructions are in episodes.csv; unseeded Python instruction-template generation prevents assuming exact episode pairing across independent runs.','', 'Both compared RoboTwin-10 releases were trained on the same ten tasks for five epochs. This report covers only the two requested tasks. See [setup](setup_report.md), [source checks](source_check.json), [protocol](planned_protocol.json), [final audit](final_audit.json), [task CSV](task_results.csv), and [episode CSV](episodes.csv).','']
 (R/'final_report.md').write_text('\n'.join(lines))
 (R/'finalizer_status.json').write_text(json.dumps(dict(status='complete',finished_unix=time.time(),totals=totals),indent=2))
 print(json.dumps(totals,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  (R/'finalizer_status.json').write_text(json.dumps(dict(status='error',error=repr(e),updated_unix=time.time()),indent=2));raise
