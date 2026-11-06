"""Live readable report for the local FastWAM-protocol RoboTwin evaluation."""
import json,re,time
from pathlib import Path
B=Path('/media/newdisk/loopwam_robotwin10_v0_4x1_20261105');R=Path('/media/newdisk/loopwam_robotwin10_v0_4x4_20261104/repo/evaluation_report/robotwin10_v0_4x1')
RX=re.compile(r'Success rate: (?:\x1b\[\d+m)?(\d+)/(\d+).*?current seed: (?:\x1b\[\d+m)?(\d+)')

def update():
 R.mkdir(parents=True,exist_ok=True);p=B/'results/summary.json';s=json.loads(p.read_text()) if p.exists() else dict(finished={},running={})
 rows=[]
 for task in json.loads((B/'checkpoint/evaluation_protocol.json').read_text())['tasks']:
  for setting in ['demo_clean','demo_randomized']:
   key=f'{task}/{setting}';cell=s.get('finished',{}).get(key)
   if cell:row=dict(task=task,setting=setting,status='complete',episodes=len(cell['episodes']),successes=cell['successes'])
   elif key in s.get('running',{}):
    logs=sorted((B/'results'/f'{task}__{setting}').glob('attempt_*/eval.log'));records=RX.findall(logs[-1].read_text(errors='replace')) if logs else []
    row=dict(task=task,setting=setting,status='running',episodes=int(records[-1][1]) if records else 0,successes=int(records[-1][0]) if records else 0)
   else:row=dict(task=task,setting=setting,status='failed' if key in s.get('failed',[]) else 'pending',episodes=0,successes=0)
   rows.append(row)
 progress=dict(updated_unix=time.time(),episodes=sum(x['episodes'] for x in rows),successes=sum(x['successes'] for x in rows),planned=2000,
  completed_cells=sum(x['status']=='complete' for x in rows),running_cells=sum(x['status']=='running' for x in rows),failed=s.get('failed',[]),finished=(B/'results/COMPLETE.json').exists(),cells=rows)
 if (B/'launch.json').exists():
  progress['launch']=json.loads((B/'launch.json').read_text())
  (R/'launch.json').write_text(json.dumps(progress['launch'],indent=2))
 tmp=R/'progress.json.tmp';tmp.write_text(json.dumps(progress,indent=2));tmp.replace(R/'progress.json')
 lines=['# LoopWAM v0 4/1 RoboTwin-10 — server 10.148.183.189','',f"Scored episodes: **{progress['episodes']}/2,000**; successes: **{progress['successes']}**; completed task/settings: **{progress['completed_cells']}/20**.",'',
  'Seed42;10tasks;clean+randomized;100acceptedepisodes/task/setting;unseen instructions;NFE10;predict32/execute24;v0-video4-action1 checkpoint at epoch5. Smoke/calibration excluded. Partial SR is provisional.','',
  '| Task | Setting | Status | Episodes | Successes | SR |','| --- | --- | --- | --- | --- | --- |']
 for x in rows:
  rate=f"{100*x['successes']/x['episodes']:.2f}%" if x['episodes'] else '—'
  lines.append(f"| {x['task']} | {x['setting']} | {x['status']} | {x['episodes']}/100 | {x['successes']} | {rate} |")
 lines+=['','[Detailed setup and comparability](setup_report.md) · [Progress JSON](progress.json)',
 'Rendering uses RT32/depth8/OIDN2.3.3 (Blackwell compatibility upgrade). Input preprocessing and instruction generation follow released FastWAM; the same settings were used for the preceding local v0 evaluations.']
 tmp=R/'report.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(R/'report.md')
 return progress

if __name__=='__main__':print(json.dumps({k:v for k,v in update().items() if k!='cells'},indent=2))
