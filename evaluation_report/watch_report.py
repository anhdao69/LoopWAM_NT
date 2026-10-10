"""Persistent CPU-only post-campaign finalizer. No GPU work or evaluation mutation."""
import datetime,json,os,pathlib,shutil,time,traceback
from build_report import ROOT,build,completion
MIRROR=pathlib.Path("/storage/anhdh35/LoopWAM_NT/evaluation_report")

def status(phase,**kwargs):
 p=ROOT/'report_status.json';temp=p.with_suffix('.tmp')
 temp.write_text(json.dumps({'status':phase,'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),**kwargs},indent=2));temp.replace(p)
 MIRROR.mkdir(parents=True,exist_ok=True)
 if MIRROR!=ROOT:
  target=MIRROR/p.name;staged=target.with_suffix(".tmp")
  shutil.copy2(p,staged);staged.replace(target)
if __name__=='__main__':
 try:
  while True:
   queues=completion()
   if any(x['state'].get('status') in ['failed','error'] for x in queues):
    status('upstream_failed',queues=queues,error='Evaluation failure; final scores/report withheld.');break
   if all(x['state'].get('status')=='complete' for x in queues):
    status('validating_complete_campaign',queues=queues)
    output=build(final=True)
    if MIRROR!=ROOT:shutil.copytree(ROOT,MIRROR,dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__","*.tmp"))
    output["copied_report"]=str(MIRROR/"final/report.html")
    status('complete',**output);print(json.dumps(output),flush=True);break
   status('waiting_for_all_evaluations',queues=[{'path':x['path'],'status':x['state'].get('status'),'stage':x['state'].get('stage')} for x in queues],expected_episodes=27000,final_report=str(ROOT/'final/report.html'),copied_report=str(MIRROR/'final/report.html'))
   time.sleep(30)
 except BaseException as e:
  status('failed',error=str(e),traceback=traceback.format_exc());raise
