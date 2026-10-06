"""Run from the pinned server source checkout; CPU-only verification of completed runs."""
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch

sys.path.insert(0,str(Path.cwd()/'scripts'))
from evaluate_loopwam_libero import validate_checkpoint, sha256_file
from run_dense_v2_job import verify_evaluation

ROOT=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt')
RUNS={
 'v0':('v0_scratch_fast_bs128_20261005','v0_auto_eval_job4689/inference'),
 'v1':('v1_bs128_job4691/train','v1_bs128_job4691/inference'),
 'dense_s12':('dense_v2_job4659_20261006/dense_s12_train','dense_v2_job4659_20261006/dense_s12_evaluation'),
 'v2':('dense_v2_job4659_20261006/v2_train','dense_v2_job4659_20261006/v2_evaluation')}
result={'audited_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':{}}
read=lambda p:json.loads(p.read_text())
for version,(tr,ev) in RUNS.items():
 train,evaluation=ROOT/tr,ROOT/ev
 m,t,s=read(train/'manifest.json'),read(train/'timing.json'),read(evaluation/'summary.json')
 assert m['version']==version and m['resume'] is None and m['epochs']==10 and m['global_batch']==128
 assert t['status']=='complete' and t['completed_updates']==7250 and t['windows_seen']==926780
 checkpoint=train/'latest.pt'
 payload=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
 data=read(train/'data/data_manifest.json')
 contract=validate_checkpoint(payload,data,sha256_file(train/'data/dataset_stats.json'))
 print(version,'hashing final checkpoint',flush=True)
 digest=sha256_file(checkpoint)
 assert digest==s['checkpoint_sha256']
 verify_evaluation(evaluation,version,checkpoint)
 rows=s['episodes'];assert sum(e['success'] for e in rows)==s['successes']
 assert s['success_rate']==s['successes']/100
 for task in range(10):
  rr=[e for e in rows if e['task_id']==task]
  assert len(rr)==10 and sum(e['success'] for e in rr)==s['per_task'][str(task)]['successes']
  assert {e['initial_state_index'] for e in rr}==set(range(10))
 for e in rows:
  assert e['seed']==42+e['task_id']*100000+e['episode_index']*1000
  assert read(evaluation/'episodes'/f"task{e['task_id']}_episode{e['episode_index']}.json")==e
 ranks=[]
 for p in sorted(evaluation.glob('rank*.json')):ranks+=read(p)['episodes']
 assert sorted(ranks,key=lambda e:(e['task_id'],e['episode_index']))==sorted(rows,key=lambda e:(e['task_id'],e['episode_index']))
 def video_info(e):
  p=Path(e['video']);v={'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256_file(p)}
  if shutil.which('ffprobe'):
   probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,nb_frames,width,height,duration','-of','json',str(p)],text=True))
   stream=probe['streams'][0];assert int(stream['nb_frames'])==e['steps']+e['wait_steps']+1
   v['probe']=stream
  return v
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:videos=list(pool.map(video_info,rows))
 result['runs'][version]=dict(training_dir=tr,evaluation_dir=ev,checkpoint_bytes=checkpoint.stat().st_size,
   checkpoint_sha256=digest,checkpoint_step=payload['step'],checkpoint_contract=contract,
   architecture_valid=True,complete_training=True,evaluation_coverage_valid=True,
   episodes_and_rank_files_match=True,video_count=len(videos),video_bytes=sum(v['bytes'] for v in videos),
   ffprobe_available=bool(shutil.which('ffprobe')),videos=videos,
   training_timing_mtime_utc=datetime.datetime.fromtimestamp((train/'timing.json').stat().st_mtime,datetime.timezone.utc).isoformat(),
   evaluation_manifest_mtime_utc=datetime.datetime.fromtimestamp((evaluation/'manifest.json').stat().st_mtime,datetime.timezone.utc).isoformat(),
   evaluation_summary_mtime_utc=datetime.datetime.fromtimestamp((evaluation/'summary.json').stat().st_mtime,datetime.timezone.utc).isoformat())
 del payload
 print(version,'verified',s['successes'],'/100; videos',len(videos),flush=True)
accounting=subprocess.check_output(['sacct','-j','4689,4691,4659.17','--format=JobID,JobName,State,ExitCode,Start,End,Elapsed,AllocTRES','-P'],text=True)
keep={'4689','4689.0','4689.61','4689.66','4691','4691.batch','4659.17'}
result['slurm_accounting']='\n'.join(line for i,line in enumerate(accounting.splitlines()) if i==0 or line.split('|')[0] in keep)
result['squeue']=subprocess.check_output(['squeue','-u','anhdh35','-o','%.12i %.30j %.10T %.12M %.22R'],text=True)
(ROOT/'final_results_audit_20261006.json').write_text(json.dumps(result,indent=2))
print('ALL FOUR RUNS VERIFIED',flush=True)
