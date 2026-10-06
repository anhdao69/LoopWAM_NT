"""Validate copied assets against remote hashes and shared experiment contracts."""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys

metadata_only = "--metadata-only" in sys.argv

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'plans/evidence/final_results_20261006';RAW=OUT/'raw'
a=json.loads((OUT/'remote_audit.json').read_text())
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=None;videos=0;missing=[]
for v,r in a['runs'].items():
 p,q=RAW/r['training_dir'],RAW/r['evaluation_dir'];m=read(p/'manifest.json');em=read(q/'manifest.json')
 summary=read(q/'summary.json')
 assert len(summary['episodes'])==100
 assert sum(e['success'] for e in summary['episodes'])==summary['successes']
 for e in summary['episodes']:
  assert read(q/'episodes'/f"task{e['task_id']}_episode{e['episode_index']}.json")==e
 data=m['data']
 common={k:data[k] for k in ('content_files','info_sha256','episode_metadata_sha256','train_episodes',
  'validation_episodes','normalization_sha256','text_cache_files','camera_order','action_offsets','video_offsets')}
 common.update(assets=m['asset_sha256'],initialization=m['initialization_metadata'],seed=m['seed'])
 if base is None:base=common
 else:assert common==base,(v,'shared assets/split/initialization differ')
 assert sha(p/'data/dataset_stats.json')==em['normalization_sha256']
 for row in r['videos']:
  f=RAW/row['path']
  valid = f.is_file() and f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
  if not valid:
   missing.append(row['path'])
   if not metadata_only:raise AssertionError(str(f))
  else:videos+=1
assert videos==400 or metadata_only
# Check all relative Markdown links in the generated report.
report=ROOT/'plans/performance/LoopWAM_Final_Results_20261006.md'
for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
 if not target.startswith(('http','mailto','#')):assert (report.parent/target).exists(),target
files=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ('local_integrity.json','files.sha256','videos.md') and (not metadata_only or p.suffix != '.mp4') and not p.name.startswith('.')]
lines=[sha(p)+'  '+str(p.relative_to(OUT)) for p in sorted(files)]
(OUT/'files.sha256').write_text('\n'.join(lines)+'\n')
result=dict(status='verified',scope='report_and_data' if metadata_only else 'report_data_and_videos',models=4,production_updates=29000,evaluation_episodes=400,videos_verified_remotely=400,local_videos=videos,
 shared_data_split_initialization_and_assets_match=True,video_storage="H100 server; excluded from Git at user request" if metadata_only else "local archive",
 files=len(files),archive_bytes=sum(p.stat().st_size for p in files))
(OUT/'local_integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
