import json
from pathlib import Path
def publish_runtime(path,runtime):
 path=Path(path)
 if path.exists():
  old=json.loads(path.read_text())
  comparable=lambda d:{k:v for k,v in d.items() if k!='created_unix'}
  if comparable(old)!=comparable(runtime):raise ValueError('Existing runtime manifest differs; refusing to change a bound run')
  return
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(runtime,indent=2));tmp.replace(path)
def reuse_completed(bulk,validate):
 bulk=Path(bulk);path=bulk/'results/COMPLETE.json'
 if not path.exists():return False
 c=json.loads(path.read_text());p=json.loads((bulk/'results/manifest.json').read_text())
 if c.get('failed') or c.get('running') or set(c.get('success_rates',{}))!={'demo_clean','demo_randomized'}:
  raise ValueError('Complete evidence lacks required successful-finalization metadata')
 validate(c,p['checkpoint_sha256'],p['tasks'],p['episodes'])
 launch=bulk/'launch.json';l=json.loads(launch.read_text())
 if l['status']!='complete':
  l.update(status='complete',finished_unix=path.stat().st_mtime,recovered_from_complete=True)
  tmp=launch.with_suffix('.tmp');tmp.write_text(json.dumps(l,indent=2));tmp.replace(launch)
 return True
