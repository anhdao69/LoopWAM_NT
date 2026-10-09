"""Release only an audited eager campaign, then publish both launch scripts last."""
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt')
pin='3a89bfac5cfccad66bfebbe00014f7f739a78adc';source=root/'source_kv_3a89bfa'
audit_path=root/'kv_campaign_audit_3a89bfa.json';raw=audit_path.read_bytes();audit=json.loads(raw)
assert audit['passed'] and audit['scope']=='eager_production_release' and audit['compiled_inference_disabled']
assert audit['source_revision']==pin and subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==pin
steps=subprocess.check_output(['squeue','--steps','-h','-j','4770,4771','-o','%i'],text=True).split()
assert set(steps)=={'4770.batch','4771.batch'},steps
for queue,record in audit['queues'].items():
 out=Path(record['root']);assert not (out/'release.json').exists() and not (out.parent/'launch.sh').exists()
 for label in record['stages']:assert not (out/label).exists()
for queue,record in audit['queues'].items():
 out=Path(record['root']);release=dict(record['release'],audit_file=str(audit_path),audit_sha256=hashlib.sha256(raw).hexdigest(),released_unix=time.time())
 assert release['native_compiled_verified'] is False
 with (out/'release.json').open('x') as f:json.dump(release,f,indent=2)
 peer=Path(audit['queues']['mix' if queue=='concat' else 'concat']['root'])
 script=f'''#!/usr/bin/env bash
set -euo pipefail
cd '{source}'
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts:${{PYTHONPATH:-}}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LP_NUM_THREADS=3
exec python -u scripts/run_kv_campaign.py --queue {queue} --output-root '{out}' --peer-root '{peer}' --phase run > '{out.parent}/production_3a89bfa.log' 2>&1
'''
 with (out.parent/'launch.sh.ready').open('x') as f:f.write(script)
for record in audit['queues'].values():
 parent=Path(record['root']).parent
 os.rename(parent/'launch.sh.ready',parent/'launch.sh')
print(json.dumps(dict(released=True,unix=time.time(),source_revision=pin,audit_sha256=hashlib.sha256(raw).hexdigest(),jobs=[4770,4771])))
