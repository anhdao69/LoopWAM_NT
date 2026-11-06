#!/usr/bin/env bash
set -euo pipefail
source /data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/environment.sh
export LOOPWAM_CUDA_GRAPH=1
export LOOPWAM_RUNTIME_MANIFEST=/data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/runtime_manifest.json
cd /storage/anhdh35/LoopWAM-robotwin-eval
while [ ! -f /data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/download_receipt.json ]; do sleep 10; done
python /data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/scripts/cpu_validate.py > /data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/cpu_validation.log 2>&1
python /data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/scripts/graph_gate.py > /data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/graph_gate.log 2>&1
python -c "import json; d=json.load(open('/data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/graph_gate.json')); assert d['graph_passed'] and d['max_abs_difference']==0"
python scripts/evaluate_loopwam_robotwin.py --checkpoint /data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/checkpoint/policy.pt --output-dir /data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/smoke --robotwin-root /storage/anhdh35/LoopWAM-robotwin-eval/third_party/RoboTwin --data-dir /data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/checkpoint --gpus 0 --per-gpu 4 --episodes 1 --seed 42 --expected-step 284772 --tasks adjust_bottle,click_bell --max-attempts 2 --no-progress-timeout 600 > /data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/smoke.log 2>&1
python - <<'INNER'
from pathlib import Path
import json,shutil,time
b=Path('/data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031');r=Path('/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin_v0_epoch06')
for name in ['download_receipt.json','cpu_validation.json','graph_gate.json']:
 shutil.copy2(b/name,r/name)
s=json.loads((b/'smoke/COMPLETE.json').read_text());assert len(s['finished'])==4
record=dict(passed=True,smoke_episodes=4,smoke_successes=sum(x['successes'] for x in s['finished'].values()),smoke_excluded=True,tests_passed=18,epoch=6,expected_step=284772,seed=42,nfe=10,workers=4,planned_episodes=10000,verified_unix=time.time())
(b/'preflight.json').write_text(json.dumps(record,indent=2));shutil.copy2(b/'preflight.json',r/'preflight.json');shutil.copy2(b/'smoke/COMPLETE.json',r/'smoke_complete.json')
p=r/'setup_report.md';p.write_text(p.read_text().replace('Status: downloading and validating before launch. The live report records the actual run status.','Preflight passed: checkpoint verified, strict CPU load passed, native/CUDA-graph gate passed, and four simulator smoke episodes completed. The live report records the actual benchmark status.'))
INNER
exec bash /data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/launch.sh >> /data14t/anhdh35/loopwam_robotwin_v0_epoch06_20261031/controller.log 2>&1
