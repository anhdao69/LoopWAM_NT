from pathlib import Path
import hashlib,json,sys,time,subprocess
r=Path('/mnt/data/vmo-ai-task/anhdh35/robotwin_eval_runtime_20261030');c=Path('/mnt/data/vmo-ai-task/anhdh35/LoopWAM_robotwin_queue')
sys.path.insert(0,str(c/'src'))
from fastwam.robotwin_pipeline import validate_eval_summary,fingerprint_files
for name,expected in [('cpu_tests_final.log','337 passed, 24 skipped'),('gpu_tests_final.log','344 passed, 17 skipped')]:
 assert expected in (r/name).read_text(),name
smokes={}
for name in ['simulator_smoke_v4','simulator_smoke_v6_pci']:
 result=json.loads((r/name/'COMPLETE.json').read_text());validate_eval_summary(result,result['checkpoint_sha256'],['adjust_bottle'],1);smokes[name]=result
manifest=json.loads((r/'simulator_smoke_v6_pci/manifest.json').read_text())
assert set(manifest['gpu_pci'])=={'0','1'} and manifest['renderer_shader']=='default'
files=[Path(p) for p in manifest['runtime_hashes']]+[r/'eval_env.sh',r/'train_env.sh',r/'assets_ready.json',r/'nvidia_icd.json',r/'nvidia_egl.json',r/'mesa_layer/layers/VkLayer_MESA_device_select.json',r/'mesa_layer/root/usr/lib/x86_64-linux-gnu/libVkLayer_MESA_device_select.so',r/'RoboTwin/script/eval_policy.py',r/'RoboTwin/script/test_render.py']
files+=list((r/'overlay/sapien.libs').glob('*.so'))
record=dict(passed=True,simulator_clean_and_randomized=True,multi_gpu_smoke=True,created=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c,text=True).strip(),
 tests={'cpu':{'passed':337,'skipped':24},'gpu':{'passed':344,'skipped':17}},smokes=smokes,
 runtime_hashes=fingerprint_files(files),renderer='SAPIEN3.0.0b1 default raster on NVIDIA570.158.01; Mesa device-select layer binds each worker PCI GPU',
 production_concurrency={'gpus':8,'workers_per_gpu':2,'shared_t5':1,'two_workers_per_gpu_measured':False},
 continuation_smoke='Four native updates with a checkpoint/resume boundary, queued after completed epoch5 evaluation; not yet executed')
p=r/'preflight_passed.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(p)
print(json.dumps(dict(passed=True,path=str(p),bound_files=len(record['runtime_hashes']))))
