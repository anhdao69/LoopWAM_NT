"""Read and revalidate pre-production evidence; writes only audit JSON, never releases training."""
import hashlib,json,math,re,shutil,subprocess,time
from unittest.mock import patch
import run_kv_campaign as runner
from pathlib import Path
from run_kv_campaign import ROOT,QUEUES,read,verify_train,verify_eval,validate_release_inputs,select_candidate
from run_full_libero_v0_job import source_hashes
pin='3a89bfac5cfccad66bfebbe00014f7f739a78adc'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==pin
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
 return h.hexdigest()

def verify_native(output,checkpoint,mode,video,action):
 result=read(output);proof=read(output.with_suffix('.provenance.json'))
 assert result['passed'] and result['mode']==mode and result['repeat_bit_identical']
 assert proof['source_revision']==pin and proof['test_script_sha256']==sha('scripts/verify_kv_native_inference.py')
 assert proof['checkpoint']==str(checkpoint.resolve()) and proof['checkpoint_sha256']==sha(checkpoint)
 assert proof['version']=='v0' and proof['video_loops']==video and proof['action_loops']==action and proof['action_kv_mode']==mode
 assert proof['command'][1:]==['scripts/verify_kv_native_inference.py','--checkpoint',str(checkpoint.resolve()),'--output',str(output.resolve())]
 assert proof['returncode']==0 and proof['result_sha256']==sha(output)
 return proof

def verify_train_readonly(path,spec):
 def compare_fairness(destination,payload):
  assert read(destination)==payload, 'Existing fairness evidence differs'
 with patch.object(runner,'write',side_effect=compare_fairness):return verify_train(path,spec,True)

evidence=Path('plans/evidence/kv_concat_20261008')
regressions={}
for name in ['cpu_3a89bfa','gpu_3a89bfa_4770','gpu_3a89bfa_4771']:
 assert (evidence/(name+'_passed')).is_file(),name
 p=evidence/(name+'.log');text=p.read_text();counts=re.findall(r'(\d+) passed',text)
 assert counts and int(counts[-1])==(385 if name.startswith('cpu') else 394)
 regressions[name]=dict(path=str(p.resolve()),sha256=sha(p),passed=int(counts[-1]))
reproduction=read(ROOT/'kv_evaluator_diagnosis_20261008/reproduction_passed.json')
assert reproduction['passed'] and reproduction['all100_action_traces_bit_identical'] and reproduction['all100_outcomes_steps_replans_identical']
controls={'baseline':ROOT/'kv_campaign_job4771/control_aligned_v4a1_accepted/summary.json','repeat':ROOT/'kv_campaign_job4770/control_repeat_v4a4_grouped/summary.json'}
results={}
for queue,job,peer in [('concat',4770,4771),('mix',4771,4770)]:
 root=ROOT/f'kv_campaign_job{job}/campaign_3a89bfa'
 prepared=read(root/'prepared.json')
 assert prepared['source_revision']==pin and prepared['source_hashes']==source_hashes() and prepared['queue']==queue
 assert set(prepared['configs'])=={s['label'] for s in QUEUES[queue]}
 assert (root/'native_compile_passed').is_file()
 verify_native(root/'native_compiled.json',root/f'{queue}_long_smoke_warm/latest.pt',queue,4,1)
 release=dict(queue=queue,source_revision=pin,evaluation=reproduction['selected'],native_compiled_verified=True)
 for label,path in controls.items():
  release[f'control_{label}_summary']=str(path);release[f'control_{label}_sha256']=sha(path)
 validate_release_inputs(release,prepared)
 stages={}
 for spec in QUEUES[queue]:
  label=spec['label'];config=prepared['configs'][label];bench=read(root/f'{label}_benchmarks.json')
  assert {r['backend'] for r in bench['trials']}=={'ddp','zero1','zero2'}
  chosen=select_candidate(bench['trials'],spec['mode']!='aligned')
  assert all(chosen[k]==config[k] for k in ['backend','microbatch'])
  if spec['mode']!='aligned':assert config['backend']=='ddp' and config['microbatch']==8
  for phase in (['cold','warm'] if spec['mode']!='aligned' else ['warm']):
   train=root/f'{label}_smoke_{phase}';timing=verify_train_readonly(train,spec);m=read(train/'manifest.json')
   assert m['backend']==config['backend'] and m['microbatch']==config['microbatch']
   assert read(train/'fairness.json')['passed']
  summary=verify_eval(root/f'{label}_smoke_eval',spec,42,True)
  em=read(root/f'{label}_smoke_eval/manifest.json')
  assert Path(em['checkpoint']).resolve()==(train/'latest.pt').resolve()
  assert em['workers_per_gpu']==config['workers_per_gpu'] and em['arguments']['render_threads']==config['render_threads']
  assert em['arguments']['torch_threads']==1 and em['arguments']['video_threads']==1
  assert summary['checkpoint_sha256']==sha(train/'latest.pt')
  if spec['mode']!='aligned':
   diagnostics=read(root/f'{label}_diagnostics.json')
   assert diagnostics['mode']==spec['mode'] and diagnostics['rng_and_parameters_unchanged']
  stages[label]=dict(configuration=config,training_manifest_sha256=sha(train/'manifest.json'),checkpoint_sha256=summary['checkpoint_sha256'],smoke_episodes=summary['total_episodes'])
 if queue=='concat':
  for label in ['aligned41','original44']:
   checkpoint=ROOT/('v0_v4a1_job4728_20261007/train/latest.pt' if label=='aligned41' else 'v0_scratch_fast_bs128_20261005/latest.pt')
   expected='32e143c0467faea2717195e2825eed91874c67f1efd54dada2a5be2adafece90' if label=='aligned41' else 'd75adef4068d74827921ea35a6bd8d36b5e6ad8b0b84de8dc7ad9e9766486125'
   proof=verify_native(root/f'native_compiled_{label}.json',checkpoint,'aligned',4,1 if label=='aligned41' else 4)
   assert proof['checkpoint_sha256']==expected
   for flavor in ['eager','compiled']:
    p=root.parent/f'latency_{label}_{flavor}.json';d=read(p)['models']['loopwam_v0']
    assert d['checkpoint_sha256']==expected and d['compiled']==(flavor=='compiled')
    assert len(d['samples_ms'])==100 and len(d['trials'])==2 and math.isfinite(d['mean_ms'])
 results[queue]=dict(job=job,root=str(root),release=release,stages=stages,training_hours=sum(c['training_hours'] for c in prepared['configs'].values()))
free=shutil.disk_usage(ROOT).free
assert free>150*1024**3, 'Insufficient free shared storage for campaign'
report=dict(passed=True,source_revision=pin,audited_unix=time.time(),regressions=regressions,reproduction=reproduction,queues=results,free_disk_gib=free/1024**3)
p=ROOT/'kv_campaign_audit_3a89bfa.json'
with p.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
