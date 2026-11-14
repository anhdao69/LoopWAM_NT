"""Pin released epoch10 bytes and verify unchanged native evaluator/assets."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
B=Path(__file__).resolve().parent.parent
P=Path('/data14t/anhdh35/loopwam_robotwin10_dense_s12_20261105');W=Path('/storage/anhdh35/LoopWAM-robotwin10-dense-eval');R=B/'report'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin_baseline_sources(baseline,files):
 pinned={}
 for name,expected in baseline['runtime_hashes'].items():
  p=Path(name)
  if p.name=='runtime_manifest.json':continue
  if sha(p)!=expected:raise ValueError(f'Baseline evaluator/source drift: {p}')
  pinned[str(p)]=dict(sha256=expected,size=p.stat().st_size)
 if not pinned:raise ValueError('Baseline source fingerprint is empty')
 files.update(pinned)
 return len(pinned)

def main():
 assert (W/'third_party/RoboTwin/policy/loopwam_policy').resolve()==(W/'experiments/robotwin/loopwam_policy').resolve(),'Wrong deployed policy import target'
 for item in json.loads((B/'download_receipt.json').read_text())['files']:
  p=Path(item['path']);assert p.stat().st_size==item['size'] and sha(p)==item['sha256'],p
 assert sha(B/'checkpoint/policy.pt')=='a222704c10688bf42a531a4a4175fe45a1b172c437cf9b9efa435c4b4fae1ac8'
 c=json.loads((B/'checkpoint/config.json').read_text());assert (c['epoch'],c['optimizer_update'],c['variant'])==(10,90890,'dense-s12')
 checked={};differences={}
 for name,expected in c['training_code_sha256'].items():
  actual=sha(W/name);checked[name]=dict(expected=expected,actual=actual,equal=actual==expected)
  if name.startswith('src/'):assert actual==expected,name
  else:assert name=='scripts/train_loopwam_robotwin10.py';differences[name]='Continuation training script differs; not imported by native evaluation.'
 old=json.loads((P/'runtime_manifest.json').read_text())
 def check(item):
  p,info=item;assert sha(p)==info['sha256'],f'Inherited runtime drift: {p}'
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(check,old['files'].items()))
 baseline=json.loads((B/'reference/baseline_protocol.json').read_text());plan=json.loads((B/'planned_protocol.json').read_text())
 baseline_sources={};baseline_source_count=pin_baseline_sources(baseline,baseline_sources)
 keys=['tasks','settings','episodes','seed','instruction_type','num_inference_steps','action_horizon','replan_steps','noise_seed','image_preprocessing','instruction_rng','cuda_graph','compile_action_infer','renderer_shader','vae_sha256','stats_sha256']
 for k in keys:assert plan[k]==baseline[k],k
 assert sha(W/'third_party/RoboTwin/script/eval_policy.py')==baseline['harness_sha256']
 assert sha(W/'experiments/robotwin/loopwam_policy/deploy_policy.py')==baseline['policy_sha256']
 assert json.loads((B/'checkpoint/dataset_stats.json').read_text())==json.loads((B/'reference/baseline_stats.json').read_text())
 assert plan['tasks']==c['training_contract']['tasks']
 paths=list((B/'scripts').glob('*.py'))+[B/'finalize_results.py',B/'environment.sh',B/'start_queue.sh',B/'planned_protocol.json',B/'hf_manifest.json',B/'download_receipt.json']+list((B/'checkpoint').glob('*.json'))+list((B/'reference').glob('*.json'))+list((B/'gate_inputs').glob('*.pt'))
 files=dict(old['files']);files.update(baseline_sources)
 for p in paths:files[str(p)]=dict(sha256=sha(p),size=p.stat().st_size)
 target=B/'runtime_manifest.json';runtime=dict(files=files,checkpoint_sha256=sha(B/'checkpoint/policy.pt'),host='local RTX PRO 6000',inherited_runtime=str(P/'runtime_manifest.json'),created_unix=time.time())
 if target.exists():assert json.loads(target.read_text())['files']==files,'Bound runtime changed; refuse resume'
 else:target.write_text(json.dumps(runtime,indent=2))
 evidence=dict(passed=True,native_source_matches=8,baseline_sources_pinned=baseline_source_count,source_files=checked,training_script_difference=differences,inherited_verified_files=len(old['files']),matched_protocol_fields=keys,normalization_equal=True,verified_unix=time.time())
 (R/'source_check.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
