"""Validate release bytes, original runtime and benchmark compatibility before launch."""
import concurrent.futures,hashlib,json,time
from pathlib import Path
from queue_safety import publish_runtime
B=Path(__file__).resolve().parent.parent
S=json.loads((B/'run_settings.json').read_text());W=Path(S['repo']);R=Path(S['report'])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 m=json.loads((B/'hf_manifest.json').read_text())
 for item in m['files']:
  p=B/'checkpoint'/Path(item['path']).name
  assert p.stat().st_size==item['size'],p
  if 'lfs' in item:assert sha(p)==item['lfs']['oid'],p
  else:assert hashlib.sha1(f"blob {item['size']}\0".encode()+p.read_bytes()).hexdigest()==item['oid'],p
 manifest=json.loads((B/'checkpoint/training_manifest.json').read_text())
 for path,expected in manifest['code_files_sha256'].items():assert sha(W/path)==expected,path
 old=json.loads(Path(S['inherited_runtime']).read_text())
 def verify(item):
  p,info=item;assert sha(p)==info['sha256'],f'Inherited runtime drift: {p}'
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(verify,old['files'].items()))
 baseline=json.loads((B/'reference/baseline_protocol.json').read_text());plan=json.loads((B/'planned_protocol.json').read_text())
 keys=['tasks','settings','episodes','seed','instruction_type','num_inference_steps','action_horizon','replan_steps','noise_seed','image_preprocessing','instruction_rng','cuda_graph','compile_action_infer','renderer_shader','vae_sha256']
 assert all(plan[k]==baseline[k] for k in keys)
 assert sha(W/'third_party/RoboTwin/script/eval_policy.py')==baseline['harness_sha256']
 assert sha(W/'experiments/robotwin/loopwam_policy/deploy_policy.py')==baseline['policy_sha256']
 assert json.loads((B/'checkpoint/dataset_stats.json').read_text())==json.loads((B/'reference/baseline_stats.json').read_text())
 previous=json.loads((B/'reference/baseline_config.json').read_text())['training_contract']
 differences={k:dict(current=v,previous=previous.get(k)) for k,v in manifest['contract'].items() if v!=previous.get(k)}
 evidence=dict(passed=True,checked_source_files=manifest['code_files_sha256'],normalization_json_equal=True,normalization_raw_hash_changed=plan['stats_sha256']!=baseline['stats_sha256'],training_contract_differences=differences,checked_protocol_fields=keys,inherited_runtime_files=len(old['files']),verified_unix=time.time())
 (R/'source_check.json').write_text(json.dumps(evidence,indent=2))
 files=dict(old['files'])
 paths=list((B/'scripts').glob('*.py'))+list(B.glob('*.py'))+list(B.glob('*.sh'))+list((B/'reference').glob('*.json'))+list((B/'checkpoint').glob('*.json'))+[B/'run_settings.json',B/'planned_protocol.json',B/'calibration.json',B/'hf_manifest.json',B/'download_receipt.json']
 for p in paths:files[str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size)
 runtime=dict(created_unix=time.time(),files=files,hf_revision=m['revision'],variant=S['variant'],host=S['host'],inherited_runtime=S['inherited_runtime'],renderer='rt32/depth8/OIDN2.3.3',normalization_json_equal=True)
 publish_runtime(B/'runtime_manifest.json',runtime)
 print(json.dumps(evidence,indent=2),flush=True)
if __name__=='__main__':main()
