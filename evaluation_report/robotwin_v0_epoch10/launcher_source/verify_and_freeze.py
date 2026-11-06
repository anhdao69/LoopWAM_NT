"""Verify inherited epoch-5 assets/source and bind this run's launcher files."""
import concurrent.futures,hashlib,json,subprocess,time
from pathlib import Path
B=Path('/data14t/anhdh35/loopwam_robotwin_v0_epoch10_20261106')
OLD=Path('/data14t/anhdh35/loopwam_robotwin_v0_20261030')
ROOT=Path('/storage/anhdh35/LoopWAM-robotwin-eval')
R=Path('/storage/anhdh35/LoopWAM_NT/evaluation_report/robotwin_v0_epoch10')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    old=json.loads((OLD/'runtime_manifest.json').read_text())
    def check(item):
        p,info=item
        assert sha(p)==info['sha256'],f'Epoch5 runtime changed: {p}'
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(check,old['files'].items()))
    baseline=json.loads((OLD/'results/manifest.json').read_text())
    for p,h in baseline['runtime_hashes'].items():
        assert sha(p)==h,f'Epoch5 frozen source changed: {p}'
    revision=json.loads((B/'checkpoint/config.json').read_text())['loader_source_revision']
    diff=subprocess.check_output(['git','diff',revision,'HEAD','--','src/fastwam/models/wan22/loopwam_robot.py','src/fastwam/models/wan22/loopwam.py'],cwd=ROOT,text=True)
    assert not diff,'Model/loader differs from release source'
    protocol=json.loads((B/'checkpoint/evaluation_protocol.json').read_text())
    keys=['tasks','settings','episodes','seed','instruction_type','num_inference_steps','action_horizon','replan_steps','noise_seed','image_preprocessing','instruction_rng','cuda_graph','compile_action_infer','renderer_shader','stats_sha256','vae_sha256']
    comparison={k:dict(epoch5=baseline[k],epoch10=protocol[k],equal=baseline[k]==protocol[k]) for k in keys}
    assert all(x['equal'] for x in comparison.values())
    assert sha(B/'checkpoint/dataset_stats.json')==baseline['stats_sha256']
    old.update(created_unix=time.time(),epoch=10,hf_revision=json.loads((B/'hf_manifest.json').read_text())['revision'],shared_epoch5_runtime_manifest=str(OLD/'runtime_manifest.json'))
    paths=list((B/'scripts').glob('*.py'))+list(B.glob('*.sh'))+list((B/'checkpoint').glob('*.json'))+[B/'download_receipt.json',B/'hf_manifest.json',B/'calibration.json']
    for p in paths:old['files'][str(p)]=dict(sha256=sha(p),bytes=p.stat().st_size)
    (B/'runtime_manifest.json').write_text(json.dumps(old,indent=2))
    (R/'epoch5_protocol_comparison.json').write_text(json.dumps(comparison,indent=2))
    result=dict(passed=True,checked_runtime_files=len(old['files']),checked_epoch5_source_files=len(baseline['runtime_hashes']),epoch5_source_unchanged=True,release_model_loader_diff_empty=True,release_revision=revision,eval_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),verified_unix=time.time())
    (R/'source_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
