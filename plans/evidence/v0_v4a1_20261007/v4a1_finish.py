import json,subprocess,pathlib,math,statistics
from scripts.run_v0_v4a1_job import Pipeline,source_hashes
from scripts.run_dense_v2_job import verify_benchmark_sources
R=pathlib.Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt')
O=R/'v0_v4a1_job4728_20261007'
CACHE=R/'dense_s30_job4719_20261006/production_latents'
results=[]
def validate(d):
    verify_benchmark_sources(d,source_hashes())
    assert (d['version'],d['world_size'],d['loops'],d['action_loops'],d['global_batch'])==('v0',2,4,1,128)
    assert d['precision']['policy_dtype']=='float32' and d['precision']['optimizer_moment_dtypes']==['torch.float32']
    assert d['microbatch']*2*d['accumulation']==128 and d['policy_parameters']==584536135
    assert d['structured_attention'] and d['fused']
    assert all(math.isfinite(r[k]) for r in d['records'] for k in ('seconds','loss','grad_norm'))
for backend in ('ddp','zero1','zero2'):
 for mb in (8,16):
    f=O/f'bench_r1_{backend}_{mb}/result.json'
    if f.exists():
        d=json.loads(f.read_text());validate(d);results.append(d)
    else:
        assert 'CUDA out of memory' in (O/f'bench_r1_{backend}_{mb}.log').read_text()
p=Pipeline(O)
def trial(backend,mb,label,updates):
    target=O/label
    cmd=['torchrun','--standalone','--nproc_per_node=2','scripts/benchmark_loopwam.py','--version','v0','--action-loops','1','--backend',backend,'--microbatch',str(mb),'--updates',str(updates),'--workers','4','--structured-attention','--latent-cache-dir',str(CACHE),'--output-dir',str(target)]
    p.run(label,cmd)
    d=json.loads((target/'result.json').read_text());validate(d);results.append(d)
trial('ddp',4,'bench_final_ddp4',7)
# Longer trials of the two fastest candidates resolve startup/timing noise.
for i,d in enumerate(sorted(results,key=lambda d:d['steady_seconds'])[:2]):
    trial(d['backend'],d['microbatch'],f'bench_repeat_{i}',20)
groups={}
for d in results:groups.setdefault((d['backend'],d['microbatch']),[]).extend(r['seconds'] for r in d['records'][2:])
rates={key:statistics.mean(v) for key,v in groups.items()}
best=min(rates,key=rates.get)
# Within 2%, preserve the original v0 DDP layout for the fairest control.
if rates.get(('ddp',8),float('inf')) <= rates[best]*1.02:best=('ddp',8)
candidate=dict(backend=best[0],microbatch=best[1],workers=4,checkpoint_blocks=False)
(O/'selection.json').write_text(json.dumps(dict(selected=candidate,pooled_seconds={str(k):v for k,v in rates.items()},note='Within 2 percent, retain baseline DDP microbatch8 to preserve RNG and optimizer layout.'),indent=2))
try:
    p.prepare(candidate,CACHE)
    baseline=json.loads((R/'v0_scratch_fast_bs128_20261005/manifest.json').read_text())
    actual=json.loads((O/'warm_train/manifest.json').read_text())
    for key in ('seed','global_batch','epochs','train_windows','planned_updates','planned_windows','asset_sha256'):
        assert actual[key]==baseline[key],key
    for key in ('train_episodes','validation_episodes','normalization_sha256','camera_order','video_offsets','action_offsets','content_files'):
        assert actual['data'][key]==baseline['data'][key],key
    assert actual['resume'] is None and actual['action_core_loops']==1 and actual['loops']==4
    (O/'fairness_verified.json').write_text(json.dumps(dict(baseline=str(R/'v0_scratch_fast_bs128_20261005'),checks='seed, budget, donor and VAE hashes, complete split, normalization, camera/window contract, source data hashes',video_loops=4,action_loops=1),indent=2))
    p.production()
except BaseException as exc:
    p.status("failed",error=repr(exc))
    raise
