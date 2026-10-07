import json,pathlib
from scripts.run_v0_v4a1_job import Pipeline,source_hashes,evaluation_command,verify_evaluation,timing_estimate,write_json
from scripts.run_dense_v2_job import verify_benchmark_sources
R=pathlib.Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt')
O=R/'v0_v4a1_job4728_20261007'
CACHE=R/'dense_s30_job4719_20261006/production_latents'
candidate=json.loads((O/'selection.json').read_text())['selected']
initial=source_hashes()
verify_benchmark_sources(json.loads((O/'bench_repeat_0/result.json').read_text()),initial)
for name in ('cold_train','warm_train'):
    m=json.loads((O/name/'manifest.json').read_text())
    for key,value in m['code_files_sha256'].items():
        if key.startswith('src/'):
            assert initial[key]==value,key
    t=json.loads((O/name/'timing.json').read_text())
    assert t['status']=='smoke_complete' and t['completed_updates']==10
(O/'preflight_failed_status.json').write_bytes((O/'pipeline_status.json').read_bytes())
p=Pipeline(O)
try:
    smoke=O/'smoke_evaluation_retry'
    p.run('smoke_evaluation_retry',evaluation_command(O/'warm_train',smoke,True))
    verify_evaluation(smoke,O/'warm_train/latest.pt',smoke=True)
    assert source_hashes()==initial
    cold=json.loads((O/'cold_train/timing.json').read_text())
    warm=json.loads((O/'warm_train/timing.json').read_text())
    estimate=timing_estimate(cold,warm)
    estimate['estimated_cached_production_hours']=(7250*warm['measured_mean_update_seconds']+10*warm['checkpoint_seconds_total'])/3600
    write_json(O/'prepared.json',dict(selected=candidate,production_cache=str(CACHE),source_hashes=initial,**estimate))
    baseline=json.loads((R/'v0_scratch_fast_bs128_20261005/manifest.json').read_text())
    actual=json.loads((O/'warm_train/manifest.json').read_text())
    for key in ('seed','global_batch','epochs','train_windows','planned_updates','planned_windows','asset_sha256'):
        assert actual[key]==baseline[key],key
    for key in ('train_episodes','validation_episodes','normalization_sha256','camera_order','video_offsets','action_offsets','content_files'):
        assert actual['data'][key]==baseline['data'][key],key
    assert actual['resume'] is None and actual['action_core_loops']==1 and actual['loops']==4
    (O/'fairness_verified.json').write_text(json.dumps(dict(baseline=str(R/'v0_scratch_fast_bs128_20261005'),checks='seed, budget, donor and VAE hashes, complete split, normalization, camera/window contract, source data hashes',video_loops=4,action_loops=1),indent=2))
    p.status('prepared',**estimate)
    p.production()
except BaseException as exc:
    p.status('failed',error=repr(exc))
    raise
