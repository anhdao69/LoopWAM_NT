import importlib

def test_centered_queue_has_exact_six_runs_and_four_gpu_batches(tmp_path):
    r=importlib.import_module('run_centered_partition_grid')
    labels=r.QUEUES['A']+r.QUEUES['B']
    assert len(labels)==len(set(labels))==6
    assert 'p3c6s3_k4' not in labels
    for label in labels:
        cmd=r.command(label,dict(backend='ddp',microbatch=8,checkpoint_blocks=False),tmp_path/label)
        assert '--nproc_per_node=4' in cmd and '--resume' not in cmd
        assert cmd[cmd.index('--donor-strategy')+1]=='stage_centered'
        assert cmd[cmd.index('--global-batch')+1]=='128'
        assert cmd[cmd.index('--epochs')+1]=='10'

def test_benchmark_gate_rejects_precision_and_budget_corruption():
    import copy,pytest
    from run_centered_partition_grid import validate_benchmark
    r=dict(global_batch=128,world_size=4,train_windows=277713,donor_strategy='stage_centered',
           precision=dict(policy_dtype='float32',optimizer_moment_dtypes=['torch.float32']),
           steady_seconds=2.,records=[dict(loss=1.,grad_norm=1.,seconds=2.) for _ in range(8)])
    validate_benchmark(r)
    for key,value in [('world_size',2),('global_batch',64),('train_windows',277712),('steady_seconds',float('nan'))]:
        bad=copy.deepcopy(r);bad[key]=value
        with pytest.raises(ValueError):validate_benchmark(bad)
    bad=copy.deepcopy(r);bad['precision']['optimizer_moment_dtypes']=['torch.bfloat16']
    with pytest.raises(ValueError):validate_benchmark(bad)

def test_interrupted_export_checks_weights_before_accepting(tmp_path):
    import pytest,torch
    from run_centered_partition_grid import validate_export
    source=dict(format_version='loopwam-s-partition-v1',version='v0',step=21700,architecture={'donor_strategy':'stage_centered'},mot={'x':torch.ones(2)},proprio_encoder={'x':torch.ones(1)})
    torch.save(source,tmp_path/'policy.pt')
    validate_export(tmp_path,'source-sha',source)
    bad=dict(source,mot={'x':torch.zeros(2)})
    torch.save(bad,tmp_path/'policy.pt')
    with pytest.raises(ValueError,match='export'):validate_export(tmp_path,'source-sha',source)
