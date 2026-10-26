import importlib
import pytest

def runner():
    return importlib.import_module('run_partition_grid')

def test_grid_queue_is_complete_and_disjoint():
    r=runner();items=r.QUEUES['A']+r.QUEUES['B']
    assert len(items)==len(set(items))==7
    for label in items:
        partition,loops=r.GRID[label]
        assert sum(partition)==12 and partition[0]+partition[1]*loops+partition[2]==30

def test_production_command_is_fresh_and_full(tmp_path):
    r=runner()
    for label in r.GRID:
        c=r.command(label,dict(backend='ddp',microbatch=8,checkpoint_blocks=False),tmp_path/label)
        assert '--resume' not in c and '--max-updates' not in c
        assert c[c.index('--global-batch')+1]=='128'
        assert c[c.index('--epochs')+1]=='10'
        assert c[c.index('--dataset-scope')+1]=='full_libero'
        assert c[c.index('--validation-samples')+1]=='0'
        assert c[c.index('--block-partition')+1:c.index('--block-partition')+4]==list(map(str,r.GRID[label][0]))

def test_source_guard_rejects_modified_file(tmp_path):
    r=runner(); (tmp_path/'src').mkdir();(tmp_path/'scripts').mkdir()
    f=tmp_path/'src'/'a.py';f.write_text('original')
    hashes=r.hash_sources(tmp_path);f.write_text('changed')
    with pytest.raises(ValueError,match='Source'):r.require_sources(hashes,tmp_path)

def test_production_rejects_smoke_config_change_before_starting(tmp_path):
    import json
    r=runner();sources=r.hash_sources()
    release=dict(sources=sources,config=dict(backend='ddp',microbatch=16))
    (tmp_path/'release.json').write_text(json.dumps(release))
    for lane in ('A','B'):
        (tmp_path/f'smoke_{lane}_complete.json').write_text(json.dumps(dict(sources=sources,config=dict(backend='ddp',microbatch=8))))
    with pytest.raises(ValueError,match='configuration'):r.production(tmp_path,'A')
    assert not (tmp_path/'production').exists()

def test_stale_export_cannot_be_relabelled_as_current_checkpoint(tmp_path):
    import json,hashlib
    r=runner();weights=tmp_path/'policy.pt';weights.write_bytes(b'old weights')
    info=tmp_path/'provenance.json';info.write_text(json.dumps(dict(source_checkpoint_sha256='old-source',sha256=hashlib.sha256(weights.read_bytes()).hexdigest())))
    with pytest.raises(ValueError,match='checkpoint'):r.validate_export(tmp_path,'new-source')
    r.validate_export(tmp_path,'old-source')
    weights.write_bytes(b'corrupted weights')
    with pytest.raises(ValueError):r.validate_export(tmp_path,'old-source')

def test_inference_uses_only_current_proprioception_and_image():
    import torch
    from check_partition_checkpoint import inference_inputs
    sample=dict(video=torch.randn(3,9,224,448),proprio=torch.randn(32,8),
                context=torch.randn(128,4096),context_mask=torch.ones(128,dtype=torch.bool))
    kwargs=inference_inputs(sample)
    assert kwargs['proprio'].shape==(1,8)
    assert torch.equal(kwargs['proprio'][0],sample['proprio'][0])
    assert torch.equal(kwargs['input_image'],sample['video'][:,0])
    changed={**sample,'proprio':sample['proprio'].clone()};changed['proprio'][1:]+=100
    assert torch.equal(inference_inputs(changed)['proprio'],kwargs['proprio'])
