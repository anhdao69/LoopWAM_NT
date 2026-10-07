import json
import pytest
from scripts import run_v0_v4a1_job as job
from test_dense_s30_job import training,evaluation,CANDIDATE


def test_training_rejects_wrong_action_budget(tmp_path):
    training(tmp_path)
    p=tmp_path/'manifest.json';m=json.loads(p.read_text())
    m.update(version='v0',loops=4,action_core_loops=1,policy_parameters=584536135)
    p.write_text(json.dumps(m));job.verify_training(tmp_path)
    m['action_core_loops']=4;p.write_text(json.dumps(m))
    with pytest.raises(ValueError):job.verify_training(tmp_path)


def test_eval_rejects_coupled_depth(tmp_path):
    evaluation(tmp_path)
    p=tmp_path/'manifest.json';m=json.loads(p.read_text());m.update(version='v0',loops=4,action_loops=1)
    p.write_text(json.dumps(m))
    p=tmp_path/'summary.json';s=json.loads(p.read_text());s.update(version='v0');p.write_text(json.dumps(s))
    job.verify_evaluation(tmp_path,tmp_path/'latest.pt')
    m['action_loops']=4;(tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):job.verify_evaluation(tmp_path,tmp_path/'latest.pt')


def test_failed_training_stops_queue(tmp_path,monkeypatch):
    (tmp_path/'prepared.json').write_text(json.dumps(dict(source_hashes={},selected=CANDIDATE,production_cache='cache')))
    monkeypatch.setattr(job,'source_hashes',lambda:{})
    pipeline=job.Pipeline(tmp_path);stages=[]
    def fail(stage,cmd):
        stages.append(stage)
        assert cmd[cmd.index('--action-loops')+1]=='1'
        assert '--resume' not in cmd and '--max-updates' not in cmd
        raise RuntimeError('failed')
    monkeypatch.setattr(pipeline,'run',fail)
    with pytest.raises(RuntimeError):pipeline.production()
    assert stages==['training']
