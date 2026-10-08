import json
import pytest
from scripts import run_loop_grid_job as job
from test_dense_s30_job import training,evaluation,CANDIDATE

@pytest.mark.parametrize('video,action',[(4,2),(2,2),(1,4)])
def test_depth_gates(tmp_path,monkeypatch,video,action):
    monkeypatch.setattr(job,'VIDEO_LOOPS',video);monkeypatch.setattr(job,'ACTION_LOOPS',action)
    training(tmp_path)
    p=tmp_path/'manifest.json';m=json.loads(p.read_text())
    m.update(version='v0',loops=video,action_core_loops=action,policy_parameters=584536135)
    p.write_text(json.dumps(m));job.verify_training(tmp_path)
    m['action_core_loops']=3;p.write_text(json.dumps(m))
    with pytest.raises(ValueError):job.verify_training(tmp_path)
    cmd=job.training_command(CANDIDATE,tmp_path/'train',tmp_path/'cache')
    assert cmd[cmd.index('--video-loops')+1]==str(video)
    assert cmd[cmd.index('--action-loops')+1]==str(action)
    assert '--resume' not in cmd and '--max-updates' not in cmd

@pytest.mark.parametrize('video,action',[(4,2),(2,2),(1,4)])
def test_evaluation_gates(tmp_path,monkeypatch,video,action):
    monkeypatch.setattr(job,'VIDEO_LOOPS',video);monkeypatch.setattr(job,'ACTION_LOOPS',action)
    evaluation(tmp_path)
    p=tmp_path/'manifest.json';m=json.loads(p.read_text());m.update(version='v0',loops=video,action_loops=action)
    p.write_text(json.dumps(m))
    p=tmp_path/'summary.json';s=json.loads(p.read_text());s['version']='v0';p.write_text(json.dumps(s))
    job.verify_evaluation(tmp_path,tmp_path/'latest.pt')
    m['action_loops']=3;(tmp_path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):job.verify_evaluation(tmp_path,tmp_path/'latest.pt')
