import copy
import json
import pytest
from scripts.run_loopwam_v1_job import select_microbatch, verify_finished_training


def result(micro,seconds):
    return dict(version='v1',backend='ddp',global_batch=128,microbatch=micro,accumulation=64//micro,steady_seconds=seconds,records=[dict(loss=.3,grad_norm=1.)]*5)


def test_selects_measured_fastest_and_rejects_wrong_objective():
    assert select_microbatch([result(4,8),result(8,5)])['microbatch']==8
    for key,value in [('version','v0'),('global_batch',256),('steady_seconds',float('nan'))]:
        bad=result(8,5);bad[key]=value
        with pytest.raises(ValueError):select_microbatch([bad])
    with pytest.raises(ValueError):select_microbatch([])


def completed(tmp_path):
    for name,data in [('manifest',dict(version='v1',resume=None,epochs=10,global_batch=128,planned_updates=7250,planned_windows=926780)),('timing',dict(status='complete',completed_updates=7250)),('trainer_state',dict(update=7250,windows_seen=926780))]:
        (tmp_path/(name+'.json')).write_text(json.dumps(data))
    (tmp_path/'latest.pt').touch()


def test_only_completed_fresh_v1_can_enter_final_inference(tmp_path):
    completed(tmp_path)
    assert verify_finished_training(tmp_path)['version']=='v1'
    path=tmp_path/'timing.json';path.write_text(json.dumps(dict(status='smoke_complete',completed_updates=3)))
    with pytest.raises(ValueError,match='incomplete'):verify_finished_training(tmp_path)
    completed(tmp_path)
    path=tmp_path/'manifest.json';d=json.loads(path.read_text());d['resume']='old.pt';path.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='fresh'):verify_finished_training(tmp_path)
