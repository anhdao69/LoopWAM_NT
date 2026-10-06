import json
import math
from pathlib import Path
import pytest
from scripts import run_dense_s30_job as job

CANDIDATE = dict(backend='zero1', microbatch=4, workers=8, checkpoint_blocks=False)


def training(path):
    payloads = dict(manifest=dict(version='dense_s30', resume=None, epochs=10,
        initialization_mode='canonical_wan_artifact_fresh_optimizer', global_batch=128,
        world_size=2, microbatch=4, gradient_accumulation=16, loops=1,
        policy_parameters=1416114247, train_windows=92678, planned_updates=7250,
        planned_windows=926780, policy_dtype='float32', optimizer_state_dtype='float32',
        data=dict(train_episodes=list(range(344)), validation_episodes=list(range(344,388)))),
        timing=dict(status='complete', completed_updates=7250, windows_seen=926780),
        trainer_state=dict(update=7250, windows_seen=926780, epoch=9, next_micro=math.ceil(92678/8)))
    for name, value in payloads.items():
        (path / (name + '.json')).write_text(json.dumps(value))
    (path / 'latest.pt').write_bytes(b'checkpoint')


@pytest.mark.parametrize('file,key,value', [
    ('manifest','version','dense_s12'), ('manifest','world_size',4),
    ('manifest','policy_parameters',584536135), ('manifest','resume','previous.pt'),
    ('manifest','initialization_mode','resume_checkpoint'), ('manifest','planned_updates',7249),
    ('manifest','train_windows',92679), ('manifest','policy_dtype','bfloat16'),
    ('timing','status','smoke_complete'), ('timing','completed_updates',7249),
    ('trainer_state','epoch',8), ('trainer_state','windows_seen',926779),
    ('trainer_state','next_micro',1)])
def test_final_training_gate(tmp_path,file,key,value):
    training(tmp_path)
    assert job.verify_training(tmp_path)['loops']==1
    path=tmp_path/(file+'.json'); data=json.loads(path.read_text()); data[key]=value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): job.verify_training(tmp_path)


def evaluation(path, smoke=False):
    mode='smoke' if smoke else 'final_rollout'
    tasks=range(2) if smoke else range(10)
    episodes=1 if smoke else 10
    (path/'video.mp4').write_bytes(b'video')
    m=dict(mode=mode, version='dense_s30', loops=1, suite='libero_10', world_size=2,
           checkpoint=str(path/'latest.pt'), checkpoint_sha256='sha')
    rows=[dict(task_id=i,episode_index=j,mode=mode,suite='libero_10',checkpoint_sha256='sha',video=str(path/'video.mp4'))
          for i in tasks for j in range(episodes)]
    s=dict(mode=mode,suite='libero_10',version='dense_s30',checkpoint_step=10 if smoke else 7250,
           checkpoint_sha256='sha',total_episodes=len(rows),episodes=rows,
           per_task={str(i):dict(episodes=episodes) for i in tasks},success_rate=.1)
    (path/'manifest.json').write_text(json.dumps(m)); (path/'summary.json').write_text(json.dumps(s))
    return s


@pytest.mark.parametrize('key,value',[('total_episodes',99),('mode','smoke'),('version','v2'),
                                      ('checkpoint_step',10),('checkpoint_sha256','wrong'),('suite','libero_goal')])
def test_final_evaluation_gate(tmp_path,key,value):
    s=evaluation(tmp_path)
    job.verify_evaluation(tmp_path,tmp_path/'latest.pt')
    s[key]=value;(tmp_path/'summary.json').write_text(json.dumps(s))
    with pytest.raises(ValueError):job.verify_evaluation(tmp_path,tmp_path/'latest.pt')


def test_duplicate_and_missing_video_rejected(tmp_path):
    s=evaluation(tmp_path);s['episodes'][-1]=s['episodes'][0]
    (tmp_path/'summary.json').write_text(json.dumps(s))
    with pytest.raises(ValueError,match='Duplicate'):job.verify_evaluation(tmp_path,tmp_path/'latest.pt')
    evaluation(tmp_path);(tmp_path/'video.mp4').unlink()
    with pytest.raises(ValueError,match='video missing'):job.verify_evaluation(tmp_path,tmp_path/'latest.pt')
    evaluation(tmp_path,True)
    job.verify_evaluation(tmp_path,tmp_path/'latest.pt',smoke=True)


def test_commands_are_fresh_two_rank_correct_budget(tmp_path):
    command=job.training_command(CANDIDATE,tmp_path/'train',tmp_path/'cache')
    assert '--nproc_per_node=2' in command and '--resume' not in command and '--max-updates' not in command
    assert command[command.index('--version')+1]=='dense_s30'
    assert command[command.index('--epochs')+1]=='10'
    smoke=job.training_command({**CANDIDATE,'checkpoint_blocks':True},tmp_path/'smoke',tmp_path/'cache',10)
    assert '--checkpoint-blocks' in smoke and smoke[smoke.index('--max-updates')+1]=='10'
    final=job.evaluation_command(tmp_path/'train',tmp_path/'eval')
    assert '--smoke' not in final and final[final.index('--episodes-per-task')+1]=='10'


def test_sources_include_simulator_utilities(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    path=tmp_path/'experiments/libero/helper.py';path.parent.mkdir(parents=True);path.write_text('old')
    first=job.source_hashes();path.write_text('new')
    assert first != job.source_hashes()


def test_production_changed_source_and_existing_outputs_fail_closed(tmp_path,monkeypatch):
    (tmp_path/'prepared.json').write_text(json.dumps(dict(source_hashes={'a':'old'},selected=CANDIDATE)))
    pipeline=job.Pipeline(tmp_path)
    monkeypatch.setattr(job,'source_hashes',lambda:{'a':'new'})
    with pytest.raises(ValueError,match='Source changed'):pipeline.production()
    monkeypatch.setattr(job,'source_hashes',lambda:{'a':'old'})
    (tmp_path/'production_latents').mkdir()
    with pytest.raises(ValueError,match='Production outputs exist'):pipeline.production()


def test_failed_training_never_runs_evaluation(tmp_path,monkeypatch):
    (tmp_path/'prepared.json').write_text(json.dumps(dict(source_hashes={},selected=CANDIDATE)))
    monkeypatch.setattr(job,'source_hashes',lambda:{})
    pipeline=job.Pipeline(tmp_path); stages=[]
    def fail(stage,command):
        stages.append(stage)
        raise RuntimeError('training failed')
    monkeypatch.setattr(pipeline,'run',fail)
    with pytest.raises(RuntimeError):pipeline.production()
    assert stages==['dense_s30_training']


def test_eta_includes_measured_saves():
    cold=dict(measured_mean_update_seconds=3.,checkpoint_seconds_total=20.,checkpoint_count=1)
    warm=dict(measured_mean_update_seconds=2.,checkpoint_seconds_total=10.,checkpoint_count=1)
    estimate=job.timing_estimate(cold,warm)
    assert estimate['estimated_training_hours']==(725*(3+9*2)+150)/3600


def test_episode_suite_and_post_eval_source_change_fail_closed(tmp_path,monkeypatch):
    s=evaluation(tmp_path);s['episodes'][0]['suite']='libero_goal'
    (tmp_path/'summary.json').write_text(json.dumps(s))
    with pytest.raises(ValueError,match='protocol mismatch'):job.verify_evaluation(tmp_path,tmp_path/'latest.pt')
    (tmp_path/'prepared.json').write_text(json.dumps(dict(source_hashes={},selected=CANDIDATE)))
    versions=iter([{}, {}, {'changed':'yes'}])
    monkeypatch.setattr(job,'source_hashes',lambda:next(versions))
    monkeypatch.setattr(job,'verify_training',lambda path:None)
    monkeypatch.setattr(job,'verify_evaluation',lambda *args:dict(success_rate=.5))
    pipeline=job.Pipeline(tmp_path)
    monkeypatch.setattr(pipeline,'run',lambda *args:None)
    with pytest.raises(ValueError,match='Source changed during evaluation'):pipeline.production()
