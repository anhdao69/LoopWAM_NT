"""Safety regressions: cancellation requires complete training AND evaluation."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/watch_v0_evaluate.py'
    assert path.exists(), 'v0 automatic evaluator has not been implemented'
    spec=importlib.util.spec_from_file_location('watch_v0_evaluate',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def train(tmp_path):
    p=tmp_path/'train';p.mkdir()
    write(p/'manifest.json',dict(version='v0',resume=None,epochs=10,global_batch=128,
        world_size=2,microbatch=8,gradient_accumulation=8,loops=4,policy_parameters=584536135,
        train_windows=92678,planned_updates=7250,planned_windows=926780,slurm_job_id='4689'))
    write(p/'timing.json',dict(status='complete',completed_updates=7250,windows_seen=926780))
    write(p/'trainer_state.json',dict(update=7250,epoch=9,windows_seen=926780,micro=5792))
    (p/'latest.pt').write_bytes(b'test-checkpoint')
    return p


def identity(job='4689',start='2026-10-05T00:00:00'):
    return dict(JobId=job,UserId=f'user({os.getuid()})',StartTime=start,JobState='RUNNING')


class Cluster:
    def __init__(self,train,mode='good'):
        self.train=train;self.mode=mode;self.events=[];self.evaluated=False
    def describe(self,job):
        return identity(start='changed' if self.evaluated and self.mode=='reused' else '2026-10-05T00:00:00')
    def steps(self,job):
        return ['4689.0','4689.99'] if self.evaluated and self.mode=='extra_step' else ['4689.0']
    def evaluate(self,job,train,out):
        self.events.append('evaluate');self.evaluated=True
        if self.mode=='failed':raise RuntimeError('evaluation failed')
        out.mkdir()
        m=dict(mode='final_rollout',version='v0',loops=4,checkpoint=str(train/'latest.pt'),checkpoint_sha256='hash')
        rows=[]
        for task in range(10):
            for ep in range(10):
                video=out/f'{task}_{ep}.mp4';video.write_bytes(b'video')
                rows.append(dict(task_id=task,episode_index=ep,checkpoint_sha256='hash',mode='final_rollout',video=str(video)))
        s=dict(mode='final_rollout',version='v0',checkpoint_step=7250,checkpoint_sha256='hash',total_episodes=100,
               per_task={str(t):dict(episodes=10) for t in range(10)},episodes=rows,success_rate=.3)
        if self.mode=='missing_video':Path(rows[-1]['video']).unlink()
        if self.mode=='partial':s['total_episodes']=99
        if self.mode=='duplicate':rows[-1]=rows[0]
        write(out/'manifest.json',m);write(out/'summary.json',s)
    def cancel(self,job):self.events.append(('cancel',job))


def test_success_evaluates_then_cancels_only_bound_allocation(train,tmp_path):
    m=module();c=Cluster(train);events=[]
    m.finish_run(train,tmp_path/'eval','4689',identity(),c,lambda stage,**kw:events.append(stage))
    assert c.events==['evaluate',('cancel','4689')]
    assert events[-1]=='complete'


@pytest.mark.parametrize('mode',['failed','missing_video','partial','duplicate','extra_step','reused'])
def test_failed_or_unverified_evaluation_never_cancels(train,tmp_path,mode):
    m=module();c=Cluster(train,mode)
    with pytest.raises((ValueError,RuntimeError)):
        m.finish_run(train,tmp_path/'eval','4689',identity(),c,lambda *a,**k:None)
    assert not any(isinstance(e,tuple) and e[0]=='cancel' for e in c.events)


@pytest.mark.parametrize('file,key,value',[('timing','status','running'),('timing','completed_updates',7249),
    ('trainer_state','micro',5791),('trainer_state','windows_seen',926779),('manifest','version','v1'),
    ('manifest','slurm_job_id','4659'),('manifest','resume','old.pt')])
def test_partial_wrong_run_or_resumed_training_never_evaluates(train,tmp_path,file,key,value):
    m=module();p=train/f'{file}.json';d=json.loads(p.read_text());d[key]=value;write(p,d);c=Cluster(train)
    with pytest.raises(ValueError):m.finish_run(train,tmp_path/'eval','4689',identity(),c,lambda *a,**k:None)
    assert c.events==[]


def test_unfinished_step_is_not_ready_even_if_completion_file_exists(train):
    m=module()
    assert not m.ready_for_evaluation(train,['4689.0','4689.61'],'4689.61')
    assert m.ready_for_evaluation(train,['4689.0'],'4689.61')


def test_dead_training_step_with_incomplete_training_is_error(train):
    m=module();write(train/'timing.json',dict(status='running'))
    with pytest.raises(ValueError):m.ready_for_evaluation(train,['4689.0'],'4689.61')
