import json
import pytest
from scripts.run_robustness_job import verify


def fixture_files(path):
    video=path/'video.mp4';video.write_bytes(b'nonempty')
    row=dict(suite='libero_goal_lan',task_id=0,episode_index=0,
             checkpoint_sha256='hash',video=str(video))
    (path/'rank0.jsonl').write_text(json.dumps(row)+'\n')
    (path/'done_rank0.json').write_text(json.dumps(dict(completed=1,checkpoint_sha256='hash')))
    return row


def test_valid_complete_result(tmp_path):
    fixture_files(tmp_path)
    assert len(verify(tmp_path,1,'hash',{('libero_goal_lan',0,0)}))==1


def test_missing_episode_rejected(tmp_path):
    fixture_files(tmp_path)
    with pytest.raises(ValueError,match='missing episodes'):
        verify(tmp_path,1,'hash',{('libero_goal_lan',0,0),('libero_goal_lan',0,1)})


def test_duplicate_episode_rejected(tmp_path):
    row=fixture_files(tmp_path)
    (tmp_path/'rank0.jsonl').write_text((json.dumps(row)+'\n')*2)
    (tmp_path/'done_rank0.json').write_text(json.dumps(dict(completed=2,checkpoint_sha256='hash')))
    with pytest.raises(ValueError,match='Duplicate'):
        verify(tmp_path,1,'hash')


def test_checkpoint_mix_rejected(tmp_path):
    row=fixture_files(tmp_path);row['checkpoint_sha256']='other'
    (tmp_path/'rank0.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='Checkpoint mismatch'):
        verify(tmp_path,1,'hash')


def test_missing_video_rejected(tmp_path):
    fixture_files(tmp_path);(tmp_path/'video.mp4').unlink()
    with pytest.raises(ValueError,match='missing video'):
        verify(tmp_path,1,'hash')
