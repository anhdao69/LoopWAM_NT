import json
from pathlib import Path
import pytest

from scripts.run_full_libero_v0_job import training_command, evaluation_command, verify_training, source_hashes


def test_source_freeze_binds_simulator_preprocessing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    helper = tmp_path/'experiments/libero/libero_utils.py'
    helper.parent.mkdir(parents=True)
    helper.write_text('camera_rotation = 180\n')
    before = source_hashes()
    assert 'experiments/libero/libero_utils.py' in before
    helper.write_text('camera_rotation = 0\n')
    assert source_hashes() != before


def test_commands_use_all_demonstrations_and_never_resume(tmp_path):
    cmd = training_command('ddp', 8, 4, tmp_path/'train', tmp_path/'cache')
    assert cmd[cmd.index('--dataset-scope') + 1] == 'full_libero'
    assert cmd[cmd.index('--dataset-dir') + 1] == 'data/lerobot_v30'
    assert cmd[cmd.index('--epochs') + 1] == '10'
    assert cmd[cmd.index('--global-batch') + 1] == '128'
    assert '--resume' not in cmd and '--max-updates' not in cmd
    for suite in ['libero_spatial','libero_object','libero_goal','libero_10']:
        command = evaluation_command(tmp_path/'train', tmp_path/suite, suite)
        assert command[command.index('--suite') + 1] == suite
        assert command[command.index('--episodes-per-task') + 1] == '10'
        assert '--smoke' not in command


def fixture_run(directory):
    manifest = dict(version='v0', resume=None, epochs=10, global_batch=128, world_size=4,
        microbatch=8, gradient_accumulation=4, loops=4, policy_parameters=584536135,
        train_windows=277713, val_windows=0, planned_updates=21700, planned_windows=2777130,
        initialization_mode='canonical_wan_artifact_fresh_optimizer', max_updates=None,
        dataset_scope='full_libero', validation_samples=0,
        data=dict(dataset_scope='full_libero', split='all_train', train_windows=277713,
                  suites=['libero_spatial','libero_object','libero_goal','libero_10']))
    timing = dict(status='complete',completed_updates=21700,windows_seen=2777130)
    state = dict(update=21700,windows_seen=2777130,epoch=9,next_micro=8679)
    for name, value in [('manifest',manifest),('timing',timing),('trainer_state',state)]:
        (directory/f'{name}.json').write_text(json.dumps(value))
    (directory/'latest.pt').write_bytes(b'checkpoint')
    return manifest,timing,state


def test_complete_training_gate(tmp_path):
    fixture_run(tmp_path)
    assert verify_training(tmp_path)['planned_updates'] == 21700


@pytest.mark.parametrize('file,key,value', [
    ('manifest','resume','old.pt'),('manifest','max_updates',21700),
    ('manifest','train_windows',92678),('manifest','planned_updates',7250),
    ('manifest','val_windows',100),('timing','status','running'),
    ('timing','windows_seen',2777129),('trainer_state','epoch',8),
    ('trainer_state','next_micro',8678)])
def test_incomplete_or_wrong_training_rejected(tmp_path,file,key,value):
    fixture_run(tmp_path)
    path = tmp_path/f'{file}.json'
    data=json.loads(path.read_text()); data[key]=value; path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        verify_training(tmp_path)
