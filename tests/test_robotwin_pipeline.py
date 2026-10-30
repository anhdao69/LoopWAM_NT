import copy
import json
from pathlib import Path
import pytest

def parent_contract():
    root=Path(__file__).resolve().parents[1]
    return json.loads((root/'plans/evidence/robotwin_full_v0_20261028/release_v2/smoke/manifest.json').read_text())['contract']

def test_continuation_budget_and_cosine_endpoints():
    from fastwam.robotwin_pipeline import continuation_contract,continuation_factor,validate_cursor
    c=continuation_contract(parent_contract(),'a'*64)
    assert c['epochs']==10 and c['planned_updates']==474620 and c['planned_windows']==60751030
    assert c['save_epochs']=='6,7,8,9,10' and c['phase_start_update']==237310
    assert continuation_factor(237310)==1
    assert continuation_factor(474619)==pytest.approx(.01)
    assert 3e-5*continuation_factor(237310)==3e-5
    validate_cursor(237310,30375515,5,0)
    validate_cursor(474620,60751030,10,0)
    with pytest.raises(ValueError):validate_cursor(237310,237310*128,5,0)
    with pytest.raises(ValueError):validate_cursor(47462,6075103,0,47462)

def test_handoff_rejects_wrong_boundary_or_contract():
    from fastwam.robotwin_pipeline import continuation_contract,validate_parent_state
    old=parent_contract();new=continuation_contract(old,'a'*64)
    state=dict(contract=old,update=237310,windows_seen=30375515,next_epoch=5,next_batch=0)
    payload=dict(step=237310,optimizer={'state':{1:{'step':237310}}},training_state=state)
    validate_parent_state(payload,new)
    for field,value in [('seed',43),('normalization_sha256','wrong'),('video_loops',3),('global_batch',64)]:
        bad=copy.deepcopy(payload);bad['training_state']['contract'][field]=value
        with pytest.raises(ValueError):validate_parent_state(bad,new)
    bad=copy.deepcopy(payload);bad['training_state']['update']-=1
    with pytest.raises(ValueError):validate_parent_state(bad,new)
    bad=copy.deepcopy(payload);bad.pop('optimizer')
    with pytest.raises(ValueError):validate_parent_state(bad,new)

def test_resume_log_reconciles_only_after_saved_cursor(tmp_path):
    from fastwam.robotwin_pipeline import reconcile_metrics
    p=tmp_path/'metrics.jsonl';rows=[{'update':i} for i in range(237311,237315)]
    p.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    reconcile_metrics(p,237312,first_update=237311)
    assert [json.loads(x)['update'] for x in p.read_text().splitlines()]==[237311,237312]
    assert len(list(tmp_path.glob('metrics.superseded.*.jsonl')))==1
    p.write_text(json.dumps({'update':237312})+'\n')
    with pytest.raises(ValueError):reconcile_metrics(p,237312,first_update=237311)

def test_eval_gate_requires_every_unique_episode_and_video():
    from fastwam.robotwin_pipeline import validate_eval_summary
    tasks=['task_a','task_b'];sha='b'*64
    cells={f'{t}/{c}':dict(complete=True,exit_code=0,episodes=[dict(seed=100000+i,success=bool(i%2),video_verified=True) for i in range(3)]) for t in tasks for c in ['demo_clean','demo_randomized']}
    summary=dict(checkpoint_sha256=sha,finished=cells)
    validate_eval_summary(summary,sha,tasks,episodes=3)
    for change in ['missing','duplicate','video','exit','sha']:
        bad=copy.deepcopy(summary);k=next(iter(cells))
        if change=='missing':bad['finished'].pop(k)
        if change=='duplicate':bad['finished'][k]['episodes'][1]['seed']=100000
        if change=='video':bad['finished'][k]['episodes'][0]['video_verified']=False
        if change=='exit':bad['finished'][k]['exit_code']=1
        if change=='sha':bad['checkpoint_sha256']='bad'
        with pytest.raises(ValueError):validate_eval_summary(bad,sha,tasks,episodes=3)

def test_stage_survives_controller_detach_and_prevents_duplicate_child(tmp_path):
    import subprocess,sys,time
    root=Path(__file__).resolve().parents[1];marker=tmp_path/'marker';status=tmp_path/'status.json';command=tmp_path/'command.json'
    command.write_text(json.dumps([sys.executable,'-c',f'import pathlib,time; pathlib.Path({str(marker)!r}).write_text("once"); time.sleep(1)']))
    args=[sys.executable,str(root/'scripts/robotwin_queue_stage.py'),'--command',str(command),'--status',str(status)]
    first=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    deadline=time.time()+5
    while not status.exists() and time.time()<deadline:time.sleep(.02)
    assert json.loads(status.read_text())['state']=='running'
    second=subprocess.run(args,capture_output=True,text=True)
    assert second.returncode!=0 and 'BlockingIOError' in second.stderr
    assert first.wait(5)==0 and marker.read_text()=='once'
    assert json.loads(status.read_text())['exit_code']==0

def test_stage_records_child_failure(tmp_path):
    import subprocess,sys
    root=Path(__file__).resolve().parents[1];cmd=tmp_path/'cmd.json';status=tmp_path/'status.json'
    cmd.write_text(json.dumps([sys.executable,'-c','raise SystemExit(7)']))
    p=subprocess.run([sys.executable,str(root/'scripts/robotwin_queue_stage.py'),'--command',str(cmd),'--status',str(status)],capture_output=True)
    assert p.returncode==7 and json.loads(status.read_text())['exit_code']==7

def test_training_stage_recovery_uses_committed_checkpoint():
    from fastwam.robotwin_pipeline import training_recovery_action
    assert training_recovery_action(None,237312,has_records=False)=='fresh'
    assert training_recovery_action(None,237312,has_records=True)=='quarantine'
    assert training_recovery_action(237311,237312,has_records=True)=='resume'
    assert training_recovery_action(237312,237312,has_records=True)=='done'
    assert training_recovery_action(237314,237312,has_records=True)=='done'
    with pytest.raises(ValueError):training_recovery_action(237309,237312,has_records=True)

def test_immutable_parent_hash_rechecked_before_continuation(tmp_path):
    import hashlib
    from fastwam.robotwin_pipeline import verify_file_hash
    p=tmp_path/'parent';p.write_bytes(b'original');sha=hashlib.sha256(b'original').hexdigest()
    verify_file_hash(p,sha)
    p.write_bytes(b'replacement')
    with pytest.raises(ValueError):verify_file_hash(p,sha)

def test_eval_configuration_hash_changes_when_randomization_changes(tmp_path):
    from fastwam.robotwin_pipeline import fingerprint_files
    p=tmp_path/'demo_randomized.yml';p.write_text('random_background: true')
    before=fingerprint_files([p]);p.write_text('random_background: false')
    assert fingerprint_files([p])!=before

def test_stage_resolves_python_after_environment_and_stops_on_source_failure(tmp_path):
    import importlib.util,subprocess,sys
    path=Path(__file__).resolve().parents[1]/'scripts/run_robotwin_posttrain_queue.py'
    spec=importlib.util.spec_from_file_location('queue_script',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    env=tmp_path/'env.sh';env.write_text('return 9\n')
    p=subprocess.run(module.shell(str(env),[sys.executable,'-c','print("SHOULD_NOT_RUN")']),capture_output=True,text=True)
    assert p.returncode==9 and 'SHOULD_NOT_RUN' not in p.stdout
    assert 'exec python ' in module.shell(str(env),[sys.executable,'-c','pass'])[-1]

def test_preflight_binds_runtime_and_requires_both_settings(tmp_path):
    from fastwam.robotwin_pipeline import validate_preflight,fingerprint_files
    env=tmp_path/'env.sh';env.write_text('export VALUE=1\n')
    record=dict(passed=True,simulator_clean_and_randomized=True,runtime_hashes=fingerprint_files([env]))
    validate_preflight(record)
    env.write_text('export VALUE=2\n')
    with pytest.raises(ValueError):validate_preflight(record)
    with pytest.raises(ValueError):validate_preflight(dict(passed=True,simulator_clean_and_randomized=False,runtime_hashes={}))
