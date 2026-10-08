import importlib.util
from pathlib import Path
import pytest

def module():
 p=Path(__file__).resolve().parents[1]/'scripts/run_full_loop_queue.py'
 s=importlib.util.spec_from_file_location('queue_runner',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_training_command_is_fresh_full_data_global128():
 m=module();c=m.train_command((3,3),'ddp',16,Path('fresh'),10)
 assert '--resume' not in c
 for k,v in {'--video-loops':'3','--action-loops':'3','--global-batch':'128','--epochs':'10','--seed':'42','--dataset-scope':'full_libero','--validation-samples':'0','--max-updates':'10'}.items():assert c[c.index(k)+1]==v
 assert '--nproc_per_node=2' in c

def test_final_eval_same_protocol_three_rounds():
 m=module();c=m.eval_command(Path('train'),Path('eval'),4)
 assert c[c.index('--seeds')+1:c.index('--seeds')+4]==['42']*3
 assert '--smoke' not in c and '--max-steps' not in c

def test_select_only_valid_fastest_trials():
 m=module();base=dict(global_batch=128,world_size=2,train_windows=277713,dataset_scope='full_libero',precision={'policy_dtype':'float32','optimizer_moment_dtypes':['torch.float32']},records=[dict(loss=1.,grad_norm=1.)])
 valid=dict(base,steady_seconds=3.,backend='ddp',microbatch=8)
 assert m.select_training([valid,dict(valid,steady_seconds=2.,microbatch=16)])['microbatch']==16
 with pytest.raises(ValueError):m.select_training([dict(valid,global_batch=64)])
 with pytest.raises(ValueError):m.select_training([dict(valid,records=[dict(loss=float('nan'),grad_norm=1.)])])

def test_failure_stops_before_evaluation_and_second_training(tmp_path,monkeypatch):
 import json
 from types import SimpleNamespace
 m=module();q=m.Queue.__new__(m.Queue);q.out=tmp_path;q.a=SimpleNamespace(pairs=[[4,1],[1,4]]);q.state={'source_revision':'abc'}
 (tmp_path/'release.json').write_text(json.dumps({'source_revision':'abc','approved_pairs':q.a.pairs}))
 seen=[]
 def fail(name,command):
  seen.append(name);raise RuntimeError('failed training')
 q.stage=fail
 prepared={'configs':{'41':{'backend':'ddp','microbatch':8,'workers_per_gpu':2},'14':{'backend':'ddp','microbatch':16,'workers_per_gpu':2}}}
 with pytest.raises(RuntimeError,match='failed training'):q.production(prepared)
 assert seen==['train_41']

def test_final_training_gate_rejects_incomplete_or_wrong_run(tmp_path):
 import json
 m=module()
 manifest=dict(resume=None,initialization_mode='canonical_wan_artifact_fresh_optimizer',version='v0',loops=3,action_core_loops=3,epochs=10,global_batch=128,world_size=2,microbatch=16,gradient_accumulation=4,policy_parameters=584536135,train_windows=277713,val_windows=0,planned_updates=21700,planned_windows=2777130,dataset_scope='full_libero',max_updates=None,data=dict(available_episodes=1712,task_counts={str(i):1 for i in range(40)},split='all_train',suites=list(m.SUITES)))
 timing=dict(status='complete',completed_updates=21700,windows_seen=2777130)
 state=dict(update=21700,epoch=9,next_micro=8680)
 records={'manifest':manifest,'timing':timing,'trainer_state':state}
 for name,value in records.items():(tmp_path/(name+'.json')).write_text(json.dumps(value))
 (tmp_path/'latest.pt').write_bytes(b'checkpoint')
 with pytest.raises(ValueError,match='Epoch coverage'):m.verify_train(tmp_path,[3,3])
 state['next_micro']=8679;(tmp_path/'trainer_state.json').write_text(json.dumps(state))
 assert m.verify_train(tmp_path,[3,3])['status']=='complete'
 for name,key,value in [('manifest','resume','old.pt'),('manifest','action_core_loops',4),('manifest','max_updates',21700),('timing','windows_seen',1)]:
  original=records[name][key];records[name][key]=value;(tmp_path/(name+'.json')).write_text(json.dumps(records[name]))
  with pytest.raises(ValueError):m.verify_train(tmp_path,[3,3])
  records[name][key]=original;(tmp_path/(name+'.json')).write_text(json.dumps(records[name]))
