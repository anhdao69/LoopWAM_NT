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
