import gc,json,time
from pathlib import Path
import torch
from fastwam.models.wan22.loopwam_init import load_init_artifact,build_target_experts,stage_donor_indices,sha256_file
from run_centered_partition_grid import GRID
root=Path('/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/partition_centered_20261026')
path='checkpoints/LoopWAM/wan21_compact_donors.pt'
torch.set_num_threads(4)
artifact=load_init_artifact(path);rows=[]
for label,(partition,loops) in GRID.items():
 donors=stage_donor_indices(partition)
 v,a,p=build_target_experts(artifact,donor_indices=donors)
 n=0
 for family,model in [('video',v),('action',a)]:
  for key,tensor in model.state_dict().items():
   source=key
   if key.startswith('blocks.'):
    _,i,suffix=key.split('.',2);source=f'blocks.{donors[int(i)]}.{suffix}'
   if not torch.equal(tensor,artifact[family+'_state_dict'][source]):raise AssertionError((label,family,key,source))
   n+=1
 for key,tensor in p.state_dict().items():
  assert torch.equal(tensor,artifact['proprio_state_dict'][key]);n+=1
 count=sum(t.numel() for model in (v,a,p) for t in model.parameters());assert count==584536135
 rows.append(dict(label=label,partition=list(partition),donors=list(donors),tensors_compared=n,policy_parameters=count,exact_equal=True))
 print(json.dumps(rows[-1]),flush=True)
 del v,a,p;gc.collect()
result=dict(passed=True,artifact_sha256=sha256_file(path),runs=rows,finished=time.time())
(root/'native_donor_audit.json').write_text(json.dumps(result,indent=2))
