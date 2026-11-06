"""Compare native and CUDA graph inference on captured real simulator inputs."""
import json,time,hashlib
from pathlib import Path
import torch
from fastwam.models.wan22.loopwam_robot import create_loopwam_robot
from experiments.robotwin.loopwam_policy.cuda_graph import enable_graphs
B=Path('/data14t/anhdh35/loopwam_robotwin10_v0_4x1_local_20261105');torch.set_num_threads(2)
paths=sorted((B/'gate_inputs').glob('*.pt'))
if len(paths)<4:raise ValueError('Need multiple real observation inputs before graph gate')
model,_=create_loopwam_robot('/storage/anhdh35/Light-WAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth',checkpoint_path=str(B/'checkpoint/policy.pt'),device='cuda',checkpoint_blocks=False);model.eval()
inputs=[torch.load(p,map_location='cpu',weights_only=False) for p in paths[:8]]
for i,kw in enumerate(inputs):kw['seed']=42+i%3

def predict(kw):
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):a=model.infer_action(**kw)['action'].float().cpu().clone()
 assert a.shape==(32,14) and torch.isfinite(a).all()
 return a
predict(inputs[0]);start=time.perf_counter();native=[predict(kw) for kw in inputs];native_s=(time.perf_counter()-start)/len(inputs)
report=dict(checkpoint_sha256='b0fd9da51d9ff5ab512d5729de6565e85d240a10f078ddf9f803733a77aa757d',nfe=10,atol=0,rtol=0,inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths[:8]],native_query_seconds=native_s,graph_passed=False)
try:
 enable_graphs(model)
 for i in list(range(len(inputs)))+list(reversed(range(len(inputs)))):torch.testing.assert_close(predict(inputs[i]),native[i],atol=0,rtol=0)
 start=time.perf_counter()
 for kw in inputs:predict(kw)
 report.update(graph_passed=True,max_abs_difference=0,graph_query_seconds=(time.perf_counter()-start)/len(inputs))
except Exception as e:report['error']=repr(e)
report['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
(B/'graph_gate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
