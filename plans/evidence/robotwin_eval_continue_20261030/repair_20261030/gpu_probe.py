import os,json,torch
assert torch.cuda.device_count()==8,torch.cuda.device_count()
values=[]
for gpu in range(8):
 x=torch.ones(16,device=f'cuda:{gpu}');values.append(float(x.sum()));torch.cuda.synchronize(gpu)
assert len(os.sched_getaffinity(0))>=112
print(json.dumps(dict(gpus=torch.cuda.device_count(),cpus=len(os.sched_getaffinity(0)),values=values,visible=os.getenv('CUDA_VISIBLE_DEVICES'))),flush=True)
