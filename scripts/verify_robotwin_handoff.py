"""Verify epoch5 policy and Adam checkpoint form the same immutable continuation parent."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from fastwam.models.wan22.loopwam_robot import validate_robot_payload
from fastwam.robotwin_pipeline import continuation_contract,validate_parent_state

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify(weights,recovery):
    a=torch.load(weights,map_location='cpu',mmap=True,weights_only=False)
    b=torch.load(recovery,map_location='cpu',mmap=True,weights_only=False)
    validate_robot_payload(a);validate_robot_payload(b)
    for key in ['step','architecture','training_state']:
        # elapsed time differs between consecutive weight and optimizer saves.
        if key=='training_state':
            for field in ['contract','update','windows_seen','next_epoch','next_batch']:
                if a[key][field]!=b[key][field]:raise ValueError(f'Checkpoint{field}mismatch')
        elif a[key]!=b[key]:raise ValueError(f'Checkpoint{key}mismatch')
    for family in ['mot','proprio_encoder']:
        if a[family].keys()!=b[family].keys():raise ValueError('Checkpoint keys mismatch')
        for k,x in a[family].items():
            y=b[family][k]
            if x.shape!=y.shape or x.dtype!=y.dtype or not torch.equal(x,y):raise ValueError(f'Checkpoint tensor mismatch:{family}/{k}')
    validate_parent_state(b,continuation_contract(b['training_state']['contract'],sha(recovery)))
    opt=b['optimizer'];ids=[v for g in opt['param_groups'] for v in g['params']]
    if len(set(ids))!=len(ids) or set(ids)!=set(opt['state']):raise ValueError('Optimizer parameter coverage mismatch')
    for state in opt['state'].values():
        if float(state['step'])!=237310:raise ValueError('Optimizer step differs from epoch5')
        for k in ['exp_avg','exp_avg_sq']:
            if state[k].dtype!=torch.float32 or not torch.isfinite(state[k]).all():raise ValueError('Invalid optimizer moments')
    return dict(passed=True,step=237310,weights=str(weights),weights_sha256=sha(weights),recovery=str(recovery),
        recovery_sha256=sha(recovery),optimizer_parameters=len(ids),weights_equal=True,optimizer_valid=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--recovery',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(4);result=verify(a.weights,a.recovery)
    path=Path(a.output);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
    print(json.dumps(result))
