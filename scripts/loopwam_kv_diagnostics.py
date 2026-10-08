"""Fixed heldout, RNG-neutral all-loop attention diagnostics."""
from contextlib import nullcontext
import random
import numpy as np
import torch
from torch.utils.data import default_collate


@torch.no_grad()
def collect_kv_diagnostics(model,dataset,epoch,seed=42,count=8):
    mode=model.mot.action_kv_mode
    if mode=='aligned':return None
    if mode=='mix':
        weights=model.mot.action_kv_logits.detach().float().softmax(-1)
        return dict(event='kv_diagnostics',mode=mode,epoch=epoch,weights=weights.cpu().tolist())
    if len(dataset)<count:raise ValueError('Concat diagnostics need eight heldout windows')
    indices=torch.linspace(0,len(dataset)-1,count).long().tolist()
    was_training=model.training;old_cache=getattr(model,'training_latent_cache',None)
    old_record=model.mot.record_attention_mass;old_masses=model.mot.last_attention_mass
    old_diagnostics=model.mot.last_diagnostics
    numpy_state=np.random.get_state();python_state=random.getstate()
    device=torch.device(model.device);devices=[device] if device.type=='cuda' else []
    collected={j:[] for j in range(model.mot.core_depth)}
    try:
        model.eval();model.training_latent_cache=None;model.mot.record_attention_mass=True
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
            for index in indices:
                model.mot.last_attention_mass={}
                sample=default_collate([dataset[index]])
                # Native BF16 computation; attention probability diagnostics explicitly recompute in FP32.
                with torch.autocast(device.type,dtype=torch.bfloat16) if device.type=='cuda' else nullcontext():
                    model.forward_exits(model.prepare_training_batch(sample))
                for j,values in model.mot.last_attention_mass.items():collected[j].extend(values)
        masses={str(j):torch.stack(values).mean(0).cpu().tolist() for j,values in collected.items()}
        if any(abs(sum(v)-1)>1e-5 for v in masses.values()):raise ValueError('Attention probability mass does not sum to one')
        return dict(event='kv_diagnostics',mode=mode,epoch=epoch,seed=seed,indices=indices,
                    columns=[f'video_loop_{v}' for v in range(1,model.mot.loops+1)]+['action'],
                    attention_mass=masses,probability_dtype='float32')
    finally:
        model.training_latent_cache=old_cache;model.mot.record_attention_mass=old_record
        model.mot.last_attention_mass=old_masses;model.mot.last_diagnostics=old_diagnostics
        model.train(was_training);np.random.set_state(numpy_state);random.setstate(python_state)
