"""All-loop KV numerical contracts using the production tiny Wan/Action blocks."""
import copy
import math
import subprocess
import types
from pathlib import Path
import pytest
import torch
from test_loop_mot import make_model, inputs, assert_pair_close
from fastwam.models.wan22.loop_mot import LoopMoT


def model(mode='aligned', video=4, action=1, checkpoint=False):
    base=make_model()
    return LoopMoT(dict(base.mixtures.items()), loops=video, action_loops=action,
                   action_kv_mode=mode, checkpoint_blocks=checkpoint)


def cache(m,d):
    k,v=m.prefill_video_cache_tensor(d['video_tokens'][:,:2],d['video_freqs'][:2],
        d['video_t_mod'][:,:2],d['video_context'],d['video_context_mask'][:,:2],d['attention_mask'][:2,:2])
    mask=torch.cat((d['attention_mask'][4:,:2],d['attention_mask'][4:,4:]),dim=1)
    return k,v,mask


def cached(m,d):
    k,v,mask=cache(m,d)
    return m.forward_action_with_video_cache_tensor(d['action_tokens'],d['action_freqs'],
        d['action_t_mod'],d['action_context'],d['action_context_mask'],k,v,mask)


def grads_equal(a,b,atol=5e-5,rtol=5e-4):
    aa=dict(a.named_parameters());bb=dict(b.named_parameters());assert aa.keys()==bb.keys()
    for name,p in aa.items():
        q=bb[name]
        torch.testing.assert_close(p.grad if p.grad is not None else torch.zeros_like(p),
            q.grad if q.grad is not None else torch.zeros_like(q),atol=atol,rtol=rtol,msg=name)


@pytest.mark.parametrize('structured',[False,True])
@pytest.mark.parametrize('checkpoint',[False,True])
def test_generic_aligned_training_matches_original_joint_outputs_and_all_gradients(structured,checkpoint):
    a=model(checkpoint=checkpoint);b=copy.deepcopy(a);d=inputs()
    for m in (a,b):m.structured_attention=structured;m.structured_attention_observation_tokens=2
    y=a.forward_joint_core(**d)
    z=b.forward_cached_training(**d)
    assert_pair_close(y,z)
    sum(x.square().mean() for x in y).backward();sum(x.square().mean() for x in z).backward()
    grads_equal(a,b)


@pytest.mark.parametrize('mode',['concat','mix'])
@pytest.mark.parametrize('video,action',[(4,1),(4,2),(2,4),(1,4)])
@pytest.mark.parametrize('checkpoint',[False,True])
def test_train_inference_outputs_and_parameter_gradients(mode,video,action,checkpoint):
    a=model(mode,video,action,checkpoint);b=copy.deepcopy(a);d=inputs()
    y=a.forward_joint_core(**d)[1];z=cached(b,d)
    torch.testing.assert_close(y,z,atol=3e-5,rtol=3e-5)
    y.square().mean().backward();z.square().mean().backward();grads_equal(a,b)


def plain_attention(q,k,v,mask,heads):
    b,n,width=q.shape;dh=width//heads
    q=q.float().reshape(b,n,heads,dh).transpose(1,2)
    k=k.float().reshape(b,-1,heads,dh).transpose(1,2)
    v=v.float().reshape(b,-1,heads,dh).transpose(1,2)
    scores=(q@k.transpose(-1,-2))/math.sqrt(dh)
    probabilities=scores.masked_fill(~mask,float('-inf')).softmax(-1)
    return (probabilities@v).transpose(1,2).reshape(b,n,width)


@pytest.mark.parametrize('action_loops',[1,2,4])
def test_concat_matches_independent_naive_fp32_softmax(action_loops):
    m=model('concat',action=action_loops);d=inputs();keys,values,mask=cache(m,d)
    x=d['action_tokens'];schedule=m.virtual_schedule();slot_lookup={key:i for i,(key,_) in enumerate(schedule)}
    indices=list(range(3))+list(range(3,9))*action_loops+list(range(9,12))
    for index in indices:
        block=m.mixtures['action'].blocks[index]
        io=m._build_expert_attention_io(m.mixtures['action'],block,x,d['action_freqs'],d['action_t_mod'])
        if 3<=index<9:
            slots=[slot_lookup[('core',v,index-3)] for v in range(1,5)]
        elif index<3:slots=[slot_lookup[('pre',index)]]
        else:slots=[slot_lookup[('coda',4,index-9)]]
        k=torch.cat([keys[s] for s in slots]+[io[1]],dim=1)
        v=torch.cat([values[s] for s in slots]+[io[2]],dim=1)
        expanded=torch.cat([mask[:,:2]]*len(slots)+[mask[:,2:]],dim=1)
        assert expanded.shape==(3,2*len(slots)+3)
        mixed=plain_attention(io[0],k,v,expanded,m.num_heads)
        x=m._post(block,io,mixed,d['action_context'],d['action_context_mask'])
    torch.testing.assert_close(cached(m,d),x,atol=3e-5,rtol=3e-5)


@pytest.mark.parametrize('mode',['concat','mix'])
def test_no_future_or_action_leakage_and_expanded_mask(mode):
    m=model(mode);d=inputs();original=m.forward_joint_core(**d)
    changed=dict(d,video_tokens=d['video_tokens'].clone());changed['video_tokens'][:,2:]+=50
    torch.testing.assert_close(m.forward_joint_core(**changed)[1],original[1],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(cached(m,changed),cached(m,d),atol=1e-6,rtol=1e-6)
    changed=dict(d,action_tokens=d['action_tokens']+50)
    torch.testing.assert_close(m.forward_joint_core(**changed)[0],original[0],atol=1e-6,rtol=1e-6)
    k,v,mask=cache(m,d);slots=m.action_cache_slot_sets()[3]
    kk,vv,expanded=m.action_conditioning_kv(k,v,slots,3,mask)
    assert kk.shape[1]==(8 if mode=='concat' else 2)
    assert expanded.shape==(3,kk.shape[1]+3)
    assert expanded.all() # Only observation-prefix and action keys are present.
    bad=dict(d,attention_mask=d['attention_mask'].clone());bad['attention_mask'][0,2]=True
    with pytest.raises((RuntimeError,ValueError),match='future|observation'):
        m.forward_joint_core(**bad)


@pytest.mark.parametrize('mode',['aligned','concat'])
def test_direct_action_kv_gradients_include_exact_requested_loops(mode):
    m=model(mode);saved={}
    def observer(keys,values):
        saved.update(keys=keys,values=values)
        for tensor in keys+values:tensor.retain_grad()
    _,a=m.forward_cached_training(**inputs(),cache_observer=observer)
    a.square().mean().backward()
    schedule=m.virtual_schedule()
    for slot,(key,_) in enumerate(schedule):
        if key[0]!='core':continue
        for values in saved.values():
            gradient=values[slot].grad
            if mode=='concat' or key[1]==4:
                assert gradient is not None and gradient.abs().sum()>0,(mode,key)
            else:assert gradient is None or gradient.count_nonzero()==0,(mode,key)


def test_mix_uniform_onehot_and_logit_gradients():
    m=model('mix');aligned=model();d=inputs()
    torch.testing.assert_close(m.action_kv_logits.softmax(-1),torch.full((6,2,4),.25),atol=0,rtol=0)
    assert sum(p.numel() for p in m.parameters())-sum(p.numel() for p in aligned.parameters())==48
    y=m.forward_joint_core(**d)[1];y.square().mean().backward()
    assert m.action_kv_logits.grad is not None and m.action_kv_logits.grad.abs().sum()>0
    with torch.no_grad():m.action_kv_logits.fill_(-100);m.action_kv_logits[:,:,3]=100
    assert_pair_close(m.forward_joint_core(**d),aligned.forward_joint_core(**d))


@pytest.mark.parametrize('mode',['bad',None,True])
def test_invalid_modes_fail(mode):
    with pytest.raises(ValueError,match='action_kv_mode'):model(mode)


@pytest.mark.parametrize('version',['v1','v2','dense_s12'])
@pytest.mark.parametrize('mode',['concat','mix'])
def test_non_v0_new_modes_fail(version,mode):
    base=make_model(version,loops=1 if version=='dense_s12' else 4)
    with pytest.raises(ValueError,match='v0'):
        LoopMoT(dict(base.mixtures.items()),version=version,action_kv_mode=mode)


@pytest.mark.parametrize('version,video,action',[('v0',4,4),('v0',4,1),('v0',4,2),('v0',2,2),('v0',1,4),('v1',4,4),('v2',4,4),('dense_s12',1,1),('dense_s30',1,1)])
def test_aligned_bit_identical_to_preregistered_base(version,video,action):
    source=subprocess.check_output(['git','show','8d74c8d:src/fastwam/models/wan22/loop_mot.py'],text=True,cwd=Path(__file__).resolve().parents[1])
    module=types.ModuleType('fastwam.models.wan22._kv_baseline');module.__package__='fastwam.models.wan22';exec(compile(source,'baseline_loop_mot.py','exec'),module.__dict__)
    from test_loopwam_dense_s30 import dense_mot
    new=dense_mot() if version=='dense_s30' else make_model(version,loops=video)
    new.action_loops=action
    old=module.LoopMoT(copy.deepcopy(dict(new.mixtures.items())),loops=video,version=version,action_loops=action)
    y=new.forward_joint_core(**inputs());z=old.forward_joint_core(**inputs())
    for a,b in zip(y,z):assert torch.equal(a,b)
    sum(x.square().mean() for x in y).backward();sum(x.square().mean() for x in z).backward();grads_equal(new,old,0,0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA compiled inference check')
@pytest.mark.parametrize('mode',['concat','mix'])
def test_gpu_fullgraph_compiled_cache_inference(mode):
    m=model(mode).cuda().eval();d={k:v.cuda() for k,v in inputs().items()}
    with torch.no_grad():
        k,v,mask=cache(m,d)
        args=(d['action_tokens'],d['action_freqs'],d['action_t_mod'],d['action_context'],d['action_context_mask'],k,v,mask)
        expected=m.forward_action_with_video_cache_tensor(*args).clone()
        compiled=torch.compile(m.forward_action_with_video_cache_tensor,fullgraph=True,mode='reduce-overhead')
        actual=compiled(*args)
        torch.testing.assert_close(actual,expected,atol=3e-5,rtol=3e-5)


def test_concat_attention_mass_is_probability_partition():
    m=model('concat');m.record_attention_mass=True
    with torch.no_grad():m.forward_joint_core(**inputs())
    assert set(m.last_attention_mass)==set(range(6))
    for values in m.last_attention_mass.values():
        for mass in values:
            assert mass.shape==(5,) and bool((mass>=0).all())
            torch.testing.assert_close(mass.sum(),torch.tensor(1.0),atol=1e-6,rtol=1e-6)
