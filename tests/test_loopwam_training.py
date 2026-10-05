import math
import torch
from scripts.train_loopwam import ExactDistributedBatches, lr_factor


def test_distributed_batches_cover_once_and_pad_only_tail():
    ranks=[list(ExactDistributedBatches(19,2,r,3)) for r in range(3)]
    assert all(len(b)==4 for b in ranks)
    elements=[x for rank in ranks for batch in rank for x in batch]
    assert sorted(x for x in elements if x>=0)==list(range(19))
    assert elements.count(-1)==5
    s=ExactDistributedBatches(19,2,0,3); first=list(s); s.epoch=1
    assert first!=list(s)


def test_accumulated_tail_has_correct_global_gradient():
    # Analytical simulation of DDP rank means and accumulation, N=19, global8.
    n, world, micro, global_batch=19,2,2,8
    order=list(ExactDistributedBatches(n,micro,0,world))
    peers=list(ExactDistributedBatches(n,micro,1,world))
    observed=[]
    for start in range(0,len(order),global_batch//(world*micro)):
        valid=min(global_batch,n-start*world*micro)
        gradient=0.; values=[]
        for idx in range(start,min(start+2,len(order))):
            for batch in (order[idx],peers[idx]):
                vals=[float(x+1) for x in batch if x>=0]
                values+=vals
                gradient+=(sum(vals)/micro)*(world*micro/valid)/world
        assert math.isclose(gradient,sum(values)/len(values))
        observed+=values
    assert len(observed)==n


def test_cosine_schedule():
    assert lr_factor(0,100,5)==.2
    assert lr_factor(4,100,5)==1
    assert lr_factor(5,100,5)==1
    assert math.isclose(lr_factor(99,100,5),.01)


def test_resume_sampler_skips_exact_completed_microbatches():
    sampler=ExactDistributedBatches(37,2,1,2)
    sampler.epoch=3
    complete=list(sampler)
    sampler.start_batch=4
    assert list(sampler)==complete[4:]
    assert len(sampler)==len(complete)
