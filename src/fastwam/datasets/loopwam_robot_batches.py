"""Exact real-window batches: epoch-local shuffle, no duplication or stream padding."""
import math
import torch

def training_budget(size,global_batch,epochs):
    updates=math.ceil(size/global_batch)
    return updates,updates*epochs,size%global_batch or global_batch

def epoch_rank_batches(size,global_batch,world,rank,seed,start_batch=0):
    if min(size,global_batch,world)<=0 or not 0<=rank<world:raise ValueError('Invalid distributed batch shape')
    order=torch.randperm(size,generator=torch.Generator().manual_seed(seed)).numpy()
    for start in range(start_batch*global_batch,size,global_batch):
        chunk=order[start:start+global_batch];q,r=divmod(len(chunk),world)
        lo=rank*q+min(rank,r);count=q+int(rank<r)
        yield chunk[lo:lo+count].tolist()
