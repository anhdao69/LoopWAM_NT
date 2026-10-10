"""Residual Transport primitives. No changes to the pretrained parent graph."""
from __future__ import annotations
from dataclasses import dataclass, fields
from functools import partial
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .wan_video_dit import sinusoidal_embedding_1d, flash_attention, modulate

SCHEDULES = {
    "parent10": (4,)*10, "parent20": (4,)*20, "parent5": (4,)*5,
    "parent4": (4,)*4, "parent2": (4,4), "parent1": (4,),
    "S1": (4,)+(2,)*9, "S2": (4,)+(1,)*9,
    "S3": (4,3,2,2,2,1,1,1,1,1), "S4": (4,4)+(1,)*8,
    "S5": (4,2,2,2,2), "S6": (4,1,1,1,1), "S7": (4,2), "S8": (4,1),
}

@dataclass
class TransportHistory:
    residual: torch.Tensor
    previous_residual: torch.Tensor | None
    previous_tau: torch.Tensor | None
    tau: torch.Tensor
    delta_embedding: torch.Tensor | None

def detach_boundary(z, history):
    if history is None:
        return z.detach(), None
    return z.detach(), TransportHistory(**{
        field.name: (getattr(history,field.name).detach() if getattr(history,field.name) is not None else None)
        for field in fields(history)})

class ResidualTransport(nn.Module):
    def __init__(self, kind="T4", width=512, rank=32, freq_dim=32):
        super().__init__()
        if kind not in {f"T{i}" for i in range(7)}:
            raise ValueError("Transport must be T0 through T6")
        self.kind, self.freq_dim = kind, freq_dim
        if kind in {"T3","T4","T5"}:
            self.gate = nn.Sequential(nn.Linear(2*freq_dim,64),nn.SiLU(),nn.Linear(64,width))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, math.log(19))
        if kind in {"T4","T5"}:
            self.A=nn.Linear(width,rank,bias=False)
            self.B=nn.Linear(rank,width,bias=False)
            nn.init.zeros_(self.B.weight)
            self.film=nn.Sequential(nn.Linear(2*freq_dim+(width if kind=="T5" else 0),64),
                                    nn.SiLU(),nn.Linear(64,rank))
        if kind=="T6":
            self.attn=nn.MultiheadAttention(width,8 if width%8==0 else 1,batch_first=True)
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, p, history, next_tau):
        if self.kind=="T0" or history is None:
            return torch.zeros_like(p)
        r=history.residual
        if self.kind=="T1": return r
        if self.kind=="T2":
            if history.previous_residual is None: return r
            rho=(next_tau-history.tau)/(history.tau-history.previous_tau)
            return r+rho[:,None,None]*(r-history.previous_residual)
        if self.kind=="T6":
            return r+self.attn(p,r,r,need_weights=False)[0]
        features=torch.cat([sinusoidal_embedding_1d(self.freq_dim,t)
                            for t in (history.tau,next_tau)],-1)
        gate=self.gate(features).sigmoid()[:,None,:]
        if self.kind in {"T4","T5"}:
            f=features[:,None,:].expand(-1,r.shape[1],-1)
            if self.kind=="T5":
                if history.delta_embedding is None:
                    raise ValueError("T5 needs the frozen embedding of the actual Euler delta")
                f=torch.cat((f,history.delta_embedding),-1)
            r=r+self.B(self.A(r)*(1+self.film(f)))
        return gate*r

class StepCondition(nn.Module):
    def __init__(self,width=512,freq_dim=32):
        super().__init__()
        self.width,self.freq_dim=width,freq_dim
        self.net=nn.Sequential(nn.Linear(freq_dim,128),nn.SiLU(),nn.Linear(128,6*width))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,d):
        return self.net(sinusoidal_embedding_1d(self.freq_dim,d)).unflatten(-1,(6,self.width))

class LoRALinear(nn.Module):
    def __init__(self,base,rank=16,alpha=16):
        super().__init__()
        self.base=base.requires_grad_(False)
        self.A=nn.Linear(base.in_features,rank,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        self.B=nn.Linear(rank,base.out_features,bias=False,device=base.weight.device,dtype=base.weight.dtype)
        nn.init.zeros_(self.B.weight)
        self.scale=alpha/rank
    @property
    def weight(self): return self.base.weight
    @property
    def bias(self): return self.base.bias
    def forward(self,x): return self.base(x)+self.B(self.A(x))*self.scale
    def merged(self):
        layer=copy.deepcopy(self.base)
        with torch.no_grad(): layer.weight.add_(self.B.weight@self.A.weight,alpha=self.scale)
        return layer

def install_core_lora(expert,rank=16,alpha=16):
    for block in expert.blocks[3:9]:
        for attn in (block.self_attn,block.cross_attn):
            for name in ("q","k","v","o"):
                setattr(attn,name,LoRALinear(getattr(attn,name),rank,alpha))
        for i,layer in enumerate(block.ffn):
            if isinstance(layer,nn.Linear): block.ffn[i]=LoRALinear(layer,rank,alpha)

def merge_lora(module):
    for name,child in list(module.named_children()):
        if isinstance(child,LoRALinear): setattr(module,name,child.merged())
        else: merge_lora(child)
    return module

def cache_text_kv(expert,context):
    return [(b.cross_attn.norm_k(b.cross_attn.k(context)),b.cross_attn.v(context))
            for b in expert.blocks]

def _block(mot,index,slot,x,freqs,tmod,context,context_mask,keys,values,mask,text_kv):
    meter=getattr(mot,"rt_measurement",None)
    if meter is not None: meter["forward"]+=1
    block=mot.mixtures["action"].blocks[index]
    io=mot._build_expert_attention_io(mot.mixtures["action"],block,x,freqs,tmod)
    mixed=mot._mixed_attention(io[0],torch.cat((keys[slot],io[1]),1),
                              torch.cat((values[slot],io[2]),1),mask)
    if text_kv is None: return mot._post(block,io,mixed,context,context_mask)
    x=block.gate(io[3],io[4],block.self_attn.o(mixed))
    ca=block.cross_attn
    q=ca.norm_q(ca.q(block.norm3(x)))
    k,v=text_kv[index]
    x=x+ca.o(flash_attention(q,k,v,ca.num_heads,ctx_mask=context_mask.unsqueeze(1)))
    return block.gate(x,io[7],block.ffn(modulate(block.norm2(x),io[5],io[6])))

def action_core(mot,tokens,freqs,tmod,context,context_mask,keys,values,mask,*,
                loops,transport,history,next_tau,text_kv=None,trace=None,checkpoint_blocks=False):
    if mot.version!="v0" or mot.loops!=4 or not 1<=loops<=4:
        raise ValueError("RT requires the v0 four-loop video parent and 1..4 action loops")
    if len(keys)!=30 or len(values)!=30:
        raise ValueError("RT needs every parent video-cache slot")
    slots=list(range(3))+[3+6*(4-loops+r)+j for r in range(loops) for j in range(6)]+list(range(27,30))
    schedule=mot.virtual_schedule()
    x=tokens; p=None; residual=None
    for n,slot in enumerate(slots):
        if n==3:
            p=x
            if loops<4 and history is not None and transport.kind!="T0":
                x=p+transport(p,history,next_tau)
        index=schedule[slot][1]
        if trace is not None: trace.append((index,slot))
        fn=partial(_block,mot,index,slot)
        args=(x,freqs,tmod,context,context_mask,keys,values,mask,text_kv)
        x=checkpoint(fn,*args,use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled() else fn(*args)
        meter=getattr(mot,"rt_measurement",None)
        if meter is not None and x.requires_grad:
            def measured_backward(gradient,counter=meter):
                counter["backward"]+=1
                return gradient
            x.register_hook(measured_backward)
        if n==3+6*loops-1: residual=x-p
    return x,residual

@torch.no_grad()
def teacher_target(velocity,z,tau,next_tau):
    steps=round((tau-next_tau)*10)
    if steps<1 or abs((tau-next_tau)*10-steps)>1e-6 or abs(tau*10-round(tau*10))>1e-6:
        raise ValueError("Teacher targets must lie on the deployed 0.1 grid")
    x=z.detach()
    grid=torch.linspace(1.,0.,11,device=z.device,dtype=z.dtype)
    start=round((1-tau)*10)
    # Return the velocity directly at NFE10 (avoids subtractive cancellation).
    if steps==1:
        return velocity(x,grid[start].expand(len(x))).detach()
    for i in range(steps):
        x=x+(grid[start+i+1]-grid[start+i])*velocity(x,grid[start+i].expand(len(x)))
    return ((x-z.detach())/(grid[start+steps]-grid[start])).detach()

def gripper_command(actions,scale,offset):
    """Continuous LIBERO command; positive closes, negative opens."""
    raw=(actions[...,6]-offset[6])/scale[6]
    return 1-2*raw

def endpoint_losses(student,teacher,scale,offset):
    end=(student[:,:10].float()-teacher[:,:10].float()).square().mean((1,2))
    target=gripper_command(teacher[:,:10],scale,offset)
    sign=torch.where(target>=0,1.,-1.)  # deployed raw > .5 opens; equality closes
    grip=F.softplus(-5*sign*gripper_command(student[:,:10],scale,offset)).mean(1)
    return end,grip
