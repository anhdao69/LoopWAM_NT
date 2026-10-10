"""Trainable action policy, with a shared frozen video parent and frozen teacher."""
from __future__ import annotations
import copy
import torch
from torch import nn
from .loop_mot import LoopMoT
from .loopwam import masked_action_loss
from .wan_video_dit import sinusoidal_embedding_1d
from .residual_transport import (ResidualTransport, StepCondition, TransportHistory,
    install_core_lora, merge_lora, cache_text_kv, action_core, teacher_target,
    detach_boundary, endpoint_losses)

class TransportPolicy(nn.Module):
    def __init__(self,parent,*,kind="T4",scope="pb",step_cond=False,regime="fu",
                 anchor=True,checkpoint_blocks=True,loss_weights=None):
        super().__init__()
        if scope not in {"pa","pb","pc"} or regime not in {"fu","tf"}:
            raise ValueError("Invalid RT parameter scope or training regime")
        self.parent=parent.requires_grad_(False).eval()
        self.scope,self.regime,self.anchor=scope,regime,anchor and scope!="pa"
        self.checkpoint_blocks=checkpoint_blocks
        self.loss_weights=loss_weights or dict(local=1.,end=.5,grip=.1,anchor=.5)
        action=copy.deepcopy(parent.action_expert)
        width=action.hidden_dim
        if scope=="pb": install_core_lora(action)
        if scope=="pc": action.requires_grad_(True)
        # Initialize paired-control adapters before variant-specific modules consume RNG.
        self.transport=ResidualTransport(kind,width).to(parent.device)
        self.step_condition=StepCondition(width).to(parent.device) if step_cond else None
        # Register a shared frozen video expert; teacher action remains independent.
        self.student=LoopMoT({"video":parent.video_expert,"action":action},loops=4,version="v0")
        self.delta_encoder=copy.deepcopy(parent.action_expert.action_encoder).requires_grad_(False)
        if not any(p.requires_grad for p in self.parameters()):
            raise ValueError("This scope/transport has no trainable parameters")
        self.teacher_passes=0
        self.student_block_calls=0

    def train(self,mode=True):
        super().train(mode)
        self.parent.eval()
        self.delta_encoder.eval()
        self.student.mixtures["video"].eval()
        return self

    def optimizer_groups(self):
        adapter=[]; expert=[]; transport=[]
        for name,p in self.named_parameters():
            if not p.requires_grad: continue
            if name.startswith(("transport.","step_condition.")): transport.append(p)
            elif self.scope=="pc": expert.append(p)
            else: adapter.append(p)
        groups=[]
        for ps,lr,decay in ((transport,1e-3,0.),(adapter,2e-4,0.),(expert,3e-5,.01)):
            if ps: groups.append(dict(params=ps,lr=lr,base_lr=lr,weight_decay=decay))
        return groups

    @torch.no_grad()
    def prefill(self,sample):
        context,mask=self.parent._append_proprio_to_context(sample["context"],sample["context_mask"],
                                                            sample["proprio"][:,0])
        latents=sample["first_frame_latents"]
        video=self.parent.video_expert.prepare(x=latents,timestep=latents.new_zeros(len(latents)),
            context=context,context_mask=mask,action=None,
            fuse_vae_embedding_in_latents=bool(self.parent.video_expert.fuse_vae_embedding_in_latents))
        n=video[0].shape[1]
        attention=self.parent._build_mot_attention_mask(n,32,video[9],latents.device)
        keys,values=self.parent.mot.prefill_video_cache_tensor(video[0],video[5],video[2],
            video[3],video[4],attention[:n,:n])
        return context,mask,keys,values,attention[n:,:]

    @staticmethod
    def conditioning(expert,context,mask):
        embedded=expert.text_embedding(context)
        return embedded,mask[:,None,:].expand(-1,32,-1),cache_text_kv(expert,embedded)

    def velocity(self,z,tau,conditioning,cache,*,teacher=False,loops=4,d=0.,history=None,native_timestep=None):
        mot=self.parent.mot if teacher else self.student
        expert=mot.mixtures["action"]
        context,mask,text_kv=conditioning
        native_timestep=tau*1000 if native_timestep is None else native_timestep
        t=expert.time_embedding(sinusoidal_embedding_1d(expert.freq_dim,native_timestep))
        tmod=expert.time_projection(t).unflatten(1,(6,expert.hidden_dim))
        if not teacher and self.step_condition is not None:
            tmod=tmod+self.step_condition(tau.new_full(tau.shape,d))
        tokens=expert.action_encoder(z)
        transport=self.transport
        output,residual=action_core(mot,tokens,expert.get_freqs(z.shape[1]),tmod,context,mask,
            *cache,loops=loops,transport=transport,history=None if teacher else history,next_tau=tau,
            text_kv=text_kv,checkpoint_blocks=self.checkpoint_blocks and not teacher)
        if teacher: self.teacher_passes+=1
        else: self.student_block_calls+=6+6*loops
        return expert.post(output),residual

    def forward(self,sample,schedule,noise,anchor_noise,anchor_tau):
        if self.regime=="tf" and len(schedule)!=10:
            raise ValueError("Teacher forcing is defined only on the NFE10 grid")
        context,mask,keys,values,attention=self.prefill(sample)
        cache=(keys,values,attention)
        with torch.no_grad():
            teacher_condition=self.conditioning(self.parent.action_expert,context,mask)
        student_condition=self.conditioning(self.student.mixtures["action"],context,mask)
        def teacher(z,t):
            return self.velocity(z,t,teacher_condition,cache,teacher=True)[0]
        # One deployed parent trajectory, shared endpoint and TF states.
        teacher_states=[]; endpoint=noise.detach()
        grid=torch.linspace(1.,0.,11,device=noise.device,dtype=noise.dtype)
        with torch.no_grad():
            for j in range(10):
                teacher_states.append(endpoint)
                endpoint=endpoint+(grid[j+1]-grid[j])*teacher(endpoint,grid[j].expand(len(endpoint)))
        n=len(schedule); z=noise; history=None; local=z.new_zeros(len(z))
        for j,k in enumerate(schedule):
            if j<=n-3: z,history=detach_boundary(z,history)
            tau=1-j/n; next_tau=1-(j+1)/n
            start=round(j*10/n); stop=round((j+1)*10/n)
            current_tau=grid[start].expand(len(z))
            source=teacher_states[j] if self.regime=="tf" else z
            target=teacher_target(teacher,source,tau,next_tau)
            v,r=self.velocity(source,current_tau,student_condition,cache,
                              loops=k,d=1/n,history=history)
            local=local+masked_action_loss(v,target,sample["action_is_pad"])/n
            delta=(grid[stop]-grid[start])*v
            new_history=TransportHistory(r,None if history is None else history.residual,
                None if history is None else history.tau,current_tau,
                self.delta_encoder(delta) if self.transport.kind=="T5" else None)
            z=z+delta
            history=new_history
        end,grip=endpoint_losses(z,endpoint,sample["action_scale"],sample["action_offset"])
        anchor=z.new_zeros(len(z))
        if self.anchor:
            clean=sample["action"]
            noisy=anchor_tau[:,None,None]*anchor_noise+(1-anchor_tau[:,None,None])*clean
            v,_=self.velocity(noisy,anchor_tau,student_condition,cache,loops=4,d=0.)
            anchor=masked_action_loss(v,anchor_noise-clean,sample["action_is_pad"])
            anchor=anchor*self.parent.train_action_scheduler.training_weight(anchor_tau*1000)
        components=dict(local=local,end=end,grip=grip,anchor=anchor)
        loss=sum(self.loss_weights[key]*value for key,value in components.items())
        # Fixed masking avoids boolean-index graphs and keeps dummy DDP ranks participating.
        valid=sample["sample_valid"].to(loss)
        return (loss*valid).sum(),{key:(value.detach()*valid).sum() for key,value in components.items()}

    @torch.no_grad()
    def rollout(self,sample,schedule,noise):
        context,mask,keys,values,attention=self.prefill(sample)
        condition=self.conditioning(self.student.mixtures["action"],context,mask)
        z=noise; history=None; n=len(schedule)
        times,deltas=self.parent.infer_action_scheduler.build_inference_schedule(n,z.device,z.dtype,shift_override=1.)
        for j,k in enumerate(schedule):
            tau=(times[j]/1000).expand(len(z))
            v,r=self.velocity(z,tau,condition,(keys,values,attention),loops=k,d=1/n,history=history,native_timestep=times[j].expand(len(z)))
            delta=v*deltas[j]
            history=TransportHistory(r,None if history is None else history.residual,
                None if history is None else history.tau,tau,
                self.delta_encoder(delta) if self.transport.kind=="T5" else None)
            z=z+delta
        return z
