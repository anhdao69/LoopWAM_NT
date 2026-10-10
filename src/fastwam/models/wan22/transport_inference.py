"""LIBERO-compatible inference and one reusable CUDA graph per depth schedule."""
from __future__ import annotations
import torch
from .loopwam import create_loopwam
from .transport_policy import TransportPolicy
from .residual_transport import SCHEDULES

class TransportInference:
    def __init__(self,checkpoint,vae_path,device="cuda:0",schedule="S2",zero_shot_kind=None):
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False,mmap=True)
        config=payload.get("transport_config")
        if config is None and zero_shot_kind is None:
            raise ValueError("Parent checkpoint requires explicit zero_shot_kind=T0/T1/T2")
        self.parent=create_loopwam(checkpoint_path=checkpoint,vae_path=vae_path,device=device,allow_transport_backbone=True)
        config=config or dict(kind=zero_shot_kind,step_cond=False)
        # Preserve the trained BF16 operation order; merged LoRA has different rounding.
        scope=config["scope"] if "student_action" in payload else "pc"
        self.policy=TransportPolicy(self.parent,kind=config["kind"],scope=scope,
            step_cond=config["step_cond"],checkpoint_blocks=False).to(device).eval().requires_grad_(False)
        if "student_action" in payload:
            self.policy.student.mixtures["action"].load_state_dict(payload["student_action"],strict=True)
        if "transport" in payload:
            self.policy.transport.load_state_dict(payload["transport"],strict=True)
            if self.policy.step_condition is not None:
                self.policy.step_condition.load_state_dict(payload["step_condition"],strict=True)
        self.device=torch.device(device); self.schedule=schedule; self.graphs={}
        self.mot=self.policy.student
        self.version="v0"; self.torch_dtype=torch.float32

    def eval(self):
        self.policy.eval()
        return self

    @torch.no_grad()
    def actions(self,sample,noise,schedule=None,use_graph=False):
        name=schedule or self.schedule
        if name not in SCHEDULES: raise ValueError("Unknown RT schedule")
        depths=SCHEDULES[name]
        if not use_graph:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                return self.policy.rollout(sample,depths,noise)
        key=(name,tuple(noise.shape),tuple(sample["context"].shape))
        if key not in self.graphs:
            static={k:v.clone() for k,v in sample.items()}
            static_noise=noise.clone()
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.autocast("cuda",dtype=torch.bfloat16):
                for _ in range(3): self.policy.rollout(static,depths,static_noise)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph),torch.autocast("cuda",dtype=torch.bfloat16):
                result=self.policy.rollout(static,depths,static_noise)
            self.graphs[key]=(graph,static,static_noise,result)
        graph,static,static_noise,result=self.graphs[key]
        for k,v in sample.items(): static[k].copy_(v)
        static_noise.copy_(noise)
        graph.replay()
        return result.clone()

    @torch.no_grad()
    def infer_action(self,prompt=None,input_image=None,action_horizon=32,proprio=None,
                     context=None,context_mask=None,seed=None,rand_device="cpu",
                     compile_action_infer=False,schedule=None,num_inference_steps=None,**kwargs):
        if action_horizon!=32 or proprio is None:
            raise ValueError("RT expects 32 actions and current proprioception")
        if prompt is not None:
            if context is not None: raise ValueError("Pass prompt or cached context")
            context,context_mask=self.parent.encode_prompt(prompt)
        if context is None or context_mask is None: raise ValueError("Text context is required")
        if kwargs.get("text_cfg_scale",1.)!=1. or kwargs.get("sigma_shift",1.) not in (None,1.):
            raise ValueError("RT requires CFG=1 and action shift=1")
        chosen=schedule or self.schedule
        if num_inference_steps is not None and num_inference_steps!=len(SCHEDULES[chosen]):
            raise ValueError("NFE must match the selected RT schedule")
        image=input_image.unsqueeze(0) if input_image.ndim==3 else input_image
        context=context.unsqueeze(0) if context.ndim==2 else context
        context_mask=context_mask.unsqueeze(0) if context_mask.ndim==1 else context_mask
        proprio=proprio.reshape(1,1,8)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            latents=self.parent._encode_input_image_latents_tensor(image.to(self.device,dtype=torch.float32))
        sample=dict(first_frame_latents=latents,context=context.to(self.device,dtype=torch.float32),
                    context_mask=context_mask.to(self.device,dtype=torch.bool),
                    proprio=proprio.to(self.device,dtype=torch.float32))
        generator=None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        noise=torch.randn((1,32,7),generator=generator,device=rand_device,dtype=torch.float32).to(self.device)
        action=self.actions(sample,noise,chosen,compile_action_infer)
        return {"action":action[0].cpu().float()}
