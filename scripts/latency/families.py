"""Model-family adapters for measure_latency.py. Each adapter answers one batch-one LIBERO query per backend.

query(record, config, backend) -> (raw normalized action chunk [32, 7] on CPU, host LIBERO actions, preprocessing ms)
Imports resolve against the --code-root worktree, which measure_latency.py puts first on sys.path.
"""
import json, subprocess, time
from pathlib import Path

import torch

from fast_graph import WholeQueryGraph, bf16_linear


def git_head(root):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def seed42_noise(shape):
    generator = torch.Generator(device="cpu").manual_seed(42)
    return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)


class _Base:
    def __init__(self, a):
        from scripts.evaluate_loopwam_libero import ObservationAdapter
        stats_path = Path(a.stats) if a.stats else Path(a.checkpoint).parent / "dataset_stats.json"
        self.adapter = ObservationAdapter(json.loads(stats_path.read_text()), a.text_cache_dir, None)
        self.stats_path = str(stats_path)
        self._texts, self._gpu_texts, self.fast = {}, {}, False

    def text(self, language):
        if language not in self._texts:
            self._texts[language] = self.adapter.text(language)
        return self._texts[language]

    def gpu_text(self, language):
        """Per-episode constant: the fast backend keeps the cached T5 context resident on the GPU."""
        if language not in self._gpu_texts:
            context, mask = self.text(language)
            context = context.unsqueeze(0) if context.ndim == 2 else context
            mask = mask.unsqueeze(0) if mask.ndim == 1 else mask
            self._gpu_texts[language] = (context.to("cuda", dtype=torch.float32), mask.to("cuda", dtype=torch.bool))
        return self._gpu_texts[language]

    def observe(self, record):
        t0 = time.perf_counter()
        image, proprio = self.adapter.observation(record["obs"])
        return image, proprio, (time.perf_counter() - t0) * 1e3


class RT(_Base):
    """Residual-transport / elastic checkpoints: fastwam.models.wan22.transport_inference.TransportInference.
    config = schedule name in residual_transport.SCHEDULES (per-denoising-step depth list)."""

    def __init__(self, a, extra):
        super().__init__(a)
        from fastwam.models.wan22.transport_inference import TransportInference
        from fastwam.models.wan22.residual_transport import SCHEDULES
        self.schedules = SCHEDULES
        self.m = TransportInference(a.checkpoint, a.vae_path, device="cuda:0",
                                    zero_shot_kind=extra.get("zero_shot_kind")).eval()
        self.graphs = {}

    def info(self):
        return dict(loader="TransportInference", policy_parameters=sum(p.numel() for p in self.m.policy.parameters()),
                    dataset_stats=self.stats_path)

    def config_info(self, config):
        return dict(depths=list(self.schedules[config]), denoising_steps=len(self.schedules[config]))

    def enable_fast(self):
        self.fast = True
        return bf16_linear(self.m.policy, self.m.parent)

    def release_graphs(self):
        self.m.graphs.clear()
        self.graphs.clear()

    def _whole(self, config):
        if config not in self.graphs:
            depths, m = self.schedules[config], self.m

            def fn(image, context, mask, proprio, noise):
                latents = m.parent._encode_input_image_latents_tensor(image)
                sample = dict(first_frame_latents=latents, context=context, context_mask=mask, proprio=proprio)
                return m.policy.rollout(sample, depths, noise)
            self.graphs[config] = WholeQueryGraph(fn)
        return self.graphs[config]

    @torch.no_grad()
    def query(self, record, config, backend):
        image, proprio, pre = self.observe(record)
        if backend in ("eager", "graph"):
            context, mask = self.text(record["language"])
            self.m.schedule = config
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                action = self.m.infer_action(input_image=image, proprio=proprio, context=context, context_mask=mask,
                    action_horizon=32, num_inference_steps=len(self.schedules[config]), text_cfg_scale=1.0, seed=42,
                    compile_action_infer=backend == "graph")["action"]
        else:
            context, mask = self.gpu_text(record["language"])
            with torch.inference_mode():
                out = self._whole(config)(image=image.unsqueeze(0).to(torch.float32), context=context, mask=mask,
                                          proprio=proprio.reshape(1, 1, 8).to(torch.float32),
                                          noise=seed42_noise((1, 32, 7)))
            action = out[0].cpu().float()
        return action, self.adapter.libero_actions(action), pre


class Native(_Base):
    """Models answered by FastWAM.infer_action (released FastWAM; LoopWAM v0/v1/v2/dense; KV-concat/mix; partition).
    graph = the init-variant evaluation backend (evaluate_v0_matched.py Policy.graphed): one CUDA graph of the native
    proprio-token + video prefill + ten denoising steps, VAE eager. compile = upstream compile_action_infer=True
    (Inductor reduce-overhead). config = 'native' (each checkpoint's stored depth)."""

    STEPS = 10

    def __init__(self, a, extra, model):
        super().__init__(a)
        self.m = model.eval()
        self.m.requires_grad_(False)
        self.fuse = bool(getattr(self.m.video_expert, "fuse_vae_embedding_in_latents", False))
        self.core_graph, self.whole = None, None

    def info(self):
        mot = getattr(self.m, "mot", None)
        return dict(loader=type(self.m).__name__, version=getattr(self.m, "version", None),
                    video_loops=getattr(mot, "loops", None), action_loops=getattr(mot, "action_loops", None),
                    action_kv_mode=getattr(mot, "action_kv_mode", None),
                    parameters=sum(p.numel() for p in self.m.parameters()), dataset_stats=self.stats_path)

    def config_info(self, config):
        return dict(denoising_steps=self.STEPS)

    def enable_fast(self):
        self.fast = True
        return bf16_linear(self.m)

    def release_graphs(self):
        self.core_graph, self.whole = None, None

    def _noise(self):
        m = self.m
        return seed42_noise((1, 32, m.action_expert.action_dim)).to(device=m.device, dtype=m.torch_dtype)

    def _core(self, latents, proprio, context, mask, noise):
        # FastWAM.infer_action after noise/VAE: proprio token, video prefill, ten action denoising steps.
        m = self.m
        context, mask = m._append_proprio_to_context(context=context, context_mask=mask, proprio=proprio)
        timestep_video = torch.zeros((latents.shape[0],), dtype=latents.dtype, device=m.device)
        (tokens, _t, t_mod, video_context, video_mask, freqs, _f, _h, _w, per_frame) = m.video_expert.prepare(
            x=latents, timestep=timestep_video, context=context, context_mask=mask, action=None,
            fuse_vae_embedding_in_latents=self.fuse)
        n = int(tokens.shape[1])
        attention = m._build_mot_attention_mask(video_seq_len=n, action_seq_len=noise.shape[1],
                                                video_tokens_per_frame=per_frame, device=tokens.device)
        cache_k, cache_v = m.mot.prefill_video_cache_tensor(
            video_tokens=tokens, video_freqs=freqs, video_t_mod=t_mod, video_context=video_context,
            video_context_mask=video_mask, video_attention_mask=attention[:n, :n])
        timesteps, deltas = m.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=self.STEPS, device=m.device, dtype=noise.dtype, shift_override=None)
        x = noise
        for step_t, delta in zip(timesteps, deltas):
            pred = m._denoise_action_with_video_cache(
                latents_action=x, timestep_action=step_t.unsqueeze(0).to(dtype=x.dtype, device=m.device),
                context=context, context_mask=mask, video_cache_k=cache_k, video_cache_v=cache_v,
                action_attention_mask=attention[n:, :])
            x = m.infer_action_scheduler.step(pred, delta, x)
        return x

    def _graphed(self, image, proprio, context, mask):
        m = self.m
        image = image.unsqueeze(0) if image.ndim == 3 else image
        latents = m._encode_input_image_latents_tensor(input_image=image.to(device=m.device, dtype=m.torch_dtype))
        args = [latents, proprio.unsqueeze(0).to(device=m.device, dtype=m.torch_dtype),
                (context.unsqueeze(0) if context.ndim == 2 else context).to(device=m.device, dtype=m.torch_dtype),
                (mask.unsqueeze(0) if mask.ndim == 1 else mask).to(device=m.device, dtype=torch.bool), self._noise()]
        if self.core_graph is None:
            self.static = [x.clone() for x in args]
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                for _ in range(3):
                    self._core(*self.static)
            torch.cuda.current_stream().wait_stream(stream)
            self.core_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.core_graph), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                self.core_out = self._core(*self.static)
        for dst, src in zip(self.static, args):
            if dst.shape != src.shape or dst.dtype != src.dtype:
                raise ValueError("CUDA graph input signature changed")
            dst.copy_(src)
        self.core_graph.replay()
        return self.core_out[0].to(device="cpu", dtype=torch.float32)

    def _fast(self, image, proprio, language):
        m = self.m
        if self.whole is None:
            def fn(image, proprio, context, mask, noise):
                latents = m._encode_input_image_latents_tensor(input_image=image)
                return self._core(latents, proprio, context, mask, noise)
            self.whole = WholeQueryGraph(fn)
        context, mask = self.gpu_text(language)
        image = image.unsqueeze(0) if image.ndim == 3 else image
        out = self.whole(image=image.to(dtype=m.torch_dtype), proprio=proprio.unsqueeze(0).to(dtype=m.torch_dtype),
                         context=context.to(dtype=m.torch_dtype), mask=mask, noise=self._noise())
        return out[0].to(device="cpu", dtype=torch.float32)

    @torch.no_grad()
    def query(self, record, config, backend):
        image, proprio, pre = self.observe(record)
        context, mask = self.text(record["language"])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if backend in ("eager", "compile"):
                action = self.m.infer_action(prompt=None, input_image=image, proprio=proprio, context=context,
                    context_mask=mask, action_horizon=32, num_inference_steps=self.STEPS, text_cfg_scale=1.0,
                    seed=42, compile_action_infer=backend == "compile")["action"]
                action = action.to(device="cpu", dtype=torch.float32)
            elif backend == "graph":
                action = self._graphed(image, proprio, context, mask)
            else:
                action = self._fast(image, proprio, record["language"])
        return action, self.adapter.libero_actions(action), pre


def loopwam(a, extra):
    from fastwam.models.wan22.loopwam import create_loopwam
    return Native(a, extra, create_loopwam(checkpoint_path=a.checkpoint, vae_path=a.vae_path, device="cuda:0",
                                           model_dtype=torch.float32, **extra.get("loader", {})))


def fastwam(a, extra):
    """Released FastWAM, loaded exactly as scripts/measure_policy_latency.py (main) does."""
    from omegaconf import OmegaConf
    from fastwam.runtime import create_fastwam
    config = OmegaConf.to_container(OmegaConf.load(Path(a.code_root) / "configs/model/fastwam.yaml"), resolve=False)
    config.pop("_target_")
    config["proprio_dim"] = 8
    for expert in ["video_dit_config", "action_dit_config"]:
        config[expert]["action_dim"] = 7
        config[expert]["use_gradient_checkpointing"] = False
    config.update(skip_dit_load_from_pretrain=True, load_text_encoder=False, device="cuda:0",
                  model_dtype=torch.float32)
    model = create_fastwam(**config)
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model.mot.load_state_dict(payload["mot"], strict=True)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    del payload
    return Native(a, extra, model)


class Chrono(_Base):
    """ChronoLoop LIBERO checkpoints (branch chrono): create_chronoloop + chrono_infer_action, as
    scripts/evaluate_chronoloop_libero.py. Stateful: the memory state is carried across the queries of an episode
    (each suite block of the observation file is one episode) and CL-FRAME reads the image of three queries before.
    graph is not offered (the ChronoLoop evaluation has no graph path); fast = BF16 Linear weights + one CUDA graph
    of the whole query with the state (zeros at episode start, which _prefix also uses for state=None) as input."""

    stateful = True

    def __init__(self, a, extra):
        super().__init__(a)
        from fastwam.models.wan22.chronoloop import ChronoConfig, create_chronoloop
        payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False, mmap=True)
        self.cfg = ChronoConfig(**payload["chrono"])
        del payload
        self.m, _ = create_chronoloop(self.cfg, a.vae_path, checkpoint_path=a.checkpoint, device="cuda:0")
        self.m.eval()
        self.m.requires_grad_(False)
        self.whole = None
        self.reset()

    def info(self):
        return dict(loader="create_chronoloop", chrono=dict(self.cfg.__dict__),
                    parameters=sum(p.numel() for p in self.m.parameters()), dataset_stats=self.stats_path)

    def config_info(self, config):
        return dict(denoising_steps=10)

    def reset(self):
        self.episode, self.state, self.images = None, None, []

    def release_graphs(self):
        self.whole = None

    def enable_fast(self):
        self.fast = True
        return bf16_linear(self.m)

    def _core(self, image, proprio, context, mask, noise, state, history):
        # chrono_infer_action from the proprio token on, with the sampler noise and the state passed in.
        m, cfg = self.m, self.m.chrono
        context, mask = m._append_proprio_to_context(context, mask, proprio)
        first = m._encode_input_image_latents_tensor(image)
        video = m.video_expert.prepare(x=first, timestep=torch.zeros((1,), dtype=first.dtype, device=m.device),
            context=context, context_mask=mask, action=None, fuse_vae_embedding_in_latents=True)
        new_state = None
        if not cfg.gated:
            ko, vo = m.mot.prefill_video_cache_tensor(video[0], video[5], video[2], video[3], video[4],
                torch.ones(video[0].shape[1], video[0].shape[1], dtype=torch.bool, device=m.device))
            kp = vp = None
        else:
            hist = m._encode_input_image_latents_tensor(history) if cfg.history_frame else None
            p_tokens, p_freqs, p_tmod, p_mask, s_in = m._prefix(video, state, hist, None)
            p = p_tokens.shape[1]
            ko, vo, kp, vp, h, obs_h = m.mot.prefill_prefixed(torch.cat((p_tokens, video[0]), 1),
                torch.cat((p_freqs, video[5]), 0), torch.cat((p_tmod, video[2]), 1), video[3],
                torch.cat((p_mask, video[4]), 1), prefix_len=p, readonly=cfg.mem_write == "external")
            if cfg.memory_tokens:
                new_state, _ = m._readout(h, s_in, obs_h, video[3], video[4], video[5], video[2])
        times, deltas = m.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=10, device=m.device, dtype=noise.dtype)
        n_obs, x = video[0].shape[1], noise
        amask = torch.ones(32, n_obs + 32, dtype=torch.bool, device=m.device)
        for t_step, delta in zip(times, deltas):
            a = m.action_expert.prepare(action_tokens=x, timestep=t_step.unsqueeze(0).to(x.dtype),
                                        context=context, context_mask=mask)
            if kp is None:
                tokens = m.mot.forward_action_with_video_cache_tensor(a[0], a[5], a[2], a[3], a[4], ko, vo, amask)
            else:
                tokens = m.mot.action_with_prefixed_cache(a[0], a[5], a[2], a[3], a[4], ko, vo, kp, vp)
            x = m.infer_action_scheduler.step(m.action_expert.post(tokens), delta, x)
        return (x, new_state) if new_state is not None else (x,)

    @torch.no_grad()
    def query(self, record, config, backend):
        if record["suite"] != self.episode:
            self.episode, self.state, self.images = record["suite"], None, []
        image, proprio, pre = self.observe(record)
        self.images.append(image)
        k = len(self.images) - 1
        history = self.images[max(0, k - 3)] if self.cfg.history_frame else None
        m = self.m
        if backend == "eager":
            context, mask = self.text(record["language"])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = m.chrono_infer_action(image, proprio, context, mask, state=self.state, history_image=history,
                                            num_inference_steps=10, seed=42)
            action, self.state = out["action"], out["state"]
        else:
            if self.whole is None:
                self.whole = WholeQueryGraph(lambda **kw: self._core(**kw))
            context, mask = self.gpu_text(record["language"])
            inputs = dict(image=image.unsqueeze(0).to(m.torch_dtype), proprio=proprio.reshape(1, -1).to(m.torch_dtype),
                          context=context.to(m.torch_dtype), mask=mask,
                          noise=seed42_noise((1, 32, m.action_expert.action_dim)).to(m.torch_dtype))
            if self.cfg.memory_tokens:
                inputs["state"] = (self.state if self.state is not None else
                                   torch.zeros(1, self.cfg.memory_tokens, m.video_expert.hidden_dim, dtype=torch.float32))
            else:
                inputs["state"] = torch.zeros(1)          # unused; keeps one call signature
            inputs["history"] = (history.unsqueeze(0).to(m.torch_dtype) if history is not None else torch.zeros(1))
            with torch.inference_mode():
                out = self.whole(**inputs)
            action = out[0][0].float().cpu()
            self.state = out[1].clone() if len(out) > 1 else None
        return action, self.adapter.libero_actions(action), pre


FAMILIES = dict(rt=RT, loopwam=loopwam, fastwam=fastwam, chrono=Chrono)


def build(family, a, extra):
    import fastwam as package
    if not Path(package.__file__).resolve().is_relative_to(Path(a.code_root).resolve()):
        raise SystemExit(f"fastwam resolved to {package.__file__}, not to --code-root {a.code_root}")
    return FAMILIES[family](a, extra)
