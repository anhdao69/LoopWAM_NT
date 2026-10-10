"""ChronoLoop: a recurrent memory written by LoopWAM's shared core loops (plans/ChronoLoop.md).

Prefix tokens P ride in front of the video stream: [P | obs | fut].

* P queries attend jointly over [P, obs] (no gate).
* Every non-P query (obs, future video, actions) adds a gated read
  ``tanh(alpha_h) * Attn_h(q, K_P, V_P)`` to its parent attention (rule G-a), with
  one alpha per head per physical block per expert; alpha = 0 gives the parent exactly.
* P never sees future frames or noisy actions, identically in training and prefill.

Prefix kinds:
  memory  : M = 16 tokens m = gamma * s + e, identity RoPE, clean (t = 0) modulation.
            Read-out h = P state after the last core loop (before the coda).
            U1: w = c tanh(LN(h)/c); s_k = lambda s_{k-1} + (1 - lambda) w.
            mem_source learned (carry s), reset (s = 0 every query: CL-REG), oracle (s from a
            privileged stage-label embedding: CL-ORACLE). mem_write loop (written by the core)
            or external (CL-W2: memory read-only in the backbone; a separate Wan block updates it).
  history : CL-FRAME, the clean frame from query k-3 (392 tokens) at temporal RoPE position -2.

All memory state is functional: callers pass s_{k-1} and receive (outputs, h_k, s_k).
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .loop_mot import LoopMoT
from .loopwam import LoopWAM, _validate_checkpoint_depth, masked_action_loss
from .wan_video_dit import flash_attention

FORMAT = 'chronoloop-v1'


@dataclass(frozen=True)
class ChronoConfig:
    memory_tokens: int = 0            # M; 0 = no memory
    mem_source: str = 'none'          # none | learned | reset | oracle
    mem_write: str = 'none'           # none | loop | external
    history_frame: int = 0            # CL-FRAME: queries back (3), 0 = off
    action_loops: int = 4
    video_loops: int = 4
    clamp_c: float = 3.0
    lambda_logit_init: float = 2.2
    oracle_labels: int = 64

    def __post_init__(self):
        if self.memory_tokens:
            if self.mem_source not in {'learned', 'reset', 'oracle'}:
                raise ValueError('memory runs need mem_source learned|reset|oracle')
            if self.mem_source != 'oracle' and self.mem_write not in {'loop', 'external'}:
                raise ValueError('memory runs need mem_write loop|external')
        elif self.mem_source != 'none' or self.mem_write != 'none':
            raise ValueError('mem_source/mem_write require memory_tokens > 0')
        if self.history_frame and self.memory_tokens:
            raise ValueError('CL-FRAME has no memory tokens')
        if self.history_frame not in (0, 3):
            raise ValueError('history_frame must be 0 or 3 (plan: query k-3)')

    @property
    def gated(self):
        return bool(self.memory_tokens or self.history_frame)

    @property
    def carries_state(self):
        return self.memory_tokens > 0 and self.mem_source == 'learned'

    def flags_hash(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def _gate(x, g, heads):
    """Per-head scale of an attention output [B, S, H*Dh] by g [H]."""
    return (x.unflatten(-1, (heads, -1)) * g.to(x.dtype).view(heads, 1)).flatten(-2)


class ChronoMemory(nn.Module):
    """Memory parameters (all trained at the memory learning rate, weight decay 0)."""

    def __init__(self, cfg: ChronoConfig, dim: int, heads: int, blocks: int, updater_block=None):
        super().__init__()
        self.cfg = cfg
        self.alpha_video = nn.Parameter(torch.zeros(blocks, heads))
        self.alpha_action = nn.Parameter(torch.zeros(blocks, heads))
        if cfg.memory_tokens:
            m = cfg.memory_tokens
            self.e = nn.Parameter(torch.zeros(m, dim))
            self.gamma = nn.Parameter(torch.ones(dim))
            self.a = nn.Parameter(torch.full((dim,), cfg.lambda_logit_init))
            self.ln = nn.LayerNorm(dim, elementwise_affine=True)
            if cfg.mem_write == 'external':
                self.updater = updater_block
            if cfg.mem_source == 'oracle':
                self.oracle = nn.Embedding(cfg.oracle_labels, m * dim)
                nn.init.zeros_(self.oracle.weight)

    @torch.no_grad()
    def initialize_from_obs(self, obs_tokens: torch.Tensor, seed: int = 0):
        """gamma = per-channel RMS(obs after prepare)/c; e ~ N(0, std(obs)^2)."""
        x = obs_tokens.detach().float().reshape(-1, obs_tokens.shape[-1])
        self.gamma.copy_(x.square().mean(0).sqrt() / self.cfg.clamp_c)
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.e.copy_(torch.randn(self.e.shape, generator=g).to(self.e) * x.std().to(self.e))

    def tokens(self, s):
        return self.gamma * s + self.e

    def squash(self, h):
        c = self.cfg.clamp_c
        return c * torch.tanh(self.ln(h.float()) / c)

    def update(self, s_prev, w_hat):
        lam = torch.sigmoid(self.a)
        return lam * s_prev + (1 - lam) * w_hat

    def oracle_state(self, labels):
        c = self.cfg.clamp_c
        return c * torch.tanh(self.oracle(labels).view(-1, self.cfg.memory_tokens, self.e.shape[-1]) / c)


class ChronoLoopMoT(LoopMoT):
    """LoopMoT with gated prefix tokens; prefix-free calls use the parent path unchanged."""

    def attach_memory(self, memory: ChronoMemory):
        self.memory = memory

    # ---------------------------------------------------------------- attention core
    def _prefixed_attention(self, vio, aio, p, n_obs, g_v, g_a, readonly):
        H = self.num_heads
        vq, vk, vv = vio[:3]
        nv = vq.shape[1]
        out = []
        if not readonly:
            out.append(flash_attention(vq[:, :p], vk[:, :p + n_obs], vv[:, :p + n_obs], H))
        obs = flash_attention(vq[:, p:p + n_obs], vk[:, p:p + n_obs], vv[:, p:p + n_obs], H)
        fut = (flash_attention(vq[:, p + n_obs:], vk[:, p:], vv[:, p:], H) if nv > p + n_obs else None)
        act = None
        q_np = vq[:, p:]
        if aio is not None:
            aq, ak, av = aio[:3]
            act = flash_attention(aq, torch.cat((vk[:, p:p + n_obs], ak), 1), torch.cat((vv[:, p:p + n_obs], av), 1), H)
            q_np = torch.cat((q_np, aq), 1)
        mem = flash_attention(q_np, vk[:, :p], vv[:, :p], H)
        nvid = nv - p
        video_np = obs if fut is None else torch.cat((obs, fut), 1)
        video_np = video_np + _gate(mem[:, :nvid], g_v, H)
        video = torch.cat(out + [video_np], 1) if not readonly else video_np
        action = act + _gate(mem[:, nvid:], g_a, H) if aio is not None else None
        return video, action

    def _chrono_block(self, i, p, n_obs, readonly, video, action, g_v, g_a, video_freqs, action_freqs,
                      video_t_mod, action_t_mod, vctx, vmask, actx, amask):
        vb = self.mixtures['video'].blocks[i]
        vio = self._build_expert_attention_io(self.mixtures['video'], vb, video, video_freqs, video_t_mod)
        aio = None
        if action is not None:
            ab = self.mixtures['action'].blocks[i]
            aio = self._build_expert_attention_io(self.mixtures['action'], ab, action, action_freqs, action_t_mod)
        v_mixed, a_mixed = self._prefixed_attention(vio, aio, p, n_obs, g_v, g_a, readonly)
        if readonly:
            sl = lambda t: t[:, p:]
            vio_np = (None, None, None, sl(vio[3]), sl(vio[4]), sl(vio[5]), sl(vio[6]), sl(vio[7]))
            v_new = torch.cat((video[:, :p], self._post(vb, vio_np, v_mixed, vctx, vmask[:, p:])), 1)
        else:
            v_new = self._post(vb, vio, v_mixed, vctx, vmask)
        a_new = self._post(ab, aio, a_mixed, actx, amask) if action is not None else None
        return v_new, a_new

    def _run_chrono(self, i, p, n_obs, readonly, video, action, cond):
        fn = partial(self._chrono_block, i, p, n_obs, readonly)
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(fn, video, action, *cond, use_reentrant=False)
        return fn(video, action, *cond)

    # ---------------------------------------------------------------- training
    def forward_prefixed(self, video_tokens, action_tokens, video_freqs, action_freqs, video_t_mod,
                         action_t_mod, video_context, video_context_mask, action_context,
                         action_context_mask, *, prefix_len, observation_tokens, readonly=False):
        """Returns ((video, action) after the final exit, P state and obs state after the last core loop)."""
        k, ka = self.loops, self.action_loops
        if ka > k:
            raise ValueError('ChronoLoop supports action loops <= video loops (late alignment)')
        mem = self.memory
        p, n_obs = prefix_len, observation_tokens

        def run(i, v, a):
            cond = (torch.tanh(mem.alpha_video[i]), torch.tanh(mem.alpha_action[i]), video_freqs, action_freqs,
                    video_t_mod, action_t_mod, video_context, video_context_mask, action_context, action_context_mask)
            return self._run_chrono(i, p, n_obs, readonly, v, a, cond)

        v, a = video_tokens, action_tokens
        for i in range(self.pre_depth):
            v, a = run(i, v, a)
        for r in range(1, k + 1):
            for i in range(self.pre_depth, self.pre_depth + self.core_depth):
                if r <= k - ka:
                    v, _ = run(i, v, None)
                else:
                    v, a = run(i, v, a)
            if self.collect_diagnostics:
                self.last_diagnostics[f'loop/{r}/video_state_rms'] = v[:, p:].detach().float().square().mean().sqrt()
        h, obs_h = v[:, :p], v[:, p:p + n_obs]
        for i in range(self.pre_depth + self.core_depth, self.unique_depth):
            v, a = run(i, v, a)
        return (v, a), h, obs_h

    # ---------------------------------------------------------------- inference
    def prefill_prefixed(self, video_tokens, video_freqs, video_t_mod, video_context, video_context_mask,
                         *, prefix_len, readonly=False):
        """Clean prefix [P | obs] through every virtual layer; separate obs / P caches."""
        mem, p = self.memory, prefix_len
        n_obs = video_tokens.shape[1] - p
        x = video_tokens
        ko, vo, kp, vp, h = [], [], [], [], None
        sched = self.virtual_schedule()
        last_core = max(n for n, (key, _) in enumerate(sched) if key[0] == 'core')
        for n, (_, i) in enumerate(sched):
            vb = self.mixtures['video'].blocks[i]
            vio = self._build_expert_attention_io(self.mixtures['video'], vb, x, video_freqs, video_t_mod)
            ko.append(vio[1][:, p:]); vo.append(vio[2][:, p:]); kp.append(vio[1][:, :p]); vp.append(vio[2][:, :p])
            mixed, _ = self._prefixed_attention(vio, None, p, n_obs, torch.tanh(mem.alpha_video[i]), None, readonly)
            if readonly:
                sl = lambda t: t[:, p:]
                vio_np = (None, None, None, sl(vio[3]), sl(vio[4]), sl(vio[5]), sl(vio[6]), sl(vio[7]))
                x = torch.cat((x[:, :p], self._post(vb, vio_np, mixed, video_context, video_context_mask[:, p:])), 1)
            else:
                x = self._post(vb, vio, mixed, video_context, video_context_mask)
            if n == last_core:
                h, obs_h = x[:, :p], x[:, p:]
        return ko, vo, kp, vp, h, obs_h

    def action_with_prefixed_cache(self, action_tokens, action_freqs, action_t_mod, action_context,
                                   action_context_mask, ko, vo, kp, vp):
        mem, H = self.memory, self.num_heads
        sched = self.virtual_schedule()
        x = action_tokens
        for slot in self.action_cache_slots():
            _, i = sched[slot]
            ab = self.mixtures['action'].blocks[i]
            io = self._build_expert_attention_io(self.mixtures['action'], ab, x, action_freqs, action_t_mod)
            base = flash_attention(io[0], torch.cat((ko[slot], io[1]), 1), torch.cat((vo[slot], io[2]), 1), H)
            read = flash_attention(io[0], kp[slot], vp[slot], H)
            mixed = base + _gate(read, torch.tanh(mem.alpha_action[i]), H)
            x = self._post(ab, io, mixed, action_context, action_context_mask)
        return x


class ChronoLoopWAM(LoopWAM):
    """LoopWAM v0 with ChronoLoop memory. Without prefix tokens it is the parent model."""

    def configure_chrono(self, cfg: ChronoConfig, parent_sha256: Optional[str] = None):
        self.chrono = cfg
        self.parent_sha256 = parent_sha256
        if not cfg.gated:
            self.mot.memory = None
            return None
        video = self.video_expert
        updater = None
        if cfg.mem_write == 'external':
            updater = copy.deepcopy(video.blocks[self.mot.pre_depth])   # init from the first core block
        memory = ChronoMemory(cfg, video.hidden_dim if hasattr(video, 'hidden_dim') else video.dim,
                              self.mot.num_heads, self.mot.unique_depth, updater)
        memory.to(device=self.device, dtype=self.torch_dtype)
        self.mot.attach_memory(memory)
        return memory

    # ---------------------------------------------------------------- helpers
    @property
    def memory(self):
        return self.mot.memory

    def _encode_training_video(self, sample, input_video, tiled=False):
        if 'latents' in sample:
            return sample['latents']
        return super()._encode_training_video(sample, input_video, tiled)

    def _history_freqs(self, h, w):
        freq_f, freq_h, freq_w = self.video_expert.freqs
        f = torch.conj(freq_f[2]).view(1, 1, 1, -1).expand(1, h, w, -1)     # temporal position -2
        return torch.cat([f, freq_h[:h].view(1, h, 1, -1).expand(1, h, w, -1),
                          freq_w[:w].view(1, 1, w, -1).expand(1, h, w, -1)], -1).reshape(h * w, 1, -1)

    def _identity_freqs(self, n, like):
        return torch.ones((n, 1, like.shape[-1]), dtype=like.dtype, device=like.device)

    def _prefix(self, video, s_prev=None, history_latents=None, oracle_labels=None):
        """Build (P tokens, P freqs, P t_mod, context mask) and the state fed in."""
        tokens, t, t_mod, ctx, ctx_mask, freqs, f, hh, ww, tpf = video
        b = tokens.shape[0]
        cfg = self.chrono
        if cfg.memory_tokens:
            mem = self.memory
            if cfg.mem_source == 'oracle':
                s_in = mem.oracle_state(oracle_labels)
            elif cfg.mem_source == 'reset' or s_prev is None:
                s_in = torch.zeros(b, cfg.memory_tokens, tokens.shape[-1], device=tokens.device, dtype=torch.float32)
            else:
                s_in = s_prev
            p_tokens = mem.tokens(s_in).to(tokens.dtype)
            p_freqs = self._identity_freqs(cfg.memory_tokens, freqs)
        else:
            s_in = None
            hist = history_latents.to(device=tokens.device, dtype=tokens.dtype)
            x = self.video_expert.patchify(hist)
            p_tokens = x.flatten(2).transpose(1, 2).contiguous()
            p_freqs = self._history_freqs(x.shape[3], x.shape[4])
        p = p_tokens.shape[1]
        p_tmod = t_mod[:, :1].expand(-1, p, *t_mod.shape[2:])
        p_mask = ctx_mask[:, :1].expand(-1, p, -1)
        return p_tokens, p_freqs, p_tmod, p_mask, s_in

    def _readout(self, h, s_in, obs_after_core=None, video_ctx=None, video_ctx_mask=None, obs_freqs=None,
                 obs_tmod=None):
        cfg, mem = self.chrono, self.memory
        if not cfg.memory_tokens or cfg.mem_source == 'oracle':
            return None, None
        if cfg.mem_write == 'external':
            h = self._external_write(s_in, obs_after_core, video_ctx, video_ctx_mask, obs_freqs, obs_tmod)
        w_hat = mem.squash(h)
        return mem.update(s_in.float(), w_hat), w_hat

    def _external_write(self, s_in, obs_h, ctx, ctx_mask, obs_freqs, obs_tmod):
        """CL-W2: one Wan block; query = gamma*s + e, keys = [itself; obs after the last core loop]."""
        mem = self.memory
        block = mem.updater
        q_tokens = mem.tokens(s_in).to(obs_h.dtype)
        m = q_tokens.shape[1]
        x = torch.cat((q_tokens, obs_h), 1)
        freqs = torch.cat((self._identity_freqs(m, obs_freqs), obs_freqs), 0)
        tmod = torch.cat((obs_tmod[:, :1].expand(-1, m, *obs_tmod.shape[2:]), obs_tmod), 1)
        io = self.mot._build_expert_attention_io(self.video_expert, block, x, freqs, tmod)
        mixed = flash_attention(io[0][:, :m], io[1], io[2], self.mot.num_heads)
        sl = lambda t: t[:, :m]
        io_m = (None, None, None, sl(io[3]), sl(io[4]), sl(io[5]), sl(io[6]), sl(io[7]))
        return self.mot._post(block, io_m, mixed, ctx, ctx_mask[:, :m])

    # ---------------------------------------------------------------- training
    def window_losses(self, sample, noise, s_prev=None):
        """One window per row. Returns (per-row weighted loss [B], logs, s_new or None)."""
        noisy = self.prepare_training_batch(sample, noise_video=noise['video'], noise_action=noise['action'],
                                            timestep_video=noise['t_video'], timestep_action=noise['t_action'])
        video = self.video_expert.prepare(x=noisy['latents_video'], timestep=noisy['timestep_video'],
            context=noisy['context'], context_mask=noisy['context_mask'], action=None,
            fuse_vae_embedding_in_latents=True)
        action = self.action_expert.prepare(action_tokens=noisy['latents_action'],
            timestep=noisy['timestep_action'], context=noisy['context'], context_mask=noisy['context_mask'])
        n_obs = int(video[9])
        s_new = w_hat = None
        if not self.chrono.gated:
            # Parent path, bit-exact (CL-0, CL-0@1).
            states = self.mot.forward_joint_exits(video_tokens=video[0], action_tokens=action[0],
                video_freqs=video[5], action_freqs=action[5], video_t_mod=video[2], action_t_mod=action[2],
                video_context=video[3], video_context_mask=video[4], action_context=action[3],
                action_context_mask=action[4], attention_mask=noisy['attention_mask'])
            (v_out, a_out), = states.values()
        else:
            p_tokens, p_freqs, p_tmod, p_mask, s_in = self._prefix(
                video, s_prev, sample.get('history_latents'), sample.get('oracle_labels'))
            p = p_tokens.shape[1]
            readonly = self.chrono.mem_write == 'external'
            (v_full, a_out), h, obs_h = self.mot.forward_prefixed(
                torch.cat((p_tokens, video[0]), 1), action[0], torch.cat((p_freqs, video[5]), 0), action[5],
                torch.cat((p_tmod, video[2]), 1), action[2], video[3], torch.cat((p_mask, video[4]), 1),
                action[3], action[4], prefix_len=p, observation_tokens=n_obs, readonly=readonly)
            v_out = v_full[:, p:]
            if self.chrono.memory_tokens:
                s_new, w_hat = self._readout(h, s_in, obs_h, video[3], video[4], video[5][:n_obs], video[2][:, :n_obs])
        pred_v = self.video_expert.post(v_out, video[1], video[6], video[7], video[8])
        pred_a = self.action_expert.post(a_out)
        raw_v = self._compute_video_loss_per_sample(pred_v[:, :, 1:], noisy['target_video'][:, :, 1:],
                                                    noisy['image_is_pad'], include_initial_video_step=False)
        raw_a = masked_action_loss(pred_a, noisy['target_action'], noisy['action_is_pad'])
        wv = self.train_video_scheduler.training_weight(noisy['timestep_video'])
        wa = self.train_action_scheduler.training_weight(noisy['timestep_action'])
        lv, la = raw_v * wv, raw_a * wa
        loss = self.loss_lambda_video * lv + self.loss_lambda_action * la
        logs = dict(video=lv.detach(), action=la.detach(), video_raw=raw_v.detach(), action_raw=raw_a.detach())
        if s_new is not None:
            logs['mem_s_rms'] = s_new.detach().float().square().mean(dim=(1, 2)).sqrt()
            logs['mem_saturated'] = (w_hat.detach().abs() > 0.95 * self.chrono.clamp_c).float().mean(dim=(1, 2))
        return loss, logs, s_new

    def forward(self, segment, s_init=None, sequential=False):
        """One DDP forward for a TBPTT segment (list of window batches) of a stream micro-batch.

        Stateless runs pass a single concatenated window batch. Stateful runs carry
        s through the windows of the segment (gradient flows through s); rows absent
        from a window (finished traversals) keep their state. Returns (sum of per-window
        losses, summed logs, detached final state).
        """
        total, logs = 0., {}
        s = s_init
        for entry in segment:
            if sequential:
                rows = entry['rows']
                loss, wlogs, s_new = self.window_losses(entry['sample'], entry['noise'], s.index_select(0, rows))
                if s_new is not None:
                    # Keep every memory parameter in the graph (DDP) even when s is not consumed later.
                    loss = loss + 0. * s_new.sum() / max(1, loss.numel())
                    s = s.index_copy(0, rows, s_new.to(s.dtype))
            else:
                loss, wlogs, s_new = self.window_losses(entry['sample'], entry['noise'], None)
                if s_new is not None:
                    loss = loss + 0. * s_new.sum() / max(1, loss.numel())
            total = total + loss.sum()
            for key, value in wlogs.items():
                logs[key] = logs.get(key, 0.) + value.float().sum()
        return total, logs, (None if s is None else s.detach())

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def chrono_infer_action(self, input_image, proprio, context, context_mask, state=None, history_image=None,
                            num_inference_steps=10, seed=None, rand_device='cpu', oracle_labels=None,
                            action_horizon=32):
        """One replanning query. Returns {'action', 'state'}; the caller owns the state
        (reset to None at episode start; settling steps never call this)."""
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        proprio = proprio.reshape(1, -1).to(device=self.device, dtype=self.torch_dtype)
        context = (context.unsqueeze(0) if context.ndim == 2 else context).to(self.device, self.torch_dtype)
        context_mask = (context_mask.unsqueeze(0) if context_mask.ndim == 1 else context_mask).to(self.device, torch.bool)
        context, context_mask = self._append_proprio_to_context(context, context_mask, proprio)
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn((1, action_horizon, self.action_expert.action_dim), generator=generator,
                                     device=rand_device, dtype=torch.float32).to(self.device, self.torch_dtype)
        first = self._encode_input_image_latents_tensor(input_image.to(self.device, self.torch_dtype))
        video = self.video_expert.prepare(x=first, timestep=torch.zeros((1,), dtype=first.dtype, device=self.device),
            context=context, context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=True)
        new_state = None
        if not self.chrono.gated:
            ko, vo = self.mot.prefill_video_cache_tensor(video[0], video[5], video[2], video[3], video[4],
                torch.ones(video[0].shape[1], video[0].shape[1], dtype=torch.bool, device=self.device))
            kp = vp = None
        else:
            hist = None
            if self.chrono.history_frame:
                himg = input_image if history_image is None else history_image
                if himg.ndim == 3: himg = himg.unsqueeze(0)
                hist = self._encode_input_image_latents_tensor(himg.to(self.device, self.torch_dtype))
            p_tokens, p_freqs, p_tmod, p_mask, s_in = self._prefix(video, state, hist, oracle_labels)
            p = p_tokens.shape[1]
            readonly = self.chrono.mem_write == 'external'
            ko, vo, kp, vp, h, obs_h = self.mot.prefill_prefixed(torch.cat((p_tokens, video[0]), 1),
                torch.cat((p_freqs, video[5]), 0), torch.cat((p_tmod, video[2]), 1), video[3],
                torch.cat((p_mask, video[4]), 1), prefix_len=p, readonly=readonly)
            if self.chrono.memory_tokens:
                new_state, _ = self._readout(h, s_in, obs_h, video[3],
                                          video[4], video[5], video[2])
        times, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps, device=self.device, dtype=latents_action.dtype)
        n_obs = video[0].shape[1]
        amask = torch.ones(action_horizon, n_obs + action_horizon, dtype=torch.bool, device=self.device)
        for t_step, delta in zip(times, deltas):
            a = self.action_expert.prepare(action_tokens=latents_action, timestep=t_step.unsqueeze(0).to(latents_action.dtype),
                                           context=context, context_mask=context_mask)
            if kp is None:
                tokens = self.mot.forward_action_with_video_cache_tensor(a[0], a[5], a[2], a[3], a[4], ko, vo, amask)
            else:
                tokens = self.mot.action_with_prefixed_cache(a[0], a[5], a[2], a[3], a[4], ko, vo, kp, vp)
            latents_action = self.infer_action_scheduler.step(self.action_expert.post(tokens), delta, latents_action)
        return {'action': latents_action[0].float().cpu(), 'state': new_state}

    # ---------------------------------------------------------------- checkpoints
    def save_chrono(self, path, optimizer=None, step=None, training_state=None, weights_only=False):
        path = Path(path)
        payload = dict(format_version=FORMAT, chrono=asdict(self.chrono), chrono_flags_sha256=self.chrono.flags_hash(),
                       parent_sha256=self.parent_sha256, architecture=self.architecture_metadata, version=self.version,
                       video_loops=self.mot.loops, action_loops=self.mot.action_loops, loop_alignment='late',
                       mot=self.mot.state_dict(), proprio_encoder=self.proprio_encoder.state_dict(), step=step,
                       training_state=None if weights_only else training_state)
        if optimizer is not None and not weights_only:
            payload['optimizer'] = optimizer.state_dict()
        tmp = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, tmp)
        tmp.replace(path)


def create_chronoloop(cfg: ChronoConfig, vae_path, *, parent_path=None, checkpoint_path=None, device='cpu',
                      checkpoint_blocks=False, model_dtype=torch.float32, parent_sha256=None):
    """Fresh ChronoLoop from the full-suite v0 4/4 parent, or an exact ChronoLoop checkpoint.

    The checkpoint guard refuses a checkpoint whose memory flags differ from ``cfg``.
    """
    from .loopwam_init import load_wan21_vae, target_configs
    from .wan_video_dit import WanVideoDiT
    from .action_dit import ActionDiT
    source = checkpoint_path or parent_path
    payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    if checkpoint_path is not None:
        if payload.get('format_version') != FORMAT:
            raise ValueError('Expected a ChronoLoop checkpoint')
        if payload['chrono'] != asdict(cfg):
            raise ValueError(f'Checkpoint guard: memory flags differ: saved {payload["chrono"]} vs requested {asdict(cfg)}')
    else:
        if payload.get('format_version') != 'loopwam-s-v1' or payload['version'] != 'v0':
            raise ValueError('ChronoLoop initializes from a LoopWAM v0 checkpoint')
        _validate_checkpoint_depth(payload)
        loops = payload['inference_loops']
        if (payload.get('video_loops', loops), payload.get('action_loops', loops)) != (4, 4):
            raise ValueError('ChronoLoop parent must be the v0 4/4 checkpoint')
    metadata = payload['architecture']
    video_cfg, action_cfg = target_configs(12)
    if metadata['target_video_config'] != video_cfg or metadata['target_action_config'] != action_cfg:
        raise ValueError('Checkpoint architecture is not the declared LoopWAM-S')
    video, action = WanVideoDiT(**video_cfg), ActionDiT(**action_cfg)
    vae = load_wan21_vae(vae_path, device=device, dtype=torch.bfloat16 if str(device).startswith('cuda') else torch.float32)
    video, action = video.to(device=device, dtype=model_dtype), action.to(device=device, dtype=model_dtype)
    mot = ChronoLoopMoT({'video': video, 'action': action}, loops=cfg.video_loops, version='v0',
                        checkpoint_blocks=checkpoint_blocks,
                        action_loops=None if cfg.action_loops == cfg.video_loops else cfg.action_loops)
    model = ChronoLoopWAM(video_expert=video, action_expert=action, mot=mot, vae=vae, text_dim=4096,
        proprio_dim=8, device=device, torch_dtype=model_dtype, version='v0', exit_weight_scale=payload['exit_weight_scale']
        if 'exit_weight_scale' in payload else 1.0, architecture_metadata=metadata,
        video_train_shift=5.0, video_infer_shift=5.0, action_train_shift=1.0, action_infer_shift=1.0)
    parent_state = {k: v for k, v in payload['mot'].items() if not k.startswith('memory.')}
    missing, unexpected = model.mot.load_state_dict(parent_state, strict=False)
    if missing or unexpected:
        raise ValueError(f'Backbone load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}')
    model.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
    memory = model.configure_chrono(cfg, parent_sha256=payload.get('parent_sha256', parent_sha256))
    if checkpoint_path is not None and memory is not None:
        mem_state = {k[len('memory.'):]: v for k, v in payload['mot'].items() if k.startswith('memory.')}
        memory.load_state_dict(mem_state, strict=True)
    model.loaded_payload_meta = {k: payload.get(k) for k in ('step', 'chrono_flags_sha256', 'parent_sha256')}
    return model, payload
