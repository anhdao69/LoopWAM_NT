"""Two-expert depth recurrence with shared exit decoding and virtual KV caches."""
from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .mot import MoT


def xsa_projection(attention_out: torch.Tensor, own_query_value: torch.Tensor,
                   num_heads: int) -> torch.Tensor:
    """Remove each head's own-value direction, with FP32 arithmetic (eps=1e-12)."""
    if attention_out.shape != own_query_value.shape:
        raise ValueError("XSA requires query-aligned own values with the output shape.")
    shape = attention_out.shape
    out = attention_out.float().reshape(*shape[:-1], num_heads, -1)
    own = own_query_value.float().reshape(*shape[:-1], num_heads, -1)
    unit = F.normalize(own, dim=-1, eps=1e-12)
    return (out - (out * unit).sum(dim=-1, keepdim=True) * unit).reshape(shape).to(attention_out.dtype)


class LoopMoT(MoT):
    """Twelve physical block pairs executing pre(3), core(6)*K, coda(3).

    ``num_layers`` remains the physical depth for serialization. Cache lists use
    ``virtual_schedule`` order: every repeated block has a distinct cache slot.
    Coda execution branches off the recurrent state and never changes that state.
    """

    unique_depth = 12
    pre_depth = 3
    core_depth = 6
    post_depth = 3
    trained_max_loops = 4

    def effective_depth(self, loops: Optional[int] = None) -> int:
        k = self.loops if loops is None else self._validate_loops(loops)
        return self.pre_depth + self.core_depth * k + self.post_depth

    def __init__(self, mixtures: Dict[str, nn.Module], loops: int = 4,
                 version: str = "v0", checkpoint_blocks: bool = False,
                 mot_checkpoint_mixed_attn: bool = False,
                 collect_diagnostics: bool = False):
        if set(mixtures) != {"video", "action"}:
            raise ValueError("LoopMoT requires exactly the video and action experts.")
        if version not in {"v0", "v1", "v2"}:
            raise ValueError(f"Unsupported LoopMoT version: {version}")
        # Canonical token order is always video followed by action.
        super().__init__({name: mixtures[name] for name in ("video", "action")},
                         mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn)
        if self.num_layers != 12:
            raise ValueError("LoopMoT requires 12 physical blocks per expert (3/6/3).")
        self.version = version
        self.loops = loops
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.collect_diagnostics = bool(collect_diagnostics)
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @staticmethod
    def _validate_loops(loops: int) -> int:
        if isinstance(loops, bool) or not isinstance(loops, int) or not 1 <= loops <= 4:
            raise ValueError("loops must be an integer in [1, 4].")
        return loops

    @property
    def loops(self) -> int:
        return self._loops

    @loops.setter
    def loops(self, value: int):
        self._loops = self._validate_loops(value)

    def virtual_schedule(self, loops: Optional[int] = None) -> tuple:
        """Return ((stage, [one-based loop,] block), physical index) entries."""
        k = self.loops if loops is None else self._validate_loops(loops)
        return (tuple((("pre", j), j) for j in range(3))
                + tuple((("core", r, j), 3 + j) for r in range(1, k + 1) for j in range(6))
                + tuple((("coda", k, j), 9 + j) for j in range(3)))

    @property
    def schedule(self) -> tuple[int, ...]:
        return tuple(index for _, index in self.virtual_schedule())

    @property
    def virtual_keys(self) -> tuple:
        return tuple(key for key, _ in self.virtual_schedule())

    def _post(self, block, io, mixed, context, context_mask):
        if context is None:
            return self._apply_expert_post_block(block, io[3], mixed, *io[4:8], None)
        return self._apply_expert_post_block_tensor(block, io[3], mixed, *io[4:8], context, context_mask)

    def _joint_block(self, physical_index, video_tokens, action_tokens, video_freqs,
                     action_freqs, video_t_mod, action_t_mod, video_context,
                     video_context_mask, action_context, action_context_mask, attention_mask):
        video_block = self.mixtures["video"].blocks[physical_index]
        action_block = self.mixtures["action"].blocks[physical_index]
        video_io = self._build_expert_attention_io(
            self.mixtures["video"], video_block, video_tokens, video_freqs, video_t_mod)
        action_io = self._build_expert_attention_io(
            self.mixtures["action"], action_block, action_tokens, action_freqs, action_t_mod)
        mixed = self._mixed_attention(
            torch.cat((video_io[0], action_io[0]), dim=1),
            torch.cat((video_io[1], action_io[1]), dim=1),
            torch.cat((video_io[2], action_io[2]), dim=1), attention_mask)
        if self.version == "v2" and 3 <= physical_index < 9:
            mixed = xsa_projection(mixed, torch.cat((video_io[2], action_io[2]), dim=1), self.num_heads)
        n = video_tokens.shape[1]
        return (self._post(video_block, video_io, mixed[:, :n], video_context, video_context_mask),
                self._post(action_block, action_io, mixed[:, n:], action_context, action_context_mask))

    def _run_pair(self, physical_index, video_tokens, action_tokens, conditioning):
        # Bind the index now: backward recomputation must not capture a loop variable.
        fn = partial(self._joint_block, physical_index)
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            return checkpoint(fn, video_tokens, action_tokens, *conditioning, use_reentrant=False)
        return fn(video_tokens, action_tokens, *conditioning)

    def forward_joint_exits(
        self, video_tokens: torch.Tensor, action_tokens: torch.Tensor,
        video_freqs: torch.Tensor, action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor, action_t_mod: torch.Tensor,
        video_context: torch.Tensor, video_context_mask: torch.Tensor,
        action_context: torch.Tensor, action_context_mask: torch.Tensor,
        attention_mask: torch.Tensor, loops: Optional[int] = None,
        exits: Optional[Sequence[int]] = None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        self.last_diagnostics = {}
        k = self.loops if loops is None else self._validate_loops(loops)
        if exits is None:
            exits = (k,) if self.version == "v0" else tuple(range(1, k + 1))
        exits = tuple(exits)
        if (not exits or len(set(exits)) != len(exits)
                or any(isinstance(e, bool) or not isinstance(e, int) or not 1 <= e <= k for e in exits)):
            raise ValueError("exits must be distinct integer loop indices within [1, loops].")
        n = video_tokens.shape[1] + action_tokens.shape[1]
        if attention_mask.shape[-2:] != (n, n):
            raise ValueError("Joint attention mask must match the combined token sequence.")
        if self.compile_training_layers:
            raise ValueError("LoopMoT layer compilation is not enabled; compile verified fixed-K callables explicitly.")
        conditioning = (video_freqs, action_freqs, video_t_mod, action_t_mod,
                        video_context, video_context_mask, action_context, action_context_mask, attention_mask)
        state = (video_tokens, action_tokens)
        for i in range(3):
            state = self._run_pair(i, *state, conditioning)
        result = {}
        for r in range(1, max(exits) + 1):
            for i in range(3, 9):
                state = self._run_pair(i, *state, conditioning)
            if self.collect_diagnostics:
                # Outside checkpointed blocks, so backward recomputation cannot
                # mutate diagnostics or retain a second autograd graph.
                for name, hidden in zip(("video", "action"), state):
                    self.last_diagnostics[f"loop/{r}/{name}_state_rms"] = hidden.detach().float().square().mean().sqrt()
            if r in exits:
                decoded = state
                for i in range(9, 12):
                    decoded = self._run_pair(i, *decoded, conditioning)
                result[r] = decoded
        return result

    def forward_joint_core(
        self, video_tokens: torch.Tensor, action_tokens: torch.Tensor,
        video_freqs: torch.Tensor, action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor, action_t_mod: torch.Tensor,
        video_context: torch.Tensor, video_context_mask: torch.Tensor,
        action_context: torch.Tensor, action_context_mask: torch.Tensor,
        attention_mask: torch.Tensor, loops: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.loops if loops is None else self._validate_loops(loops)
        return self.forward_joint_exits(
            video_tokens, action_tokens, video_freqs, action_freqs, video_t_mod, action_t_mod,
            video_context, video_context_mask, action_context, action_context_mask,
            attention_mask, loops=k, exits=(k,))[k]

    def prefill_video_cache_tensor(
        self, video_tokens: torch.Tensor, video_freqs: torch.Tensor, video_t_mod: torch.Tensor,
        video_context: torch.Tensor, video_context_mask: torch.Tensor,
        video_attention_mask: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        expert = self.mixtures["video"]
        x = video_tokens
        keys, values = [], []
        if video_attention_mask.shape[-2:] != (x.shape[1], x.shape[1]):
            raise ValueError("Video attention mask must match the video token sequence.")
        for _, i in self.virtual_schedule():
            block = expert.blocks[i]
            io = self._build_expert_attention_io(expert, block, x, video_freqs, video_t_mod)
            mixed = self._mixed_attention(*io[:3], video_attention_mask)
            if self.version == "v2" and 3 <= i < 9:
                mixed = xsa_projection(mixed, io[2], self.num_heads)
            keys.append(io[1])
            values.append(io[2])
            x = self._post(block, io, mixed, video_context, video_context_mask)
        return keys, values

    def forward_action_with_video_cache_tensor(
        self, action_tokens: torch.Tensor, action_freqs: torch.Tensor, action_t_mod: torch.Tensor,
        action_context: torch.Tensor, action_context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor], video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        schedule = self.virtual_schedule()
        if len(video_cache_k) != len(schedule) or len(video_cache_v) != len(schedule):
            raise ValueError(f"Video cache must contain {len(schedule)} virtual layers for K={self.loops}.")
        n = action_tokens.shape[1]
        nv = video_cache_k[0].shape[1]
        if action_attention_mask.shape[-2:] != (n, nv + n):
            raise ValueError("Action attention mask must have action query rows and video/action key columns.")
        expert = self.mixtures["action"]
        x = action_tokens
        for slot, (_, i) in enumerate(schedule):
            block = expert.blocks[i]
            io = self._build_expert_attention_io(expert, block, x, action_freqs, action_t_mod)
            mixed = self._mixed_attention(io[0], torch.cat((video_cache_k[slot], io[1]), dim=1),
                                          torch.cat((video_cache_v[slot], io[2]), dim=1), action_attention_mask)
            if self.version == "v2" and 3 <= i < 9:
                # Queries are action tokens; their own values exclude cached video.
                mixed = xsa_projection(mixed, io[2], self.num_heads)
            x = self._post(block, io, mixed, action_context, action_context_mask)
        return x

    @staticmethod
    def _legacy_context(payload, tokens):
        if payload is None or payload.get("context") is None:
            return None, None
        context = payload["context"]
        mask = payload.get("mask")
        if mask is None:
            mask = torch.ones(tokens.shape[0], tokens.shape[1], context.shape[1],
                              dtype=torch.bool, device=tokens.device)
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        return context, mask

    def forward(self, embeds_all, attention_mask, freqs_all, context_all, t_mod_all):
        video_context, video_mask = self._legacy_context(context_all.get("video"), embeds_all["video"])
        action_context, action_mask = self._legacy_context(context_all.get("action"), embeds_all["action"])
        video, action = self.forward_joint_core(
            embeds_all["video"], embeds_all["action"], freqs_all["video"], freqs_all["action"],
            t_mod_all["video"], t_mod_all["action"], video_context, video_mask,
            action_context, action_mask, attention_mask)
        return {"video": video, "action": action}

    def prefill_video_cache(self, video_tokens, video_freqs, video_t_mod,
                            video_context_payload, video_attention_mask):
        context, mask = self._legacy_context(video_context_payload, video_tokens)
        keys, values = self.prefill_video_cache_tensor(
            video_tokens, video_freqs, video_t_mod, context, mask, video_attention_mask)
        return [{"k": k, "v": v} for k, v in zip(keys, values)]

    def forward_action_with_video_cache(self, action_tokens, action_freqs, action_t_mod,
                                        action_context_payload, video_kv_cache,
                                        attention_mask, video_seq_len):
        context, mask = self._legacy_context(action_context_payload, action_tokens)
        return self.forward_action_with_video_cache_tensor(
            action_tokens, action_freqs, action_t_mod, context, mask,
            [entry["k"] for entry in video_kv_cache], [entry["v"] for entry in video_kv_cache],
            attention_mask[video_seq_len:, :])
