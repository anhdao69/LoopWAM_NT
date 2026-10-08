"""Two-expert depth recurrence with shared exit decoding and virtual KV caches."""
from __future__ import annotations

from functools import partial
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .mot import MoT
from .wan_video_dit import flash_attention


def validate_action_kv_mode(mode: str, version: str) -> str:
    if mode not in {"aligned", "concat", "mix"}:
        raise ValueError("action_kv_mode must be aligned, concat, or mix")
    if mode != "aligned" and version != "v0":
        raise ValueError("All-loop action KV supports v0 only")
    return mode


def resolve_loop_count(version: str, loops: Optional[int] = None) -> int:
    """Resolve architecture depth; Dense controls always execute their blocks once."""
    if version not in {"v0", "v1", "v2", "dense_s12", "dense_s30"}:
        raise ValueError(f"Unsupported LoopMoT version: {version}")
    if loops is None:
        loops = 1 if version in {"dense_s12", "dense_s30"} else 4
    if isinstance(loops, bool) or not isinstance(loops, int) or not 1 <= loops <= 4:
        raise ValueError("loops must be an integer in [1, 4].")
    if version in {"dense_s12", "dense_s30"} and loops != 1:
        raise ValueError(f"{version} requires exactly one pass (loops=1).")
    return loops


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


def structured_mixed_attention(
    q_video: torch.Tensor, k_video: torch.Tensor, v_video: torch.Tensor,
    q_action: torch.Tensor, k_action: torch.Tensor, v_action: torch.Tensor,
    observation_tokens: int, num_heads: int,
) -> torch.Tensor:
    """Exact canonical F/U/A attention using mask-free SDPA calls.

    Callers must establish the canonical mask first: observed video reads only
    observed video, future video reads all video, and actions read observed video
    plus actions. Text cross-attention and XSA are outside this operation.
    """
    n = observation_tokens
    if isinstance(n, bool) or not isinstance(n, int) or not 0 < n <= q_video.shape[1]:
        raise ValueError("observation_tokens must be a positive video-prefix length.")
    observed = flash_attention(q_video[:, :n], k_video[:, :n], v_video[:, :n], num_heads)
    parts = [observed]
    if q_video.shape[1] > n:
        parts.append(flash_attention(q_video[:, n:], k_video, v_video, num_heads))
    parts.append(flash_attention(q_action, torch.cat((k_video[:, :n], k_action), dim=1),
                                torch.cat((v_video[:, :n], v_action), dim=1), num_heads))
    return torch.cat(parts, dim=1)


class LoopMoT(MoT):
    """Shared-depth models use pre(3), core(6)*K, coda(3).

    Native-layer Dense-S30 uses 30 distinct pairs, grouped 3/24/3, once.

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

    def __init__(self, mixtures: Dict[str, nn.Module], loops: Optional[int] = None,
                 version: str = "v0", checkpoint_blocks: bool = False,
                 mot_checkpoint_mixed_attn: bool = False,
                 collect_diagnostics: bool = False, action_loops: Optional[int] = None,
                 action_kv_mode: str = "aligned"):
        if set(mixtures) != {"video", "action"}:
            raise ValueError("LoopMoT requires exactly the video and action experts.")
        loops = resolve_loop_count(version, loops)
        # Canonical token order is always video followed by action.
        super().__init__({name: mixtures[name] for name in ("video", "action")},
                         mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn)
        self.unique_depth = 30 if version == "dense_s30" else 12
        self.core_depth = 24 if version == "dense_s30" else 6
        if self.num_layers != self.unique_depth:
            raise ValueError(f"{version} requires {self.unique_depth} physical blocks per expert.")
        self.version = version
        self.action_kv_mode = validate_action_kv_mode(action_kv_mode, version)
        self.trained_max_loops = 1 if version in {"dense_s12", "dense_s30"} else 4
        self.loops = loops
        self.action_loops = action_loops
        if self.action_kv_mode == "mix":
            self.action_kv_logits = nn.Parameter(torch.zeros(
                self.core_depth, self.num_heads, loops, dtype=torch.float32,
                device=next(self.mixtures["video"].parameters()).device))
        self.checkpoint_blocks = bool(checkpoint_blocks)
        self.collect_diagnostics = bool(collect_diagnostics)
        self.last_diagnostics: dict[str, torch.Tensor] = {}
        # Explicit runtime optimization; model weights and mask semantics do not change.
        self.structured_attention = False
        self.structured_attention_observation_tokens: Optional[int] = None
        self.record_attention_mass = False
        self.last_attention_mass = {}

    def _validate_loops(self, loops: int) -> int:
        return resolve_loop_count(self.version, loops)

    @property
    def loops(self) -> int:
        return self._loops

    @loops.setter
    def loops(self, value: int):
        self._loops = self._validate_loops(value)

    @property
    def action_loops(self) -> int:
        return self.loops if self._action_loops is None else self._action_loops

    @action_loops.setter
    def action_loops(self, value):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= self.trained_max_loops:
                raise ValueError("action_loops must be an integer within the architecture loop capacity")
            if value != self.loops and self.version != "v0":
                raise ValueError("Asymmetric depth currently supports v0 only")
        self._action_loops = value

    def action_cache_slots(self) -> tuple[int, ...]:
        """Late alignment clamped at video loop one when action is deeper."""
        slots = {key: slot for slot, (key, _) in enumerate(self.virtual_schedule())}
        return (tuple(slots[("pre", j)] for j in range(self.pre_depth))
                + tuple(slots[("core", max(1, self.loops-self.action_loops+r), j)]
                        for r in range(1, self.action_loops+1) for j in range(self.core_depth))
                + tuple(slots[("coda", self.loops, j)] for j in range(self.post_depth)))

    def action_cache_slot_sets(self) -> tuple[tuple[int, ...], ...]:
        """One set per action block; core sets enumerate video loops in order."""
        schedule = self.virtual_schedule()
        lookup = {key: slot for slot, (key, _) in enumerate(schedule)}
        result = []
        for slot in self.action_cache_slots():
            key, _ = schedule[slot]
            if self.action_kv_mode != "aligned" and key[0] == "core":
                result.append(tuple(lookup[("core", v, key[2])]
                                    for v in range(1, self.loops + 1)))
            else:
                result.append((slot,))
        return tuple(result)

    def action_conditioning_kv(self, keys, values, slots, physical_index, mask):
        """Combine already-RoPE'd observation keys. Never rotate cache entries again."""
        core = self.pre_depth <= physical_index < self.pre_depth + self.core_depth
        if len(slots) == 1 and not (core and self.action_kv_mode == "mix"):
            return keys[slots[0]], values[slots[0]], mask
        if not core or len(slots) != self.loops:
            raise ValueError("All-loop conditioning requires one matching core slot per video loop")
        nobs = keys[slots[0]].shape[1]
        if self.action_kv_mode == "concat":
            expanded = torch.cat([mask[:, :nobs]] * len(slots) + [mask[:, nobs:]], dim=1)
            return (torch.cat([keys[s] for s in slots], dim=1),
                    torch.cat([values[s] for s in slots], dim=1), expanded)
        if self.action_kv_mode != "mix":
            raise ValueError("Aligned action conditioning requires a single slot")
        weights = self.action_kv_logits[physical_index - self.pre_depth].float().softmax(-1)
        def fuse(cache):
            tensors = torch.stack([cache[s] for s in slots])
            k, b, n, width = tensors.shape
            # FP32 weights and accumulation, including under BF16 autocast.
            with torch.autocast(device_type=tensors.device.type, enabled=False):
                heads = tensors.float().reshape(k, b, n, self.num_heads, width // self.num_heads)
                fused = (heads * weights.transpose(0, 1)[:, None, None, :, None]).sum(0)
            return fused.reshape(b, n, width).to(tensors.dtype)
        return fuse(keys), fuse(values), mask

    def _action_cached_block(self, index, slots, x, freqs, t_mod, context,
                             context_mask, keys, values, mask):
        block = self.mixtures["action"].blocks[index]
        io = self._build_expert_attention_io(self.mixtures["action"], block, x, freqs, t_mod)
        key, value, expanded = self.action_conditioning_kv(keys, values, slots, index, mask)
        key = torch.cat((key, io[1]), dim=1)
        value = torch.cat((value, io[2]), dim=1)
        mixed = self._mixed_attention(io[0], key, value, expanded)
        if self.record_attention_mass and self.pre_depth <= index < self.pre_depth + self.core_depth:
            self._record_action_attention_mass(index, io[0], key, expanded, keys[0].shape[1], len(slots))
        return self._post(block, io, mixed, context, context_mask)

    def _record_action_attention_mass(self, index, query, key, mask, nobs, copies):
        if torch.is_grad_enabled() or self.action_kv_mode != "concat":
            raise ValueError("Attention-mass diagnostics require concat and disabled gradients")
        b, n, width = query.shape
        head_dim = width // self.num_heads
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = query.float().reshape(b, n, self.num_heads, head_dim).transpose(1, 2)
            k = key.float().reshape(b, -1, self.num_heads, head_dim).transpose(1, 2)
            scores = (q @ k.transpose(-1, -2)) * (head_dim ** -.5)
            probabilities = scores.masked_fill(~mask, float('-inf')).softmax(-1)
            fractions = [probabilities[..., v*nobs:(v+1)*nobs].sum(-1).mean()
                         for v in range(copies)]
            fractions.append(probabilities[..., copies*nobs:].sum(-1).mean())
        self.last_attention_mass.setdefault(index - self.pre_depth, []).append(torch.stack(fractions))

    def _forward_action_slot_sets(self, action_tokens, action_freqs, action_t_mod,
                                 action_context, action_context_mask, keys, values, mask,
                                 slot_sets=None, checkpoint_action=False):
        schedule = self.virtual_schedule()
        if len(keys) != len(schedule) or len(values) != len(schedule):
            raise ValueError("Video cache must contain every virtual layer")
        if mask.ndim != 2 or mask.dtype != torch.bool or mask.shape != (action_tokens.shape[1], keys[0].shape[1] + action_tokens.shape[1]):
            raise ValueError("Action mask must contain observation-prefix and action columns")
        slot_sets = self.action_cache_slot_sets() if slot_sets is None else slot_sets
        if len(slot_sets) != self.pre_depth + self.core_depth*self.action_loops + self.post_depth:
            raise ValueError("Slot sets must cover every action block exactly once")
        x = action_tokens
        action_states = []
        waiting_state = None
        for number, slots in enumerate(slot_sets):
            if not slots or any(s < 0 or s >= len(schedule) for s in slots):
                raise ValueError("Invalid action cache slots")
            index = schedule[slots[0]][1]
            if any(schedule[s][1] != index for s in slots):
                raise ValueError("Action KV may combine only the same physical video block")
            fn = partial(self._action_cached_block, index, slots)
            args = (x, action_freqs, action_t_mod, action_context, action_context_mask, keys, values, mask)
            x = (checkpoint(fn, *args, use_reentrant=False)
                 if checkpoint_action and self.training and torch.is_grad_enabled() else fn(*args))
            if self.collect_diagnostics:
                if number == self.pre_depth - 1:
                    waiting_state = x.detach().float().square().mean().sqrt()
                if index == self.pre_depth + self.core_depth - 1:
                    action_states.append(x.detach().float().square().mean().sqrt())
        if self.collect_diagnostics:
            for r in range(1, self.loops + 1):
                action_r = max(0, r - (self.loops - self.action_loops))
                self.last_diagnostics[f"loop/{r}/action_state_rms"] = (
                    waiting_state if action_r == 0 else action_states[action_r - 1])
        return x

    def forward_cached_training(self, video_tokens, action_tokens, video_freqs, action_freqs,
                                video_t_mod, action_t_mod, video_context, video_context_mask,
                                action_context, action_context_mask, attention_mask,
                                observed=None, slot_sets=None, cache_observer=None):
        """Differentiable full-video pass followed by mode-specific action KV access.

        Aligned can explicitly use this path for equivalence tests. Its production
        joint/asymmetric paths remain unchanged. Cache entries are prefix views,
        never detached, so all selected loops receive direct action gradients.
        """
        nv = video_tokens.shape[1]
        if attention_mask.ndim != 2 or attention_mask.dtype != torch.bool:
            raise ValueError("Cached training requires a 2D boolean causal mask")
        if self.structured_attention:
            observed = self._validate_structured_mask(attention_mask, nv)
        nobs = observed if observed is not None else int(attention_mask[nv, :nv].sum().item())
        if not 0 < nobs <= nv:
            raise ValueError("Action conditioning needs a nonempty observation prefix")
        torch._assert_async(~attention_mask[:nv, nv:].any(), "Video cannot read actions")
        torch._assert_async(~attention_mask[:nobs, nobs:nv].any(), "Observation tokens cannot read future video")
        expected = torch.arange(nv, device=attention_mask.device) < nobs
        torch._assert_async((attention_mask[nv:, :nv] == expected).all(), "Actions must read only the observation prefix")
        self.last_diagnostics = {}
        x = video_tokens
        keys, values = [], []
        for virtual, index in self.virtual_schedule():
            fn = partial(self._video_only_block, index)
            args = (x, video_freqs, video_t_mod, video_context, video_context_mask,
                    attention_mask[:nv, :nv], observed, True)
            x, key, value = (checkpoint(fn, *args, use_reentrant=False)
                            if self.checkpoint_blocks and self.training and torch.is_grad_enabled() else fn(*args))
            keys.append(key[:, :nobs]); values.append(value[:, :nobs])
            if self.collect_diagnostics and virtual[0] == "core" and virtual[2] == self.core_depth - 1:
                self.last_diagnostics[f"loop/{virtual[1]}/video_state_rms"] = x.detach().float().square().mean().sqrt()
        if cache_observer is not None:
            cache_observer(keys, values)
        mask = torch.cat((attention_mask[nv:, :nobs], attention_mask[nv:, nv:]), dim=1)
        action = self._forward_action_slot_sets(action_tokens, action_freqs, action_t_mod,
            action_context, action_context_mask, keys, values, mask, slot_sets, self.checkpoint_blocks)
        return x, action

    def _video_only_block(self, physical_index, tokens, freqs, t_mod, context, context_mask,
                          attention_mask, observed=None, return_cache=False):
        expert = self.mixtures["video"]
        block = expert.blocks[physical_index]
        io = self._build_expert_attention_io(expert, block, tokens, freqs, t_mod)
        if observed is None:
            mixed = self._mixed_attention(*io[:3], attention_mask)
        else:
            q, k, v = io[:3]
            mixed = torch.cat((flash_attention(q[:, :observed], k[:, :observed], v[:, :observed], self.num_heads),
                               flash_attention(q[:, observed:], k, v, self.num_heads)), dim=1)
        updated = self._post(block, io, mixed, context, context_mask)
        return (updated, io[1], io[2]) if return_cache else updated

    def _forward_action_deeper(self, video_tokens, action_tokens, video_freqs, action_freqs,
                              video_t_mod, action_t_mod, video_context, video_context_mask,
                              action_context, action_context_mask, attention_mask, observed):
        """Differentiable video prefill, then deeper action execution with reused KV.

        Every KV remains connected to the video graph. No future tokens are
        exposed to actions; repeated use accumulates gradients into the same KV.
        """
        nv = video_tokens.shape[1]
        nobs = observed if observed is not None else int(attention_mask[nv, :nv].sum().item())
        if not 0 < nobs <= nv:
            raise ValueError("Action conditioning needs a nonempty observed video prefix")
        torch._assert_async(~attention_mask[:nv, nv:].any(), "Video cannot read actions")
        expected = torch.arange(nv, device=attention_mask.device) < nobs
        torch._assert_async((attention_mask[nv:, :nv] == expected).all(),
                            "Actions must read only the contiguous observation prefix")
        x = video_tokens
        keys, values = [], []
        for _, index in self.virtual_schedule():
            fn = partial(self._video_only_block, index)
            args = (x, video_freqs, video_t_mod, video_context, video_context_mask,
                    attention_mask[:nv, :nv], observed, True)
            x, key, value = (checkpoint(fn, *args, use_reentrant=False)
                            if self.checkpoint_blocks and self.training and torch.is_grad_enabled()
                            else fn(*args))
            keys.append(key[:, :nobs]); values.append(value[:, :nobs])
        mask = torch.cat((attention_mask[nv:, :nobs], attention_mask[nv:, nv:]), dim=1)
        action = self.forward_action_with_video_cache_tensor(action_tokens, action_freqs, action_t_mod,
                     action_context, action_context_mask, keys, values, mask)
        return x, action

    def virtual_schedule(self, loops: Optional[int] = None) -> tuple:
        """Return ((stage, [one-based loop,] block), physical index) entries."""
        k = self.loops if loops is None else self._validate_loops(loops)
        return (tuple((("pre", j), j) for j in range(self.pre_depth))
                + tuple((("core", r, j), self.pre_depth + j)
                        for r in range(1, k + 1) for j in range(self.core_depth))
                + tuple((("coda", k, j), self.pre_depth + self.core_depth + j)
                        for j in range(self.post_depth)))

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
                     video_context_mask, action_context, action_context_mask, attention_mask,
                     structured_observation_tokens=None):
        video_block = self.mixtures["video"].blocks[physical_index]
        action_block = self.mixtures["action"].blocks[physical_index]
        video_io = self._build_expert_attention_io(
            self.mixtures["video"], video_block, video_tokens, video_freqs, video_t_mod)
        action_io = self._build_expert_attention_io(
            self.mixtures["action"], action_block, action_tokens, action_freqs, action_t_mod)
        if structured_observation_tokens is None:
            mixed = self._mixed_attention(
                torch.cat((video_io[0], action_io[0]), dim=1),
                torch.cat((video_io[1], action_io[1]), dim=1),
                torch.cat((video_io[2], action_io[2]), dim=1), attention_mask)
        else:
            mixed = structured_mixed_attention(*video_io[:3], *action_io[:3],
                observation_tokens=structured_observation_tokens, num_heads=self.num_heads)
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

    def _validate_structured_mask(self, attention_mask, video_tokens):
        observed = self.structured_attention_observation_tokens
        if (isinstance(observed, bool) or not isinstance(observed, int)
                or not 0 < observed <= video_tokens):
            raise ValueError("Set structured_attention_observation_tokens to the clean observation-prefix length.")
        if attention_mask.ndim != 2 or attention_mask.dtype != torch.bool:
            raise ValueError("Structured attention requires the canonical 2D boolean mask.")
        expected = torch.zeros_like(attention_mask)
        expected[:observed, :observed] = True
        expected[observed:video_tokens, :video_tokens] = True
        expected[video_tokens:, :observed] = True
        expected[video_tokens:, video_tokens:] = True
        # Validate once per forward, outside checkpoint recomputation. On CUDA
        # this check stays on device rather than synchronizing with the host.
        torch._assert_async((attention_mask == expected).all(),
                            "Structured attention requires the canonical F/U/A mask.")
        return observed

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
            exits = (k,) if self.version in {"v0", "dense_s12", "dense_s30"} else tuple(range(1, k + 1))
        exits = tuple(exits)
        if (not exits or len(set(exits)) != len(exits)
                or any(isinstance(e, bool) or not isinstance(e, int) or not 1 <= e <= k for e in exits)):
            raise ValueError("exits must be distinct integer loop indices within [1, loops].")
        asymmetric = self.action_loops != self.loops
        if asymmetric and (k != self.loops or exits != (k,)):
            raise ValueError("Asymmetric v0 requires its configured final video exit")
        n = video_tokens.shape[1] + action_tokens.shape[1]
        if attention_mask.shape[-2:] != (n, n):
            raise ValueError("Joint attention mask must match the combined token sequence.")
        if self.compile_training_layers:
            raise ValueError("LoopMoT layer compilation is not enabled; compile verified fixed-K callables explicitly.")
        observed = (self._validate_structured_mask(attention_mask, video_tokens.shape[1])
                    if self.structured_attention else None)
        if self.action_kv_mode != "aligned":
            if k != self.loops or exits != (k,):
                raise ValueError("All-loop action KV requires its configured final video exit")
            return {k: self.forward_cached_training(video_tokens, action_tokens, video_freqs, action_freqs,
                video_t_mod, action_t_mod, video_context, video_context_mask, action_context,
                action_context_mask, attention_mask, observed)}
        if asymmetric and self.action_loops > k:
            return {k: self._forward_action_deeper(video_tokens, action_tokens, video_freqs, action_freqs,
                video_t_mod, action_t_mod, video_context, video_context_mask, action_context,
                action_context_mask, attention_mask, observed)}
        conditioning = (video_freqs, action_freqs, video_t_mod, action_t_mod,
                        video_context, video_context_mask, action_context, action_context_mask,
                        attention_mask, observed)
        state = (video_tokens, action_tokens)
        for i in range(self.pre_depth):
            state = self._run_pair(i, *state, conditioning)
        result = {}
        for r in range(1, max(exits) + 1):
            for i in range(self.pre_depth, self.pre_depth + self.core_depth):
                if asymmetric and r <= k - self.action_loops:
                    video_args = (state[0], video_freqs, video_t_mod, video_context,
                                  video_context_mask, attention_mask[:video_tokens.shape[1], :video_tokens.shape[1]], observed)
                    fn = partial(self._video_only_block, i)
                    updated = (checkpoint(fn, *video_args, use_reentrant=False)
                               if self.checkpoint_blocks and self.training and torch.is_grad_enabled()
                               else fn(*video_args))
                    state = (updated, state[1])
                else:
                    state = self._run_pair(i, *state, conditioning)
            if self.collect_diagnostics:
                # Outside checkpointed blocks, so backward recomputation cannot
                # mutate diagnostics or retain a second autograd graph.
                for name, hidden in zip(("video", "action"), state):
                    self.last_diagnostics[f"loop/{r}/{name}_state_rms"] = hidden.detach().float().square().mean().sqrt()
            if r in exits:
                decoded = state
                for i in range(self.pre_depth + self.core_depth, self.unique_depth):
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
        if self.action_kv_mode != "aligned":
            return self._forward_action_slot_sets(action_tokens, action_freqs, action_t_mod,
                action_context, action_context_mask, video_cache_k, video_cache_v, action_attention_mask)
        schedule = self.virtual_schedule()
        if len(video_cache_k) != len(schedule) or len(video_cache_v) != len(schedule):
            raise ValueError(f"Video cache must contain {len(schedule)} virtual layers for K={self.loops}.")
        n = action_tokens.shape[1]
        nv = video_cache_k[0].shape[1]
        if action_attention_mask.shape[-2:] != (n, nv + n):
            raise ValueError("Action attention mask must have action query rows and video/action key columns.")
        expert = self.mixtures["action"]
        x = action_tokens
        for slot in self.action_cache_slots():
            _, i = schedule[slot]
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
