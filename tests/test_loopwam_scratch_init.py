import math

import pytest
import torch

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.loopwam_init import _wan_scratch_state
from fastwam.models.wan22.wan_video_dit import WanVideoDiT


def small_video():
    with torch.device('meta'):
        return WanVideoDiT(hidden_dim=256, ffn_dim=512, in_dim=16, out_dim=16, patch_size=(1, 2, 2),
                           has_image_input=False, seperated_timestep=True, text_dim=64, freq_dim=32, eps=1e-6,
                           num_heads=2, attn_head_dim=128, num_layers=2, video_attention_mask_mode='first_frame_causal')


def test_scratch_state_follows_wan_recipe_and_is_seeded():
    video = small_video()
    state = _wan_scratch_state(video, torch.Generator().manual_seed(0), zero_weight=('head.head',))
    assert set(state) == set(video.state_dict())
    assert not state['head.head.weight'].any() and not state['head.head.bias'].any()
    assert torch.equal(state['blocks.0.norm3.weight'], torch.ones(256))
    w = state['blocks.0.ffn.0.weight']
    assert w.std().item() == pytest.approx(math.sqrt(2 / (w.shape[0] + w.shape[1])), rel=.05)
    assert state['text_embedding.0.weight'].std().item() == pytest.approx(.02, rel=.05)
    assert state['blocks.0.modulation'].std().item() == pytest.approx(256 ** -.5, rel=.1)
    again = _wan_scratch_state(small_video(), torch.Generator().manual_seed(0), zero_weight=('head.head',))
    assert all(torch.equal(state[k], again[k]) for k in state)


def test_action_scratch_state_skips_new_parameters():
    with torch.device('meta'):
        action = ActionDiT(hidden_dim=128, ffn_dim=256, action_dim=7, text_dim=64, freq_dim=32, eps=1e-6,
                           num_heads=1, attn_head_dim=128, num_layers=1)
    state = _wan_scratch_state(action, torch.Generator().manual_seed(0), skip=('action_encoder', 'head'))
    assert not any(k.startswith(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES) for k in state)
    assert set(state) == {k for k in action.state_dict() if not k.startswith(ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES)}
