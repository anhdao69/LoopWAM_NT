"""Frozen native VAE feature caches must not outlive clip encoding."""
import copy

import torch
from torch import nn

from fastwam.models.wan22.loopwam import LoopWAM
from fastwam.models.wan22.wan_video_vae import VideoVAE_


def test_releasing_native_vae_caches_preserves_repeated_clip_and_image_encoding():
    torch.manual_seed(57)
    native = VideoVAE_(dim=4, z_dim=2, dim_mult=[1, 2, 4, 4],
                       num_res_blocks=1, temperal_downsample=[False, True, True]).eval()
    reference = copy.deepcopy(native)
    # Exercise the production LoopWAM encoding methods around the real native
    # VAE, without allocating unrelated transformer experts or loading assets.
    policy = LoopWAM.__new__(LoopWAM)
    nn.Module.__init__(policy)
    policy.vae = nn.Module()
    policy.vae.model = native
    policy.vae.scale = [torch.zeros(2), torch.ones(2)]
    policy.vae.requires_grad_(False)
    policy.device = torch.device('cpu')
    policy.torch_dtype = torch.float32
    clip = torch.randn(2, 3, 9, 16, 16)
    other_clip = torch.randn_like(clip)
    for source, image_only in ((clip, False), (other_clip, False),
                               (clip[:, :, :1], True), (clip, False)):
        with torch.no_grad():
            expected = reference.encode(source, policy.vae.scale)
        # The unmodified native method really does retain tensor caches.
        assert any(isinstance(value, torch.Tensor) for value in reference._enc_feat_map)
        if image_only:
            actual = policy._encode_input_image_latents_tensor(source[:, :, 0])
        else:
            actual = policy._encode_video_latents(source)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not actual.requires_grad
        assert all(value is None for value in native._enc_feat_map)
        assert all(value is None for value in native._feat_map)
