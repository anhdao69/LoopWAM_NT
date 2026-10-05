"""Cache integration preserves noisy training inputs, losses and policy gradients."""
import copy
from types import MethodType

import pytest
import torch

from test_loopwam_policy import make_policy


def install_frozen_encoder(policy):
    encoded_samples = []

    @torch.no_grad()
    def encode(self, video, tiled=False, **kwargs):
        assert not tiled
        encoded_samples.append(video.shape[0])
        # Tiny deterministic stand-in for frozen native VAE output: first frame
        # and every fourth future frame, exactly representable as BF16 values.
        return video[:, :2, ::4, ::16, ::16].to(torch.bfloat16).to(torch.float32)

    policy._encode_video_latents = MethodType(encode, policy)
    return encoded_samples


def sample_and_noise():
    torch.manual_seed(97)
    video = torch.randn(3, 3, 9, 16, 32)
    # Match the production marked dummy: duplicate the first window, but mask
    # every target and keep its training index negative.
    video[2] = video[0]
    action_pad = torch.zeros(3, 8, dtype=torch.bool)
    action_pad[0, 6:] = True
    action_pad[2] = True
    image_pad = torch.zeros(3, 9, dtype=torch.bool)
    image_pad[0, 5:] = True
    image_pad[2] = True
    sample = dict(video=video, action=torch.randn(3, 8, 3), proprio=torch.randn(3, 8, 2),
        context=torch.randn(3, 3, 8), context_mask=torch.tensor([[True, True, False]] * 3),
        action_is_pad=action_pad, image_is_pad=image_pad,
        training_index=torch.tensor([0, 1, -1]), sample_valid=torch.tensor([True, True, False]))
    fixed = dict(noise_video=torch.randn(3, 2, 3, 1, 2), noise_action=torch.randn(3, 8, 3),
                 timestep_video=torch.tensor([100., 700., 300.]),
                 timestep_action=torch.tensor([800., 200., 400.]))
    return sample, fixed


def evaluate(policy, sample, fixed):
    policy.zero_grad(set_to_none=True)
    noisy = policy.prepare_training_batch(sample, **fixed)
    outputs = policy.forward_exits(noisy)
    loss, logs = policy.reduce_flow_losses(outputs, noisy)
    loss.backward()
    gradients = {name: parameter.grad.detach().clone()
                 for name, parameter in policy.named_parameters() if parameter.requires_grad}
    assert gradients['proprio_encoder.weight'].norm() > 0
    return noisy, outputs, loss.detach(), logs, gradients


def assert_identical(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_identical(actual[key], expected[key])
    elif isinstance(expected, tuple):
        for x, y in zip(actual, expected):
            assert_identical(x, y)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert actual == expected


@pytest.mark.parametrize('version', ['v0', 'v2'])
def test_online_cache_miss_and_hit_preserve_policy_loss_gradients_and_padding(tmp_path, version):
    from fastwam.datasets.loopwam_latent_cache import LoopWAMLatentCache
    online = make_policy(version)
    cached = copy.deepcopy(online)
    online_calls = install_frozen_encoder(online)
    cache_calls = install_frozen_encoder(cached)
    cached.training_latent_cache = LoopWAMLatentCache(tmp_path / version, size=2,
        provenance={'test_contract': 'frozen_bf16_encoder_v1'}, create=True,
        device='cpu', latent_shape=(2, 3, 1, 2))
    sample, fixed = sample_and_noise()
    baseline = evaluate(online, sample, fixed)
    missed = evaluate(cached, sample, fixed)
    assert_identical(missed, baseline)
    assert sum(online_calls) == 3
    assert sum(cache_calls) == 3
    cache_calls.clear()
    hit = evaluate(cached, sample, fixed)
    assert_identical(hit, baseline)
    assert sum(cache_calls) == 1, 'Only the negative-index dummy may be encoded again'
    assert cached.training_latent_cache.stats['hits'] >= 2
    assert torch.equal(hit[0]['action_is_pad'], sample['action_is_pad'])
    assert torch.equal(hit[0]['image_is_pad'], sample['image_is_pad'])
    # Padded future targets cannot change this batch's objective. This includes
    # the partial valid window and the fully padded distributed tail dummy.
    noisy = dict(hit[0], target_action=hit[0]['target_action'].clone(),
                 target_video=hit[0]['target_video'].clone())
    noisy['target_action'][sample['action_is_pad']] += 1000
    noisy['target_video'][0, :, 2:] += 1000
    noisy['target_video'][2] += 1000
    changed_loss, changed_logs = cached.reduce_flow_losses(hit[1], noisy)
    torch.testing.assert_close(changed_loss, baseline[2], rtol=0, atol=0)
    assert_identical(changed_logs, baseline[3])
    # The dummy still occupies a microbatch slot, as expected by the runner's
    # exact tail weighting, while contributing zero to both branch numerators.
    real_noisy = dict(hit[0])
    for name in ('timestep_video', 'timestep_action', 'target_video', 'target_action',
                 'image_is_pad', 'action_is_pad'):
        real_noisy[name] = real_noisy[name][:2]
    real_outputs = {k: tuple(branch[:2] for branch in pair) for k, pair in hit[1].items()}
    real_loss, _ = cached.reduce_flow_losses(real_outputs, real_noisy)
    torch.testing.assert_close(hit[2] * 3, real_loss * 2, rtol=1e-6, atol=1e-6)
    cached.training_latent_cache.close()
