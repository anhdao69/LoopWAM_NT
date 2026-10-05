"""Exact F/U/A attention decomposition against the retained masked reference."""
import copy
from contextlib import nullcontext

import pytest
import torch

from fastwam.models.wan22.wan_video_dit import flash_attention
from test_loop_mot import cached_action, inputs, make_model


def test_structured_attention_is_explicitly_opt_in():
    model = make_model()
    assert model.structured_attention is False
    assert model.structured_attention_observation_tokens is None


def canonical_mask(observed, video, action, device):
    mask = torch.zeros(video + action, video + action, dtype=torch.bool, device=device)
    mask[:observed, :observed] = True
    mask[observed:video, :video] = True
    mask[video:, :observed] = True
    mask[video:, video:] = True
    return mask


def relative_error(actual, expected):
    return (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-6)


@pytest.mark.parametrize('device,dtype', [('cpu', torch.float32), ('cpu', torch.bfloat16),
    pytest.param('cuda', torch.bfloat16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA acceptance'))])
@pytest.mark.parametrize('video_tokens', [2, 6])
def test_three_attention_calls_match_masked_output_and_all_input_gradients(device, dtype, video_tokens):
    from fastwam.models.wan22.loop_mot import structured_mixed_attention
    torch.manual_seed(81)
    # GPU gate also exercises the production 12 x 128 attention head geometry.
    heads, width = (12, 1536) if device == 'cuda' else (2, 12)
    data = [torch.randn(2, n, width, device=device, dtype=dtype, requires_grad=True)
            for n in (video_tokens,) * 3 + (3,) * 3]
    reference = [x.detach().clone().requires_grad_() for x in data]
    actual = structured_mixed_attention(*data, observation_tokens=2, num_heads=heads)
    expected = flash_attention(*(torch.cat((reference[i], reference[i+3]), dim=1) for i in range(3)),
        num_heads=heads, ctx_mask=canonical_mask(2, video_tokens, 3, device))
    tolerance = 2e-6 if dtype == torch.float32 else .025
    assert relative_error(actual, expected) < tolerance
    weight = torch.randn_like(actual).float()
    (actual.float() * weight).sum().backward()
    (expected.float() * weight).sum().backward()
    for x, ref in zip(data, reference):
        assert torch.isfinite(x.grad).all()
        assert relative_error(x.grad, ref.grad) < tolerance * 2


@pytest.mark.parametrize('version', ['v0', 'v2'])
@pytest.mark.parametrize('device,dtype', [('cpu', torch.float32), ('cpu', torch.bfloat16),
    pytest.param('cuda', torch.bfloat16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA acceptance'))])
def test_structured_joint_cache_checkpoint_outputs_and_parameter_gradients(version, device, dtype):
    reference = make_model(version).to(device)
    structured, checked, cached = (copy.deepcopy(reference) for _ in range(3))
    for model in (structured, checked, cached):
        model.structured_attention = True
        model.structured_attention_observation_tokens = 2
    checked.checkpoint_blocks = True
    data = {key: value.to(device) for key, value in inputs().items()}
    with torch.autocast(device, dtype=dtype) if dtype != torch.float32 else nullcontext():
        outputs = [model.forward_joint_core(**data)[1] for model in (reference, structured, checked)]
        outputs.append(cached_action(cached, data))
    tolerance = 5e-5 if dtype == torch.float32 else .04
    for output in outputs[1:]:
        assert relative_error(output, outputs[0]) < tolerance
    for output in outputs:
        output.float().square().mean().backward()
    def gradients(model):
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                          for p in model.parameters()])
    expected = gradients(reference)
    for model in (structured, checked, cached):
        actual = gradients(model)
        assert torch.isfinite(actual).all()
        assert relative_error(actual, expected) < tolerance * 2


def test_structured_attention_rejects_wrong_layout_or_mask():
    model, data = make_model(), inputs()
    model.structured_attention = True
    with pytest.raises(ValueError, match='observation'):
        model.forward_joint_core(**data)
    model.structured_attention_observation_tokens = 2
    changed = dict(data, attention_mask=data['attention_mask'].clone())
    changed['attention_mask'][4, 2] = True  # Action now reads future video.
    with pytest.raises(RuntimeError, match='canonical'):
        model.forward_joint_core(**changed)
    model.structured_attention_observation_tokens = 5
    with pytest.raises(ValueError, match='observation'):
        model.forward_joint_core(**data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA production-shape FlashAttention gate')
def test_cuda_native_shape_structured_flash_attention_matches_masked_reference():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from fastwam.models.wan22.loop_mot import structured_mixed_attention
    torch.manual_seed(82)
    data = [torch.randn(1, n, 1536, device='cuda', dtype=torch.bfloat16, requires_grad=True)
            for n in (1176, 1176, 1176, 32, 32, 32)]
    reference = [x.detach().clone().requires_grad_() for x in data]
    # Force the advertised fast backend here: passing numerics via a fallback
    # would not establish that the production-shape optimization is executable.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        actual = structured_mixed_attention(*data, observation_tokens=392, num_heads=12)
    expected = flash_attention(*(torch.cat((reference[i], reference[i+3]), dim=1) for i in range(3)),
        num_heads=12, ctx_mask=canonical_mask(392, 1176, 32, 'cuda'))
    assert relative_error(actual, expected) < .02
    weight = torch.randn_like(actual).float()
    (actual.float() * weight).sum().backward()
    (expected.float() * weight).sum().backward()
    for x, ref in zip(data, reference):
        assert torch.isfinite(x.grad).all()
        assert relative_error(x.grad, ref.grad) < .04
