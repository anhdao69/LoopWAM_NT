import pytest
import torch
import torch.nn.functional as F

from fastwam.models.wan22.loopwam_init import (
    NATIVE_CONFIG, canonical_ffn_indices, prune_paired_ffn, resize_action_tensor,
    target_configs, validate_source_config, validate_state_coverage, select_donor_layers,
)
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.models.wan22.action_dit import ActionDiT


def test_paired_ffn_matches_explicit_zeroed_neurons():
    torch.manual_seed(13)
    x, w1, b1, w2, b2 = torch.randn(3, 5), torch.randn(11, 5), torch.randn(11), torch.randn(5, 11), torch.randn(5)
    indices = canonical_ffn_indices(11, 7)
    a, b, c, d = prune_paired_ffn(w1, b1, w2, b2, indices)
    activation = F.gelu(F.linear(x, w1, b1), approximate='tanh')
    mask = torch.zeros(11)
    mask[indices] = 1
    expected = F.linear(activation * mask, w2, b2)
    actual = F.linear(F.gelu(F.linear(x, a, b), approximate='tanh'), c, d)
    torch.testing.assert_close(actual, expected)
    native_indices = canonical_ffn_indices()
    assert len(native_indices.unique()) == 6144
    assert native_indices[0] == 0 and native_indices[-1] == 8959


def test_resize_axis_order_and_last_dimension_scaling():
    source = torch.arange(35, dtype=torch.float32).reshape(5, 7)
    expected = F.interpolate(source.T.unsqueeze(1), size=3, mode='linear', align_corners=True).squeeze(1).T
    expected = F.interpolate(expected.unsqueeze(1), size=4, mode='linear', align_corners=True).squeeze(1)
    expected *= (7 / 4) ** 0.5
    torch.testing.assert_close(resize_action_tensor(source, (3, 4)), expected)
    torch.testing.assert_close(resize_action_tensor(torch.arange(7.), (4,)), torch.tensor([0., 2., 4., 6.]))
    assert resize_action_tensor(source, (5, 7)).data_ptr() == source.data_ptr()


def test_coverage_rejects_missing_unexpected_wrong_shape():
    expected = {'a': (2, 3), 'b': (3,)}
    validate_state_coverage({'a': torch.empty(2, 3), 'b': torch.empty(3)}, expected)
    for state in ({'a': torch.empty(2, 3)}, {'a': torch.empty(2, 3), 'b': torch.empty(3), 'c': torch.empty(1)}, {'a': torch.empty(3, 2), 'b': torch.empty(3)}):
        with pytest.raises(ValueError):
            validate_state_coverage(state, expected)


def test_native_config_rejects_cross_family_or_pruned_source():
    validate_source_config(NATIVE_CONFIG)
    for key, value in [('dim', 3072), ('ffn_dim', 6144), ('in_dim', 48), ('num_layers', 12)]:
        with pytest.raises(ValueError):
            validate_source_config({**NATIVE_CONFIG, key: value})


def test_donor_remap_keeps_globals_and_explicit_indices():
    source = {'blocks.%d.weight' % i: torch.tensor(i) for i in range(30)}
    source['global'] = torch.tensor(42)
    selected = select_donor_layers(source)
    assert [selected['blocks.%d.weight' % i].item() for i in range(12)] == [0, 1, 2, 9, 10, 11, 12, 13, 14, 27, 28, 29]
    assert selected['global'].item() == 42


@pytest.mark.parametrize('depth,expected', [(12, 584536135), (30, 1416114247)])
def test_actual_compact_parameter_count_on_meta(depth, expected):
    video_cfg, action_cfg = target_configs(depth)
    with torch.device('meta'):
        video, action = WanVideoDiT(**video_cfg), ActionDiT(**action_cfg)
        proprio = torch.nn.Linear(8, 4096)
    actual = sum(p.numel() for m in (video, action, proprio) for p in m.parameters())
    assert actual == expected


def test_materialized_experts_do_not_alias_donor_or_other_expert_storage():
    from fastwam.models.wan22.loopwam_init import _materialize
    donor = {'weight': torch.ones(2, 3), 'bias': torch.ones(2)}
    with torch.device('meta'):
        first, second = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    _materialize(first, donor, device='cpu', dtype=torch.float32)
    _materialize(second, donor, device='cpu', dtype=torch.float32)
    with torch.no_grad():
        first.weight.zero_()
    assert torch.all(second.weight == 1) and torch.all(donor['weight'] == 1)


def test_artifact_metadata_covers_all_tensors_and_rejects_wrong_ffn_rule():
    from fastwam.models.wan22.loopwam_init import (
        FORMAT_VERSION, SOURCE_MODEL, DONOR_INDICES, validate_artifact, _expected_shapes,
    )
    video_config, action_config = target_configs(30)
    with torch.device('meta'):
        video = WanVideoDiT(**video_config).state_dict()
        action = ActionDiT(**action_config).state_dict()
        proprio = torch.nn.Linear(8, 4096).state_dict()
    target_video, target_action = target_configs(12)
    payload = dict(format_version=FORMAT_VERSION, video_state_dict=video,
                   action_state_dict=action, proprio_state_dict=proprio, metadata={
                       'source_config': NATIVE_CONFIG, 'source_model': SOURCE_MODEL,
                       'source_revision': 'a' * 40, 'source_hashes': {'source': 'b' * 64},
                       'new_parameter_seed': 42,
                       'donor_indices': list(DONOR_INDICES), 'ffn_indices': canonical_ffn_indices().tolist(),
                       'target_video_config': target_video, 'target_action_config': target_action,
                       'source_tensor_map': {key: {'key': key} for key in video},
                       'tensor_map': {f'{family}.{key}': {} for family, state in [('video', video), ('action', action), ('proprio', proprio)] for key in state},
                   })
    native_shapes, _, action_shapes = _expected_shapes()
    for family, state in [('video', video), ('action', action), ('proprio', proprio)]:
        for key in state:
            is_new = family == 'proprio' or (family == 'action' and key.startswith(('head.', 'action_encoder.')))
            operation = 'copy'
            if is_new:
                operation = 'new_nn_Linear_default'
            elif family == 'action' and native_shapes[key] != action_shapes[key]:
                operation = 'sequential_linear_align_corners_then_last_dim_sqrt_scale'
            elif family == 'video' and key.startswith('blocks.'):
                if key.endswith(('ffn.0.weight', 'ffn.0.bias')):
                    operation = 'paired_ffn_select_axis_0'
                elif key.endswith('ffn.2.weight'):
                    operation = 'paired_ffn_select_axis_1'
            payload['metadata']['tensor_map'][f'{family}.{key}'] = {
                'source': None if is_new else key, 'transform': operation, 'seed': 42,
            }
    validate_artifact(payload)
    payload['metadata']['ffn_indices'][1] = 0
    with pytest.raises(ValueError, match='FFN selection'):
        validate_artifact(payload)
    payload['metadata']['ffn_indices'] = canonical_ffn_indices().tolist()
    del payload['metadata']['tensor_map']['action.time_projection.1.weight']
    with pytest.raises(ValueError, match='provenance'):
        validate_artifact(payload)
