import pytest
import torch


def _payload(variant):
    from fastwam.models.wan22.loopwam_robot import FORMAT, VARIANTS
    from fastwam.models.wan22.loopwam_init import target_configs, DONOR_INDICES
    version, video, action = VARIANTS[variant]
    v, a = target_configs(30 if version == 'dense_s30' else 12); a['action_dim'] = 14
    donors = list(range(30)) if version == 'dense_s30' else list(DONOR_INDICES)
    return dict(format_version=FORMAT, version=version, video_loops=video, action_loops=action, memory_tokens=0,
                history_frame=0, inference_loops=video, trained_max_loops=1 if version != 'v0' else 4,
                exit_weight_scale=1., architecture=dict(target_video_config=v, target_action_config=a, robot_proprio_dim=14,
                donor_indices=donors, initialization_mode='fresh_wan_donors_no_embodied_pretraining'))


def test_variant_table_matches_released_libero_family():
    from fastwam.models.wan22.loopwam_robot import VARIANTS, robot_parameter_count
    assert set(VARIANTS) == {'v0-video4-action4', 'v0-video4-action2', 'v0-video4-action1', 'v0-video2-action2',
                             'v0-video1-action4', 'dense-s12', 'dense-s30'}
    assert VARIANTS['v0-video1-action4'] == ('v0', 1, 4) and VARIANTS['dense-s30'] == ('dense_s30', 1, 1)
    assert robot_parameter_count('dense-s12') == 584536135 + 31751
    assert robot_parameter_count('dense-s30') == 1416114247 + 31751


@pytest.mark.parametrize('variant', ['v0-video4-action4', 'v0-video4-action2', 'v0-video4-action1', 'v0-video2-action2',
                                     'v0-video1-action4', 'dense-s12', 'dense-s30'])
def test_robot_payload_validates_each_variant(variant):
    from fastwam.models.wan22.loopwam_robot import validate_robot_payload, payload_variant
    p = _payload(variant)
    validate_robot_payload(p, variant); assert payload_variant(p) == variant
    other = 'dense-s12' if variant != 'dense-s12' else 'v0-video4-action4'
    with pytest.raises(ValueError):
        validate_robot_payload(p, other)
    for field, bad in (('inference_loops', 3), ('trained_max_loops', 2), ('exit_weight_scale', .5)):
        q = dict(p, **{field: bad})
        with pytest.raises(ValueError):
            validate_robot_payload(q)


def test_unsupported_depth_and_wrong_donors_are_rejected():
    from fastwam.models.wan22.loopwam_robot import validate_robot_payload
    with pytest.raises(ValueError):
        validate_robot_payload(dict(_payload('v0-video4-action4'), video_loops=3, inference_loops=3))
    p = _payload('dense-s30'); p['architecture'] = dict(p['architecture'], donor_indices=list(range(12)))
    with pytest.raises(ValueError):
        validate_robot_payload(p)


def test_original_v0_robot_checkpoint_header_still_validates():
    # The full-RoboTwin epoch-5 release predates loop_alignment/action_kv_mode keys.
    from fastwam.models.wan22.loopwam_robot import validate_robot_payload
    p = _payload('v0-video4-action4')
    assert 'loop_alignment' not in p
    validate_robot_payload(p)


def test_robotwin10_budget_and_world_invariant_batches():
    from fastwam.datasets.loopwam_robot_batches import epoch_rank_batches, training_budget
    assert training_budget(1163351, 128, 5) == (9089, 45445, 87)
    size = 128 * 3 + 87
    for epoch in range(2):
        reference = torch.randperm(size, generator=torch.Generator().manual_seed(42 + epoch)).tolist()
        for world in (1, 2, 4, 8):
            ranks = [list(epoch_rank_batches(size, 128, world, r, 42 + epoch)) for r in range(world)]
            rows = [sum([ranks[r][i] for r in range(world)], []) for i in range(4)]
            assert [len(x) for x in rows] == [128, 128, 128, 87] and sum(rows, []) == reference


def test_robotwin10_cursor_and_schedule():
    from train_loopwam_robotwin10 import validate_cursor, lr_factor, UPDATES_PER_EPOCH, WINDOWS
    validate_cursor(0, 0, 0, 0, 5); validate_cursor(UPDATES_PER_EPOCH, WINDOWS, 1, 0, 5)
    validate_cursor(5 * UPDATES_PER_EPOCH, 5 * WINDOWS, 5, 0, 5)
    validate_cursor(UPDATES_PER_EPOCH + 7, WINDOWS + 7 * 128, 1, 7, 5)
    for bad in ((1, 128, 0, 0), (5 * UPDATES_PER_EPOCH + 1, 5 * WINDOWS + 128, 5, 1)):
        with pytest.raises(ValueError):
            validate_cursor(*bad, 5)
    total = 5 * UPDATES_PER_EPOCH; warmup = int(total * .05)
    assert warmup == 2272 and lr_factor(0, total, warmup) == 1 / warmup and lr_factor(warmup, total, warmup) == 1.
    assert abs(lr_factor(total - 1, total, warmup) - .01) < 1e-12
