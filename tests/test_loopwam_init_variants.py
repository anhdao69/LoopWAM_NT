import pytest
import torch

from fastwam.models.wan22.loopwam_init import (
    CORE_SLOTS, DONOR_INDICES, INIT_VARIANTS, compose_student_state, magnitude_ffn_indices,
    random_core_blocks, select_donor_layers, target_configs, variant_block_plan,
)
from fastwam.models.wan22.wan_video_dit import DiTBlock


def _donor(layers=30):
    torch.manual_seed(3)
    state = {f'blocks.{i}.{name}': torch.randn(4, 3) + i for i in range(layers) for name in ('attn.w', 'ffn.0.weight')}
    state['patch.weight'] = torch.randn(2, 2)
    return state


def test_plans_match_the_requested_wan_blocks():
    wan = lambda v: [arg for _, arg in variant_block_plan(v)]
    assert wan('default') == list(DONOR_INDICES)
    assert wan('ffn_magnitude') == list(DONOR_INDICES)
    assert wan('core16_21') == [0, 1, 2, 15, 16, 17, 18, 19, 20, 27, 28, 29]
    assert wan('stepwise') == [0, 1, 2, 4, 8, 12, 16, 20, 24, 27, 28, 29]
    assert [p for p in variant_block_plan('core_mean')[3:9]] == [['mean', [3 + i, 9 + i, 15 + i, 21 + i]] for i in range(6)]
    assert [p for p in variant_block_plan('core_random')[3:9]] == [['random', i] for i in range(6)]
    for variant in INIT_VARIANTS:
        plan = variant_block_plan(variant)
        assert len(plan) == 12 and plan[:3] == [['wan', 0], ['wan', 1], ['wan', 2]] and plan[9:] == [['wan', 27], ['wan', 28], ['wan', 29]]
    with pytest.raises(ValueError):
        variant_block_plan('core_unknown')


def test_default_plan_reproduces_select_donor_layers_exactly():
    donor = _donor()
    state, record = compose_student_state(donor, variant_block_plan('default'))
    expected = select_donor_layers(donor)
    assert state.keys() == expected.keys()
    assert all(torch.equal(state[k], expected[k]) and state[k].data_ptr() != expected[k].data_ptr() for k in state)
    assert record['blocks.3.attn.w'] == dict(source='blocks.9.attn.w', transform='copy')


def test_core_mean_is_fp64_elementwise_mean_of_the_four_donors():
    donor = _donor()
    state, record = compose_student_state(donor, variant_block_plan('core_mean'))
    for i, slot in enumerate(CORE_SLOTS):
        for name in ('attn.w', 'ffn.0.weight'):
            group = [donor[f'blocks.{j}.{name}'].double() for j in (3 + i, 9 + i, 15 + i, 21 + i)]
            expected = (sum(group) / 4).float()
            torch.testing.assert_close(state[f'blocks.{slot}.{name}'], expected, rtol=0, atol=0)
            assert state[f'blocks.{slot}.{name}'].dtype == torch.float32
    assert torch.equal(state['blocks.0.attn.w'], donor['blocks.0.attn.w'])
    assert torch.equal(state['blocks.11.attn.w'], donor['blocks.29.attn.w'])
    assert record['blocks.5.attn.w']['transform'] == 'elementwise_mean_fp64'


def test_magnitude_selection_matches_explicit_scores_and_is_sorted():
    torch.manual_seed(0)
    w1, w2 = torch.randn(40, 6), torch.randn(6, 40)
    chosen = magnitude_ffn_indices(w1, w2, 25)
    score = [float(w1[k].double().norm() * w2[:, k].double().norm()) for k in range(40)]
    expected = sorted(sorted(range(40), key=lambda k: -score[k])[:25])
    assert chosen.tolist() == expected
    # Ties keep the lower index; the result is ascending and unique.
    tied = magnitude_ffn_indices(torch.ones(5, 2), torch.ones(2, 5), 3)
    assert tied.tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        magnitude_ffn_indices(torch.ones(5, 2), torch.ones(2, 4), 3)


def test_magnitude_pruning_keeps_the_ffn_function_of_the_kept_neurons():
    torch.manual_seed(1)
    x, w1, b1, w2 = torch.randn(3, 6), torch.randn(40, 6), torch.randn(40), torch.randn(6, 40)
    idx = magnitude_ffn_indices(w1, w2, 25)
    mask = torch.zeros(40)
    mask[idx] = 1
    act = torch.nn.functional.gelu(x @ w1.T + b1, approximate='tanh')
    expected = (act * mask) @ w2.T
    actual = torch.nn.functional.gelu(x @ w1[idx].T + b1[idx], approximate='tanh') @ w2[:, idx].T
    torch.testing.assert_close(actual, expected)


def test_random_core_blocks_are_seeded_default_ditblocks_and_leave_global_rng_alone():
    torch.manual_seed(123)
    before = torch.rand(1)
    torch.manual_seed(123)
    blocks = random_core_blocks(42)
    assert torch.equal(torch.rand(1), before)
    again = random_core_blocks(42)
    video_cfg, action_cfg = target_configs(12)
    torch.manual_seed(42)
    reference = [DiTBlock(c['hidden_dim'], c['attn_head_dim'], c['num_heads'], c['ffn_dim'], c['eps'])
                 for c in [video_cfg] * 6 + [action_cfg] * 6]
    for family, offset in (('video', 0), ('action', 6)):
        assert len(blocks[family]) == 6
        for i in range(6):
            ref = reference[offset + i].state_dict()
            assert blocks[family][i].keys() == ref.keys()
            assert all(torch.equal(blocks[family][i][k], ref[k]) and torch.equal(again[family][i][k], ref[k]) for k in ref)
    # Default init: norm3 affine is ones/zeros, distinct draws per block.
    assert torch.all(blocks['video'][0]['norm3.weight'] == 1) and torch.all(blocks['video'][0]['norm3.bias'] == 0)
    assert not torch.equal(blocks['video'][0]['self_attn.q.weight'], blocks['video'][1]['self_attn.q.weight'])
    assert not torch.equal(random_core_blocks(43)['video'][0]['self_attn.q.weight'], blocks['video'][0]['self_attn.q.weight'])


def test_random_plan_requires_complete_blocks():
    donor = _donor()
    blocks = [{'attn.w': torch.zeros(4, 3), 'ffn.0.weight': torch.ones(4, 3)} for _ in range(6)]
    state, record = compose_student_state(donor, variant_block_plan('core_random'), blocks)
    assert torch.equal(state['blocks.3.ffn.0.weight'], torch.ones(4, 3))
    assert torch.equal(state['blocks.2.attn.w'], donor['blocks.2.attn.w'])
    assert record['blocks.8.attn.w'] == dict(source=None, transform='new_DiTBlock_default', core_index=5)
    with pytest.raises(ValueError):
        compose_student_state(donor, variant_block_plan('core_random'), [{'attn.w': torch.zeros(4, 3)}] * 6)
