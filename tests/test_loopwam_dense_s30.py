"""Native-layer Dense-S30: thirty distinct donors, final-only loss and caches."""
import copy

import pytest
import torch

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.models.wan22.loop_mot import LoopMoT
from fastwam.models.wan22.loopwam import LoopWAM, create_loopwam, exit_weights
from fastwam.models.wan22 import loopwam_init
from fastwam.models.wan22.mot import MoT
from test_loop_mot import inputs
from test_loopwam_policy import tiny_noisy


def tiny_configs(depth=30):
    common = dict(text_dim=8, freq_dim=8, eps=1e-6, num_heads=2,
                  attn_head_dim=6, num_layers=depth)
    return (dict(hidden_dim=12, ffn_dim=24, in_dim=2, out_dim=2,
                 patch_size=(1, 1, 1), has_image_input=False,
                 seperated_timestep=True, video_attention_mask_mode='first_frame_causal', **common),
            dict(hidden_dim=8, ffn_dim=16, action_dim=3, **common))


def dense_mot(**kwargs):
    torch.manual_seed(20)
    video_cfg, action_cfg = tiny_configs()
    return LoopMoT({'video': WanVideoDiT(**video_cfg), 'action': ActionDiT(**action_cfg)},
                   version='dense_s30', **kwargs)


def dense_policy():
    mot = dense_mot()
    vae = torch.nn.Linear(1, 1)
    vae.temporal_downsample_factor = 4
    metadata = loopwam_init.architecture_metadata_for_version({}, 'dense_s30')
    metadata['target_video_config'], metadata['target_action_config'] = tiny_configs()
    return LoopWAM(video_expert=mot.mixtures['video'], action_expert=mot.mixtures['action'],
        mot=mot, vae=vae, text_dim=4096, proprio_dim=8, device='cpu', torch_dtype=torch.float32,
        version='dense_s30', architecture_metadata=metadata,
        action_train_shift=1, action_infer_shift=1)


@pytest.mark.parametrize('optimized', [False, True])
def test_dense_native_order_all_layer_gradients_and_checkpointed_structured_attention(optimized):
    model, data = dense_mot(), inputs()
    reference = MoT(copy.deepcopy(dict(model.mixtures.items())))
    if optimized:
        model.checkpoint_blocks = True
        model.structured_attention = True
        model.structured_attention_observation_tokens = 2
    assert model.loops == model.trained_max_loops == 1
    assert model.unique_depth == model.effective_depth() == 30
    assert model.schedule == tuple(range(30))
    assert len(set(model.virtual_keys)) == 30
    outputs = model.forward_joint_exits(**data)
    assert tuple(outputs) == (1,)
    expected = reference.forward_joint_core(**data)
    for a, b in zip(outputs[1], expected):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)
    for pair in (outputs[1], expected):
        sum(x.square().mean() for x in pair).backward()
    for family in ('video', 'action'):
        blocks = model.mixtures[family].blocks
        assert len({id(block) for block in blocks}) == 30
        assert len({block.self_attn.q.weight.data_ptr() for block in blocks}) == 30
        for block in blocks:
            grad = block.self_attn.q.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert (actual.grad is None) == (expected.grad is None), name
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-4, atol=5e-5, msg=name)


def test_dense_cache_has_thirty_distinct_slots_and_matches_joint():
    model, data = dense_mot(), inputs()
    keys, values = model.prefill_video_cache_tensor(
        data['video_tokens'][:, :2], data['video_freqs'][:2], data['video_t_mod'][:, :2],
        data['video_context'], data['video_context_mask'][:, :2], data['attention_mask'][:2, :2])
    assert len(keys) == len(values) == 30
    assert len({key.data_ptr() for key in keys}) == 30
    mask = torch.cat((data['attention_mask'][4:, :2], data['attention_mask'][4:, 4:]), dim=1)
    def action(k, v):
        return model.forward_action_with_video_cache_tensor(
            data['action_tokens'], data['action_freqs'], data['action_t_mod'],
            data['action_context'], data['action_context_mask'], k, v, mask)
    torch.testing.assert_close(action(keys, values), model.forward_joint_core(**data)[1],
                               rtol=3e-5, atol=3e-5)
    with pytest.raises(ValueError, match='30 virtual layers'):
        action(keys[:12], values[:12])


@pytest.mark.parametrize('bad', [0, 2, 4, True, 1.0])
def test_dense_requires_one_pass(bad):
    with pytest.raises(ValueError):
        dense_mot(loops=bad)
    model = dense_mot()
    for call in (lambda: setattr(model, 'loops', bad), lambda: model.virtual_schedule(bad),
                 lambda: model.effective_depth(bad), lambda: exit_weights('dense_s30', bad),
                 lambda: model.forward_joint_exits(**inputs(), loops=bad)):
        with pytest.raises(ValueError):
            call()


def test_dense_rejects_twelve_physical_blocks():
    video_cfg, action_cfg = tiny_configs(12)
    with pytest.raises(ValueError, match='30 physical blocks'):
        LoopMoT({'video': WanVideoDiT(**video_cfg), 'action': ActionDiT(**action_cfg)}, version='dense_s30')


def test_metadata_preserves_artifact_and_exact_parameter_count():
    video_cfg, action_cfg = loopwam_init.target_configs(12)
    original = dict(architecture_version='LoopWAM-S-3-6-3', donor_indices=list(loopwam_init.DONOR_INDICES),
                    target_video_config=video_cfg, target_action_config=action_cfg,
                    source_hashes={'source': 'a' * 64})
    snapshot = copy.deepcopy(original)
    metadata = loopwam_init.architecture_metadata_for_version(original, 'dense_s30')
    assert original == snapshot
    assert metadata['donor_indices'] == list(range(30))
    assert metadata['architecture_version'] == 'Dense-S30-native-v1'
    assert metadata['source_hashes'] == original['source_hashes']
    with torch.device('meta'):
        model = LoopMoT({'video': WanVideoDiT(**metadata['target_video_config']),
                         'action': ActionDiT(**metadata['target_action_config'])}, version='dense_s30')
        proprio = torch.nn.Linear(8, 4096)
    count = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in proprio.parameters())
    assert count == loopwam_init.expected_policy_parameter_count('dense_s30') == 1416114247


def test_final_only_loss_strict_checkpoint_factory_roundtrip(tmp_path, monkeypatch):
    policy = dense_policy()
    noisy = tiny_noisy(policy)
    outputs = policy.forward_exits(noisy)
    loss, logs = policy.reduce_flow_losses(outputs, noisy)
    assert exit_weights('dense_s30', 1) == {1: 1.0}
    torch.testing.assert_close(loss, logs['exit1/video'] + logs['exit1/action'])
    assert not any(key.startswith('exit2/') for key in logs)
    path = tmp_path / 'dense30.pt'
    policy.save_checkpoint(path)
    monkeypatch.setattr(loopwam_init, 'target_configs', tiny_configs)
    monkeypatch.setattr(loopwam_init, 'load_wan21_vae', lambda *args, **kwargs: copy.deepcopy(policy.vae))
    restored = create_loopwam(checkpoint_path=path, vae_path='frozen-test-vae')
    assert restored.version == 'dense_s30'
    assert restored.mot.schedule == tuple(range(30))
    for a, b in zip(restored.forward_exits(noisy)[1], outputs[1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    restored.load_checkpoint(path)
    with pytest.raises(ValueError, match='dense_s30'):
        create_loopwam(checkpoint_path=path, vae_path='unused', loops=4)
    payload = torch.load(path, weights_only=False)
    for field in ('trained_max_loops', 'inference_loops'):
        corrupt = copy.deepcopy(payload)
        corrupt[field] = 4
        torch.save(corrupt, path)
        for load in (lambda: restored.load_checkpoint(path),
                     lambda: create_loopwam(checkpoint_path=path, vae_path='unused')):
            with pytest.raises(ValueError, match='dense_s30'):
                load()
    for key, value in [('architecture_version', 'LoopWAM-S-3-6-3'),
                       ('donor_indices', list(loopwam_init.DONOR_INDICES)),
                       ('target_video_config', tiny_configs(12)[0]),
                       ('target_action_config', tiny_configs(12)[1])]:
        corrupt = copy.deepcopy(payload)
        corrupt['architecture'][key] = value
        torch.save(corrupt, path)
        with pytest.raises(ValueError, match='dense_s30'):
            create_loopwam(checkpoint_path=path, vae_path='unused')
    # Correct metadata cannot hide a missing last native layer's tensor.
    del payload['mot']['mixtures.video.blocks.29.self_attn.q.weight']
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        create_loopwam(checkpoint_path=path, vae_path='frozen-test-vae')


def test_build_dense_experts_copies_every_native_donor_in_order(monkeypatch):
    # Tiny validated donor stand-in exercises the real construction/loading path.
    # Canonical artifact validation has its own production-shape coverage tests.
    model = dense_mot()
    for family in ('video', 'action'):
        for index, block in enumerate(model.mixtures[family].blocks):
            with torch.no_grad():
                block.self_attn.q.weight.fill_(index + (100 if family == 'action' else 0))
    proprio = torch.nn.Linear(8, 4096)
    artifact = dict(video_state_dict=model.mixtures['video'].state_dict(),
                    action_state_dict=model.mixtures['action'].state_dict(),
                    proprio_state_dict=proprio.state_dict())
    monkeypatch.setattr(loopwam_init, 'validate_artifact', lambda value: None)
    monkeypatch.setattr(loopwam_init, 'target_configs', tiny_configs)
    video, action, new_proprio = loopwam_init.build_target_experts(artifact, dense30=True)
    for family, expert in [('video', video), ('action', action)]:
        assert len(expert.blocks) == 30
        for index, block in enumerate(expert.blocks):
            expected = model.mixtures[family].blocks[index].self_attn.q.weight
            torch.testing.assert_close(block.self_attn.q.weight, expected, rtol=0, atol=0)
            assert block.self_attn.q.weight.data_ptr() != expected.data_ptr()
        for name, parameter in expert.state_dict().items():
            torch.testing.assert_close(parameter, artifact[f'{family}_state_dict'][name], rtol=0, atol=0)
    torch.testing.assert_close(new_proprio.weight, proprio.weight, rtol=0, atol=0)
