"""Dense-S12 control: one untied pass, matched donors and checkpoint depth."""
import copy

import pytest
import torch

from fastwam.models.wan22.loop_mot import LoopMoT
from fastwam.models.wan22.loopwam import LoopWAM, create_loopwam, exit_weights
from test_loop_mot import make_model, inputs, cached_action
from test_loopwam_policy import tiny_noisy


def dense_mot(**kwargs):
    reference = make_model('v0', loops=1)
    return LoopMoT(copy.deepcopy(dict(reference.mixtures.items())), version='dense_s12', **kwargs)


def dense_policy():
    mot = dense_mot()
    mot.mixtures['video'].seperated_timestep = True
    mot.mixtures['video'].video_attention_mask_mode = 'first_frame_causal'
    vae = torch.nn.Linear(1, 1)
    vae.temporal_downsample_factor = 4
    return LoopWAM(video_expert=mot.mixtures['video'], action_expert=mot.mixtures['action'],
        mot=mot, vae=vae, text_dim=8, proprio_dim=2, device='cpu', torch_dtype=torch.float32,
        version='dense_s12', action_train_shift=1, action_infer_shift=1)


def test_dense_matches_v0_k1_outputs_gradients_and_cached_inference():
    dense, reference, data = dense_mot(), make_model('v0', loops=1), inputs()
    assert dense.loops == dense.trained_max_loops == 1
    assert dense.schedule == tuple(range(12))
    assert dense.effective_depth() == 12
    assert dense.state_dict().keys() == reference.state_dict().keys()
    outputs = dense.forward_joint_exits(**data)
    assert tuple(outputs) == (1,)
    expected = reference.forward_joint_core(**data)
    for a, b in zip(outputs[1], expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    sum(x.square().mean() for x in outputs[1]).backward()
    sum(x.square().mean() for x in expected).backward()
    for (name, a), (_, b) in zip(dense.named_parameters(), reference.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
    torch.testing.assert_close(cached_action(dense, data), outputs[1][1], rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize('bad', [0, 2, 4, True, 1.0])
def test_dense_rejects_non_one_depth_at_every_entry(bad):
    with pytest.raises(ValueError):
        dense_mot(loops=bad)
    dense = dense_mot()
    with pytest.raises(ValueError):
        dense.loops = bad
    with pytest.raises(ValueError):
        dense.virtual_schedule(bad)
    with pytest.raises(ValueError):
        dense.effective_depth(bad)
    with pytest.raises(ValueError):
        dense.forward_joint_exits(**inputs(), loops=bad)
    with pytest.raises(ValueError):
        exit_weights('dense_s12', bad)


def test_dense_final_only_flow_loss_and_checkpoint_roundtrip(tmp_path):
    policy = dense_policy()
    noisy = tiny_noisy(policy)
    outputs = policy.forward_exits(noisy)
    loss, logs = policy.reduce_flow_losses(outputs, noisy)
    assert exit_weights('dense_s12', 1) == {1: 1.0}
    torch.testing.assert_close(loss, logs['exit1/video'] + logs['exit1/action'])
    assert not any(key.startswith('exit2/') for key in logs)
    loss.backward()
    assert torch.isfinite(loss)
    path = tmp_path / 'dense.pt'
    policy.save_checkpoint(path, step=7)
    payload = torch.load(path, weights_only=False)
    assert payload['version'] == 'dense_s12'
    assert payload['trained_max_loops'] == payload['inference_loops'] == 1
    restored = dense_policy()
    restored.load_checkpoint(path)
    for a, b in zip(restored.forward_exits(noisy)[1], outputs[1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for field in ('trained_max_loops', 'inference_loops'):
        corrupted = copy.deepcopy(payload)
        corrupted[field] = 4
        torch.save(corrupted, path)
        with pytest.raises(ValueError, match='dense_s12'):
            restored.load_checkpoint(path)
        with pytest.raises(ValueError, match='dense_s12'):
            create_loopwam(checkpoint_path=path, vae_path='unused')


def test_dense_checkpoint_factory_resolves_depth_before_resource_loading(tmp_path):
    policy = dense_policy()
    path = tmp_path / 'dense.pt'
    policy.save_checkpoint(path)
    with pytest.raises(ValueError, match='dense_s12'):
        create_loopwam(checkpoint_path=path, vae_path='unused', loops=4)


def test_dense_same_canonical_donors_and_exact_parameter_count():
    from fastwam.models.wan22.loopwam_init import target_configs, DONOR_INDICES
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.action_dit import ActionDiT
    assert DONOR_INDICES == (0, 1, 2, 9, 10, 11, 12, 13, 14, 27, 28, 29)
    video_cfg, action_cfg = target_configs(12)
    with torch.device('meta'):
        dense = LoopMoT({'video': WanVideoDiT(**video_cfg), 'action': ActionDiT(**action_cfg)},
                        version='dense_s12')
        proprio = torch.nn.Linear(8, 4096)
    assert sum(p.numel() for p in dense.parameters()) + sum(p.numel() for p in proprio.parameters()) == 584536135


@pytest.mark.parametrize('version', ['v0', 'v1', 'v2'])
def test_existing_versions_still_default_to_four(version):
    reference = make_model()
    model = LoopMoT(dict(reference.mixtures.items()), version=version)
    assert model.loops == model.trained_max_loops == 4


def test_dense_checkpoint_factory_roundtrip_uses_saved_version_and_one_pass(tmp_path, monkeypatch):
    # Only substitute resource sizes and frozen VAE I/O. Exercise the real
    # factory, expert constructors, wrapper, strict loading and numerical output.
    from fastwam.models.wan22 import loopwam_init
    policy = dense_policy()
    common = dict(text_dim=8, freq_dim=8, eps=1e-6, num_heads=2,
                  attn_head_dim=6, num_layers=12)
    video_cfg = dict(hidden_dim=12, ffn_dim=24, in_dim=2, out_dim=2,
                     patch_size=(1, 1, 1), has_image_input=False,
                     seperated_timestep=True, video_attention_mask_mode='first_frame_causal', **common)
    action_cfg = dict(hidden_dim=8, ffn_dim=16, action_dim=3, **common)
    policy.proprio_encoder = torch.nn.Linear(8, 4096)
    policy.architecture_metadata = dict(target_video_config=video_cfg, target_action_config=action_cfg)
    path = tmp_path / 'dense.pt'
    policy.save_checkpoint(path)
    monkeypatch.setattr(loopwam_init, 'target_configs', lambda depth: (video_cfg, action_cfg))
    monkeypatch.setattr(loopwam_init, 'load_wan21_vae', lambda *args, **kwargs: copy.deepcopy(policy.vae))
    restored = create_loopwam(checkpoint_path=path, vae_path='frozen-test-vae')
    assert restored.version == 'dense_s12'
    assert restored.mot.loops == restored.mot.trained_max_loops == 1
    noisy = tiny_noisy(policy)
    for actual, expected in zip(restored.forward_exits(noisy)[1], policy.forward_exits(noisy)[1]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(restored.policy_parameters(), policy.policy_parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
