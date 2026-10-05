"""Numerical contracts for LoopMoT using the production Wan/Action blocks."""
import copy
import importlib.util

import pytest
import torch

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT, precompute_freqs_cis


def test_loop_mot_is_implemented():
    assert importlib.util.find_spec("fastwam.models.wan22.loop_mot") is not None


def make_model(version="v0", loops=4, checkpoint_blocks=False):
    from fastwam.models.wan22.loop_mot import LoopMoT
    torch.manual_seed(20)
    common = dict(text_dim=8, freq_dim=8, eps=1e-6, num_heads=2,
                  attn_head_dim=6, num_layers=12)
    video = WanVideoDiT(hidden_dim=12, ffn_dim=24, in_dim=2, out_dim=2,
                        patch_size=(1, 1, 1), has_image_input=False, **common)
    action = ActionDiT(hidden_dim=8, ffn_dim=16, action_dim=3, **common)
    return LoopMoT({"video": video, "action": action}, loops=loops,
                   version=version, checkpoint_blocks=checkpoint_blocks)


def inputs():
    torch.manual_seed(21)
    # Two observed tokens, two future tokens, three action tokens.
    mask = torch.zeros(7, 7, dtype=torch.bool)
    mask[:2, :2] = True
    mask[2:4, :4] = True
    mask[4:, :2] = True
    mask[4:, 4:] = True
    return dict(video_tokens=torch.randn(1, 4, 12), action_tokens=torch.randn(1, 3, 8),
                video_freqs=precompute_freqs_cis(6, 4).unsqueeze(1),
                action_freqs=precompute_freqs_cis(6, 3).unsqueeze(1),
                video_t_mod=torch.randn(1, 4, 6, 12) * .1,
                action_t_mod=torch.randn(1, 6, 8) * .1,
                video_context=torch.randn(1, 3, 12),
                action_context=torch.randn(1, 3, 8),
                video_context_mask=torch.tensor([[[True, True, False]]]).expand(1, 4, 3),
                action_context_mask=torch.tensor([[[True, True, False]]]).expand(1, 3, 3),
                attention_mask=mask)


def assert_pair_close(actual, expected):
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-5)


def test_k1_equals_dense_and_k4_equals_explicit_unroll():
    model, data = make_model(), inputs()
    dense = MoT(copy.deepcopy(dict(model.mixtures.items())))
    assert_pair_close(model.forward_joint_core(**data, loops=1), dense.forward_joint_core(**data))
    expanded = MoT(copy.deepcopy(dict(model.mixtures.items())))
    indices = list(range(3)) + list(range(3, 9)) * 4 + list(range(9, 12))
    for name in ("video", "action"):
        expanded.mixtures[name].blocks = torch.nn.ModuleList([
            copy.deepcopy(model.mixtures[name].blocks[i]) for i in indices])
    expanded.num_layers = 30
    actual = model.forward_joint_core(**data)
    expected = expanded.forward_joint_core(**data)
    assert_pair_close(actual, expected)
    sum(x.square().mean() for x in actual).backward()
    sum(x.square().mean() for x in expected).backward()
    for name in ("video", "action"):
        for i in range(3, 9):
            grad = model.mixtures[name].blocks[i].self_attn.q.weight.grad
            unrolled_grad = sum(expanded.mixtures[name].blocks[j].self_attn.q.weight.grad
                                for j, physical in enumerate(indices) if physical == i)
            torch.testing.assert_close(grad, unrolled_grad, atol=3e-5, rtol=3e-4)
    assert len(model.schedule) == 30
    assert len(set(model.virtual_keys)) == 30
    assert len(list(model.parameters())) == len(list(dense.parameters()))


@pytest.mark.parametrize("version", ["v0", "v1", "v2"])
def test_exit_truncation_and_coda_does_not_feed_back(version):
    model, data = make_model(version), inputs()
    all_exits = model.forward_joint_exits(**data, exits=(1, 2, 3, 4))
    assert tuple(all_exits) == (1, 2, 3, 4)
    for k in range(1, 5):
        assert_pair_close(all_exits[k], model.forward_joint_core(**data, loops=k))
    assert_pair_close(all_exits[4], model.forward_joint_exits(**data, exits=(4,))[4])
    default = model.forward_joint_exits(**data)
    assert tuple(default) == ((4,) if version == "v0" else (1, 2, 3, 4))


@pytest.mark.parametrize("version", ["v0", "v2"])
def test_future_and_action_cannot_leak_into_observation_or_action(version):
    model, data = make_model(version), inputs()
    baseline = model.forward_joint_core(**data)
    changed = dict(data, video_tokens=data["video_tokens"].clone())
    changed["video_tokens"][:, 2:] += 50
    other = model.forward_joint_core(**changed)
    torch.testing.assert_close(other[1], baseline[1], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(other[0][:, :2], baseline[0][:, :2], atol=1e-6, rtol=1e-6)
    changed = dict(data, action_tokens=data["action_tokens"] + 50)
    torch.testing.assert_close(model.forward_joint_core(**changed)[0], baseline[0], atol=1e-6, rtol=1e-6)


def cached_action(model, data):
    # Production inference prefills only the clean observation, not future video.
    keys, values = model.prefill_video_cache_tensor(
        data["video_tokens"][:, :2], data["video_freqs"][:2], data["video_t_mod"][:, :2],
        data["video_context"], data["video_context_mask"][:, :2], data["attention_mask"][:2, :2])
    assert len(keys) == 3 + 6 * model.loops + 3
    assert keys[3].data_ptr() != keys[9].data_ptr()
    action_mask = torch.cat((data["attention_mask"][4:, :2], data["attention_mask"][4:, 4:]), dim=1)
    return model.forward_action_with_video_cache_tensor(
        data["action_tokens"], data["action_freqs"], data["action_t_mod"],
        data["action_context"], data["action_context_mask"], keys, values, action_mask)


@pytest.mark.parametrize("version", ["v0", "v2"])
def test_cache_matches_joint_outputs_and_gradients(version):
    joint, data = make_model(version), inputs()
    cached = copy.deepcopy(joint)
    a = joint.forward_joint_core(**data)[1]
    b = cached_action(cached, data)
    torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
    a.square().mean().backward()
    b.square().mean().backward()
    for (name, p), (_, q) in zip(joint.named_parameters(), cached.named_parameters()):
        # Joint concatenation can create an explicit zero gradient where the
        # cache graph has no edge (e.g. final video Q and output projection).
        p_grad = p.grad if p.grad is not None else torch.zeros_like(p)
        q_grad = q.grad if q.grad is not None else torch.zeros_like(q)
        torch.testing.assert_close(p_grad, q_grad, atol=4e-5, rtol=5e-4, msg=name)


def test_xsa_headwise_fp32_zero_and_near_zero_values():
    from fastwam.models.wan22.loop_mot import xsa_projection
    out = torch.tensor([[[1., 2., 3., 4.]]], dtype=torch.bfloat16)
    own = torch.tensor([[[1., 0., 0., 2.]]], dtype=torch.bfloat16)
    actual = xsa_projection(out, own, num_heads=2)
    torch.testing.assert_close(actual, torch.tensor([[[0., 2., 3., 0.]]], dtype=torch.bfloat16))
    torch.testing.assert_close(xsa_projection(out, torch.zeros_like(out), 2), out)
    tiny = torch.full((1, 1, 4), 1e-15)
    expected = out.float().reshape(1, 1, 2, 2)
    unit = torch.nn.functional.normalize(tiny.reshape(1, 1, 2, 2), dim=-1, eps=1e-12)
    expected = (expected - (expected * unit).sum(-1, keepdim=True) * unit).flatten(2)
    torch.testing.assert_close(xsa_projection(out, tiny, 2), expected.to(out.dtype))
    assert actual.dtype == out.dtype


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_checkpointed_joint_pair_matches_all_exit_gradients(version):
    plain, data = make_model(version), inputs()
    checked = copy.deepcopy(plain)
    checked.checkpoint_blocks = True
    a, b = plain.forward_joint_exits(**data), checked.forward_joint_exits(**data)
    for k in a:
        assert_pair_close(a[k], b[k])
    sum(x.square().mean() for pair in a.values() for x in pair).backward()
    sum(x.square().mean() for pair in b.values() for x in pair).backward()
    for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-4, msg=name)


def test_legacy_forward_routes_same_recurrence():
    model, data = make_model("v2"), inputs()
    result = model(
        {n: data[n + "_tokens"] for n in ("video", "action")}, data["attention_mask"],
        {n: data[n + "_freqs"] for n in ("video", "action")},
        {n: {"context": data[n + "_context"], "mask": data[n + "_context_mask"]}
         for n in ("video", "action")},
        {n: data[n + "_t_mod"] for n in ("video", "action")})
    assert_pair_close((result["video"], result["action"]), model.forward_joint_core(**data))


def test_invalid_schedule_and_stale_cache_rejected():
    model, data = make_model(), inputs()
    with pytest.raises(ValueError):
        model.loops = 0
    with pytest.raises(ValueError):
        model.forward_joint_exits(**data, exits=(0,))
    with pytest.raises(ValueError):
        model.forward_joint_exits(**data, loops=2, exits=(3,))
    with pytest.raises(ValueError):
        model.forward_action_with_video_cache_tensor(
            data["action_tokens"], data["action_freqs"], data["action_t_mod"],
            data["action_context"], data["action_context_mask"], [], [], data["attention_mask"][4:])


def test_explicit_depth_metadata():
    model = make_model()
    assert (model.unique_depth, model.pre_depth, model.core_depth, model.post_depth) == (12, 3, 6, 3)
    assert model.trained_max_loops == 4
    assert [model.effective_depth(k) for k in range(1, 5)] == [12, 18, 24, 30]


def test_v2_matches_explicit_core_only_head_projection():
    model, data = make_model("v2"), inputs()
    reference = copy.deepcopy(model)
    state = (data["video_tokens"], data["action_tokens"])
    for physical in list(range(3)) + list(range(3, 9)) * 4 + list(range(9, 12)):
        ios = []
        blocks = []
        for name, x in zip(("video", "action"), state):
            expert = reference.mixtures[name]
            block = expert.blocks[physical]
            blocks.append(block)
            ios.append(reference._build_expert_attention_io(
                expert, block, x, data[name + "_freqs"], data[name + "_t_mod"]))
        q, k, v = [torch.cat((ios[0][j], ios[1][j]), dim=1) for j in range(3)]
        # Independent explicit per-head implementation, including placement.
        mixed_heads = []
        for h in range(2):
            sl = slice(6 * h, 6 * (h + 1))
            weights = (q[..., sl] @ k[..., sl].transpose(-2, -1) / 6 ** .5)
            weights = weights.masked_fill(~data["attention_mask"], -torch.inf).softmax(-1)
            out = weights @ v[..., sl]
            if 3 <= physical < 9:
                unit = v[..., sl] / v[..., sl].norm(dim=-1, keepdim=True).clamp_min(1e-12)
                out = out - (out * unit).sum(-1, keepdim=True) * unit
            mixed_heads.append(out)
        mixed = torch.cat(mixed_heads, dim=-1)
        state = tuple(reference._apply_expert_post_block_tensor(
            block, io[3], part, *io[4:8], data[name + "_context"], data[name + "_context_mask"])
            for name, block, io, part in zip(("video", "action"), blocks, ios,
                                             (mixed[:, :4], mixed[:, 4:])))
    assert_pair_close(model.forward_joint_core(**data), state)


@pytest.mark.parametrize("loops", [1, 4])
def test_legacy_cache_and_loop_budget(loops):
    model, data = make_model("v2", loops), inputs()
    payload_v = {"context": data["video_context"], "mask": data["video_context_mask"]}
    payload_a = {"context": data["action_context"], "mask": data["action_context_mask"]}
    cache = model.prefill_video_cache(data["video_tokens"], data["video_freqs"],
                                      data["video_t_mod"], payload_v, data["attention_mask"][:4, :4])
    actual = model.forward_action_with_video_cache(
        data["action_tokens"], data["action_freqs"], data["action_t_mod"], payload_a,
        cache, data["attention_mask"], 4)
    torch.testing.assert_close(actual, model.forward_joint_core(**data)[1], atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 acceptance gate")
@pytest.mark.parametrize("version", ["v0", "v2"])
def test_cuda_bf16_autocast_joint_cache_and_checkpoint(version):
    """Keep optimizer parameters FP32 and exercise the actual BF16 compute path."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support BF16")
    joint = make_model(version).cuda()
    cached, checked, reference = (copy.deepcopy(joint) for _ in range(3))
    checked.checkpoint_blocks = True
    data = {key: value.cuda() for key, value in inputs().items()}
    assert all(p.dtype == torch.float32 for p in joint.parameters())
    fp32_action = reference.forward_joint_core(**data)[1]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        joint_action = joint.forward_joint_core(**data)[1]
        cache_action = cached_action(cached, data)
        checkpoint_action = checked.forward_joint_core(**data)[1]
    # Compare aggregate error to output scale: small individual entries may cross
    # zero under BF16 while the overall prediction remains numerically faithful.
    def relative_error(actual, expected):
        return (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-6)
    assert relative_error(joint_action, fp32_action) < .08
    assert relative_error(cache_action, joint_action) < .04
    assert relative_error(checkpoint_action, joint_action) < .005
    for output in (joint_action, cache_action, checkpoint_action, fp32_action):
        output.float().square().mean().backward()
    def gradients(model):
        return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                          for p in model.parameters()])
    grads = [gradients(m) for m in (joint, cached, checked, reference)]
    assert all(torch.isfinite(g).all() for g in grads)
    assert relative_error(grads[0], grads[3]) < .15
    assert relative_error(grads[1], grads[0]) < .08
    assert relative_error(grads[2], grads[0]) < .005


def test_optional_loop_diagnostics_are_detached_and_reset():
    from fastwam.models.wan22.loop_mot import LoopMoT
    plain = make_model("v1")
    model = LoopMoT(copy.deepcopy(dict(plain.mixtures.items())), version="v1",
                    checkpoint_blocks=True, collect_diagnostics=True)
    data = inputs()
    expected = plain.forward_joint_exits(**data)
    actual = model.forward_joint_exits(**data)
    for k in actual:
        assert_pair_close(actual[k], expected[k])
    assert set(model.last_diagnostics) == {f"loop/{k}/{name}_state_rms"
        for k in range(1, 5) for name in ("video", "action")}
    before_backward = dict(model.last_diagnostics)
    sum(x.square().mean() for pair in actual.values() for x in pair).backward()
    for key, value in model.last_diagnostics.items():
        assert value.ndim == 0 and not value.requires_grad and torch.isfinite(value) and value > 0
        torch.testing.assert_close(value, before_backward[key])
    model.forward_joint_core(**data, loops=1)
    assert len(model.last_diagnostics) == 2
    model.collect_diagnostics = False
    model.forward_joint_core(**data)
    assert model.last_diagnostics == {}
