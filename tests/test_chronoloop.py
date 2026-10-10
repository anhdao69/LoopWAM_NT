"""ChronoLoop pre-round-1 tests (plans/ChronoLoop.md section 3.3). GPU + parent checkpoint required.

CHRONO_PARENT=<policy.pt> python -m pytest -q tests/test_chronoloop.py
"""
import os
from dataclasses import replace

import pytest
import torch

from fastwam.models.wan22.chronoloop import ChronoConfig, ChronoMemory, create_chronoloop

PARENT = os.environ.get('CHRONO_PARENT')
VAE = 'checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'
pytestmark = pytest.mark.skipif(not (PARENT and torch.cuda.is_available()), reason='needs GPU and CHRONO_PARENT')

CL0 = ChronoConfig()
CLA = ChronoConfig(memory_tokens=16, mem_source='learned', mem_write='loop')
CLW2 = ChronoConfig(memory_tokens=16, mem_source='learned', mem_write='external')
CLREG = ChronoConfig(memory_tokens=16, mem_source='reset', mem_write='loop')
CLFRAME = ChronoConfig(history_frame=3)
CLA1 = replace(CLA, action_loops=1)
CL01 = replace(CL0, action_loops=1)


def build(cfg, ckpt_blocks=False):
    model, _ = create_chronoloop(cfg, VAE, parent_path=PARENT, device='cuda', checkpoint_blocks=ckpt_blocks)
    model.mot.structured_attention = True
    model.mot.structured_attention_observation_tokens = 392
    if model.memory is not None and cfg.memory_tokens:
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            model.memory.e.copy_(torch.randn(model.memory.e.shape, generator=g) * 0.5)
            model.memory.gamma.fill_(0.3)
    return model


def make_batch(b, seed, device='cuda'):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(b, 128, dtype=torch.bool); mask[:, :20] = True
    sample = dict(latents=torch.randn(b, 16, 3, 28, 56, generator=g).to(device),
                  action=torch.randn(b, 32, 7, generator=g).clamp(-1, 1).to(device),
                  proprio=torch.randn(b, 32, 8, generator=g).clamp(-1, 1).to(device),
                  action_is_pad=torch.zeros(b, 32, dtype=torch.bool, device=device),
                  image_is_pad=torch.zeros(b, 9, dtype=torch.bool, device=device),
                  context=(torch.randn(b, 128, 4096, generator=g) * mask[..., None]).to(device),
                  context_mask=mask.to(device))
    sample['video'] = torch.zeros((), device=device).expand(b, 3, 9, 224, 448)
    noise = dict(video=torch.randn(b, 16, 3, 28, 56, generator=g).to(device),
                 action=torch.randn(b, 32, 7, generator=g).to(device),
                 t_video=(torch.rand(b, generator=g) * 1000).to(device),
                 t_action=(torch.rand(b, generator=g) * 1000).to(device))
    return sample, noise


@pytest.fixture(scope='module')
def cla():
    return build(CLA)


@pytest.fixture(scope='module')
def cl0():
    return build(CL0)


def losses(model, sample, noise, s=None):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, logs, s_new = model.window_losses(sample, noise, s)
    return loss, logs, s_new


def test_memory_disabled_is_parent_path(cl0):
    assert cl0.memory is None
    assert sum(p.numel() for p in cl0.parameters() if p.requires_grad) == 584536135


def test_init_equivalence_gate_closed(cla, cl0):
    sample, noise = make_batch(2, 1)
    with torch.no_grad():
        l0, logs0, _ = losses(cl0, sample, noise)
        la, logsa, _ = losses(cla, sample, noise)
        s_rand = torch.randn(2, 16, 1536, device='cuda') * 2
        lb, logsb, _ = losses(cla, sample, noise, s_rand)
    # alpha = 0: the memory content cannot change any non-memory output (exactly).
    assert torch.equal(la, lb) and torch.equal(logsa['video'], logsb['video'])
    # ... and the outputs equal the parent's (same math; GEMM shapes differ by 16 rows).
    torch.testing.assert_close(la, l0, rtol=2e-2, atol=2e-3)


def test_no_leakage_and_train_inference(cla):
    model = cla
    with torch.no_grad():
        model.memory.alpha_video.fill_(0.5); model.memory.alpha_action.fill_(0.5)
    try:
        sample, noise = make_batch(1, 2)
        s = torch.randn(1, 16, 1536, device='cuda')
        with torch.no_grad():
            _, _, s1 = losses(model, sample, noise, s)
            noise2 = dict(noise, video=torch.randn_like(noise['video']), action=torch.randn_like(noise['action']),
                          t_action=noise['t_action'] * 0.3)
            sample2 = dict(sample, latents=sample['latents'].clone())
            sample2['latents'][:, :, 1:] = torch.randn_like(sample2['latents'][:, :, 1:])
            sample2['action'] = torch.randn_like(sample['action'])
            _, _, s2 = losses(model, sample2, noise2, s)
        # Future frames, noisy actions and timesteps never reach the memory.
        assert torch.equal(s1, s2)
        # Train <-> inference: stateful prefill gives the same s_k over 5 queries.
        st_train = st_inf = None
        for k in range(5):
            sample, noise = make_batch(1, 10 + k)
            with torch.no_grad():
                _, _, st_train = losses(model, sample, noise, st_train)
                image = None
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    first = sample['latents'][:, :, :1]
                    model._encode_input_image_latents_tensor = lambda *_a, **_k: first   # bypass the VAE
                    out = model.chrono_infer_action(torch.zeros(1, 3, 224, 448), sample['proprio'][0, 0],
                                                    sample['context'][0], sample['context_mask'][0], state=st_inf,
                                                    seed=0)
                del model._encode_input_image_latents_tensor
                st_inf = out['state']
            torch.testing.assert_close(st_inf, st_train, rtol=2e-2, atol=3e-2)
    finally:
        with torch.no_grad():
            model.memory.alpha_video.zero_(); model.memory.alpha_action.zero_()


def test_bounded_state():
    mem = ChronoMemory(CLA, 64, 4, 12)
    s = torch.zeros(2, 16, 64)
    for _ in range(1000):
        s = mem.update(s, mem.squash(torch.randn(2, 16, 64) * 1e4))
    assert s.abs().max() <= CLA.clamp_c + 1e-5


def _segment(model, n, detach_each=False, ckpt=None):
    if ckpt is not None:
        model.mot.checkpoint_blocks = ckpt
    model.train()
    batches = [make_batch(2, 100 + k) for k in range(n)]
    s = torch.zeros(2, 16, 1536, device='cuda')
    states, losses_ = [], []
    for sample, noise in batches:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, _, s_new = model.window_losses(sample, noise, s.detach() if detach_each else s)
        s_new.retain_grad(); states.append(s_new); losses_.append(loss.sum())
        s = s_new
    return states, losses_


def test_temporal_credit(cla):
    model = cla
    with torch.no_grad():
        model.memory.alpha_video.fill_(0.5); model.memory.alpha_action.fill_(0.5)
    try:
        states, ls = _segment(model, 4)
        ls[3].backward()
        for s in states[:3]:
            assert s.grad is not None and s.grad.abs().sum() > 0
        model.zero_grad(set_to_none=True)
        states, ls = _segment(model, 4, detach_each=True)
        ls[3].backward()
        for s in states[:3]:
            assert s.grad is None or s.grad.abs().sum() == 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.mot.mixtures['video'].blocks[4].parameters())
        model.zero_grad(set_to_none=True)
    finally:
        with torch.no_grad():
            model.memory.alpha_video.zero_(); model.memory.alpha_action.zero_()
        model.eval()


def test_recompute_safety(cla):
    model = cla
    with torch.no_grad():
        model.memory.alpha_video.fill_(0.5); model.memory.alpha_action.fill_(0.5)
    try:
        grads = []
        for ckpt in (False, False, True):
            model.zero_grad(set_to_none=True)
            torch.manual_seed(0)
            _, ls = _segment(model, 4, ckpt=ckpt)
            sum(ls).backward()
            grads.append({n: p.grad.detach().float().clone() for n, p in model.named_parameters()
                          if p.grad is not None and ('memory' in n or 'blocks.4.' in n)})
        assert grads[0].keys() == grads[2].keys() and grads[0]
        # BF16 + attention-backward atomics are not bitwise reproducible even without
        # recomputation; checkpointing must stay within that run-to-run noise floor.
        for n in grads[0]:
            ref = grads[0][n].norm().clamp_min(1e-12)
            noise = (grads[1][n] - grads[0][n]).norm() / ref
            diff = (grads[2][n] - grads[0][n]).norm() / ref
            assert diff <= max(3 * noise, 2e-2), (n, float(diff), float(noise))
    finally:
        model.mot.checkpoint_blocks = False
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.memory.alpha_video.zero_(); model.memory.alpha_action.zero_()
        model.eval()


@pytest.mark.parametrize('cfg', [CLW2, CLREG, CLFRAME, CLA1, CL01], ids=['W2', 'REG', 'FRAME', 'A@1', '0@1'])
def test_variants_forward_backward(cfg):
    model = build(cfg, ckpt_blocks=True)
    model.train()
    sample, noise = make_batch(2, 7)
    if cfg.history_frame:
        sample['history_latents'] = torch.randn(2, 16, 1, 28, 56, device='cuda')
    s = torch.randn(2, 16, 1536, device='cuda') if cfg.memory_tokens else None
    loss, logs, s_new = losses(model, sample, noise, s)
    (loss.sum() + (0. * s_new.sum() if s_new is not None else 0.)).backward()
    assert torch.isfinite(loss).all()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing[:5]
    if cfg.memory_tokens and cfg.mem_source == 'learned':
        assert s_new.shape == (2, 16, 1536)
    if cfg.action_loops == 1:
        assert model.mot.action_loops == 1 and model.mot.loops == 4
    if not cfg.memory_tokens and not cfg.history_frame:
        with torch.no_grad():
            parent = build(replace(cfg, action_loops=cfg.action_loops))
        assert parent.memory is None


def test_checkpoint_guard(tmp_path, cla):
    path = tmp_path / 'x.pt'
    cla.save_chrono(path, step=0, weights_only=True)
    with pytest.raises(ValueError, match='Checkpoint guard'):
        create_chronoloop(CLREG, VAE, checkpoint_path=path, device='cpu')
    model, payload = create_chronoloop(CLA, VAE, checkpoint_path=path, device='cpu')
    for (n, a), (_, b) in zip(model.mot.memory.state_dict().items(), cla.mot.memory.state_dict().items()):
        assert torch.equal(a, b.cpu()), n
