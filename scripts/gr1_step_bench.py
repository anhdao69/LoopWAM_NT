#!/usr/bin/env python3
"""Single-GPU GR1 v0 training-step benchmark / profiler (synthetic batch, no DDP).

Same model build, structured F/U/A attention, block compilation, AdamW and clipping as
scripts/train_loopwam_gr1.py; one "update" = 64 windows (the per-GPU share of global 256 on 4 GPUs)
as 64/micro accumulation steps. Variants: --micro, --checkpoint-blocks/--keep-last (activations of
the last N virtual layers kept), --rope complex|real (real = same float64 math without complex
tensors, so inductor can fuse it), --compile-mode. --profile prints the top CUDA kernels of one update.
"""
import argparse
import json
import time

import torch

import fastwam.models.wan22.mot as mot_module
from fastwam.datasets.robocasa_gr1 import LATENT_SHAPE, OBS_TOKENS, ROBOT_DIM
from fastwam.models.wan22.loopwam_gr1 import PARAMETERS, RobotTrainForward, create_loopwam_gr1
from fastwam.models.wan22.wan_video_dit import rope_apply, rope_apply_real


def synthetic(mb, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    mask = torch.zeros(mb, 128, dtype=torch.bool, device=dev); mask[:, :24] = True
    sample = dict(latents=torch.randn(mb, *LATENT_SHAPE, generator=g, device=dev).bfloat16().float(),
                  video=torch.zeros((), device=dev).expand(mb, 3, 9, 256, 256),
                  action=torch.rand(mb, 32, ROBOT_DIM, generator=g, device=dev) * 2 - 1,
                  proprio=torch.rand(mb, 32, ROBOT_DIM, generator=g, device=dev) * 2 - 1,
                  action_is_pad=torch.zeros(mb, 32, dtype=torch.bool, device=dev),
                  image_is_pad=torch.zeros(mb, 9, dtype=torch.bool, device=dev),
                  context=torch.randn(mb, 128, 4096, generator=g, device=dev) * mask[..., None], context_mask=mask)
    noise = dict(video=torch.randn(mb, *LATENT_SHAPE, generator=g, device=dev),
                 action=torch.randn(mb, 32, ROBOT_DIM, generator=g, device=dev),
                 t_video=torch.rand(mb, generator=g, device=dev) * 1000, t_action=torch.rand(mb, generator=g, device=dev) * 1000)
    return sample, noise


def rope_check(dev):
    """Bitwise comparison of the two RoPE implementations on bf16 inputs with the model's complex128 freqs."""
    from fastwam.models.wan22.wan_video_dit import precompute_freqs_cis_3d
    f, h, w = (x.to(dev) for x in precompute_freqs_cis_3d(128))
    freqs = torch.cat([f[:3].view(3, 1, 1, -1).expand(3, 16, 16, -1), h[:16].view(1, 16, 1, -1).expand(3, 16, 16, -1),
                       w[:16].view(1, 1, 16, -1).expand(3, 16, 16, -1)], dim=-1).reshape(768, 1, -1)
    x = (torch.randn(4, 768, 1536, device=dev) * 3).bfloat16()
    a, b = rope_apply(x, freqs, 12), rope_apply_real(x, freqs, 12)
    diff = (a.float() - b.float()).abs()
    return dict(freqs_dtype=str(freqs.dtype), bitwise_equal_fraction=float((a == b).float().mean()), max_abs_diff=float(diff.max()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--init-artifact', default='checkpoints/LoopWAM/wan21_compact_donors.pt')
    p.add_argument('--vae-path', default='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--micro', type=int, default=16)
    p.add_argument('--per-gpu', type=int, default=64)
    p.add_argument('--checkpoint-blocks', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--keep-last', type=int, default=0)
    p.add_argument('--rope', choices=('complex', 'real'), default='complex')
    p.add_argument('--text-tokens', type=int, default=128)
    p.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--compile-mode', default=None)
    p.add_argument('--autocast', action=argparse.BooleanOptionalAction, default=True, help='--no-autocast: fp32 compute (equivalence)')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--updates', type=int, default=6)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--rope-check', action='store_true')
    p.add_argument('--equivalence', action='store_true', help='one fwd/bwd: complex twice (noise floor) vs real, same inputs')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    dev = torch.device('cuda')
    if a.rope_check:
        print(json.dumps(dict(rope_check=rope_check(dev))), flush=True)
        return
    if a.rope == 'real':
        mot_module.rope_apply = rope_apply_real
    model, _ = create_loopwam_gr1(a.vae_path, init_artifact=a.init_artifact, device=dev, checkpoint_blocks=a.checkpoint_blocks, seed=42)
    model.train(); model.mot.structured_attention = True; model.mot.structured_attention_observation_tokens = OBS_TOKENS
    model.mot.uncheckpointed_virtual_layers = a.keep_last
    if a.compile:
        torch._dynamo.config.cache_size_limit = 256; torch._dynamo.config.accumulated_cache_size_limit = 8192
        torch._inductor.config.comprehensive_padding = False
        for name in ('_joint_block', '_video_only_block'):
            setattr(model.mot, name, torch.compile(getattr(model.mot, name), mode=a.compile_mode))
    params = model.policy_parameters(); assert sum(x.numel() for x in params) == PARAMETERS
    opt = torch.optim.AdamW(params, lr=1e-5, betas=(.9, .95), eps=1e-8, weight_decay=.01, fused=True)
    runner = RobotTrainForward(model)
    assert a.per_gpu % a.micro == 0
    accum = a.per_gpu // a.micro
    batches = [synthetic(a.micro, dev, s) for s in range(min(accum, 4))]
    for sample, _ in batches:
        assert not sample['context_mask'][:, a.text_tokens:].any()
        sample['context'] = sample['context'][:, :a.text_tokens].contiguous(); sample['context_mask'] = sample['context_mask'][:, :a.text_tokens].contiguous()

    def update():
        total = 0.
        for i in range(accum):
            sample, noise = batches[i % len(batches)]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, _, _ = runner([dict(sample=sample, noise=noise)])
            (loss / a.per_gpu).backward(); total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); opt.step(); opt.zero_grad(set_to_none=True)
        return total / a.per_gpu

    if a.equivalence:
        sample, noise = batches[0]

        full = synthetic(a.micro, dev, 0)[0]  # untrimmed 128-token context (batches[0] is trimmed to --text-tokens)

        def grads(rope, s=None):
            mot_module.rope_apply = {'complex': rope_apply, 'real': rope_apply_real}[rope]
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.autocast):
                loss, logs, _ = runner([dict(sample=s or full, noise=noise)])
            loss.backward()
            return [float(loss), float(logs['video']), float(logs['action'])], torch.cat([x.grad.flatten() for x in params])

        def compare(x, y):
            return dict(loss_abs_diff=[abs(u - v) for u, v in zip(x[0], y[0])], grad_rel_l2=float((x[1] - y[1]).norm() / x[1].norm()),
                        grad_cos=float(torch.nn.functional.cosine_similarity(x[1], y[1], dim=0)))
        c1 = grads('complex'); c1 = (c1[0], c1[1].clone()); c2 = grads('complex'); c2 = (c2[0], c2[1].clone())
        r = grads('real'); r = (r[0], r[1].clone()); r2 = grads('real'); r2 = (r2[0], r2[1].clone()); rt = grads('real', sample)
        out = dict(equivalence=dict(text_tokens=sample['context'].shape[1], autocast=a.autocast, compile=a.compile, loss_complex=c1[0], loss_real=r[0], loss_real_trimmed=rt[0],
                   complex_vs_complex=compare(c1, c2), complex_vs_real=compare(c1, r), real_vs_real=compare(r, r2),
                   real_vs_real_trimmed=compare(r, rt)))
        print(json.dumps(out), flush=True)
        if a.out:
            with open(a.out, 'w') as f:
                json.dump(out, f, indent=2)
        return
    report = dict(vars(a))
    try:
        t = time.time()
        for _ in range(a.warmup):
            update()
        torch.cuda.synchronize(); report['warmup_s'] = round(time.time() - t, 1)
        torch.cuda.reset_peak_memory_stats()
        times, losses = [], []
        for _ in range(a.updates):
            torch.cuda.synchronize(); t = time.time(); losses.append(update()); torch.cuda.synchronize(); times.append(time.time() - t)
        times.sort()
        report.update(status='ok', s_per_update_median=round(times[len(times) // 2], 4), s_per_update_min=round(times[0], 4),
                      windows_per_s_per_gpu=round(a.per_gpu / times[len(times) // 2], 2), loss_last=losses[-1],
                      peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                      peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2))
    except torch.cuda.OutOfMemoryError as e:
        report.update(status='oom', error=str(e).splitlines()[0])
    print(json.dumps(report), flush=True)
    if a.profile and report.get('status') == 'ok':
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            update(); torch.cuda.synchronize()
        # Kernels only (CPU ops also carry their kernels' device time, which would double count).
        events = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total > 0]
        total = sum(e.self_device_time_total for e in events)
        cats = {}
        for e in events:
            n = e.key.lower()
            c = ('attention' if any(k in n for k in ('flash', 'fmha', 'attention')) else
                 'nccl' if 'nccl' in n else
                 'gemm' if any(k in n for k in ('gemm', 'cutlass', 'xmma', 'nvjet', 'sm90_', 'cublas')) else
                 'triton_fused' if 'triton' in n else
                 'complex_rope' if 'complex' in n else
                 'eager_elementwise' if 'elementwise' in n or 'reduce_kernel' in n else
                 'optimizer' if any(k in n for k in ('adam', 'foreach', 'multi_tensor')) else
                 'copy' if any(k in n for k in ('copy', 'memcpy', 'memset')) else 'other_eager')
            cats[c] = cats.get(c, 0) + e.self_device_time_total
        report['profile_gpu_busy_ms'] = round(total / 1e3, 1)
        print(json.dumps(dict(gpu_busy_ms=round(total / 1e3, 1), wall_ms=round(1e3 * report['s_per_update_median'], 1), categories_pct={k: round(100 * v / total, 1) for k, v in sorted(cats.items(), key=lambda x: -x[1])})), flush=True)
        for e in sorted(events, key=lambda e: -e.self_device_time_total)[:25]:
            print(f'{100 * e.self_device_time_total / total:6.2f}% {e.self_device_time_total / 1e3:9.1f}ms {e.count:6d}  {e.key[:110]}', flush=True)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
