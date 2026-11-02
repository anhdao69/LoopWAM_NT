#!/usr/bin/env python3
"""Per-variant GPU smoke for RoboTwin-10: construction, gradients, save/load round trip.

python scripts/smoke_robotwin10_variants.py --out DIR [--variants v0-video4-action4,dense-s30] [--batch 2]

For each variant: fresh canonical-donor model with CPU-seeded 14-D interfaces; parameter count; interfaces equal
across variants; one BF16 forward/backward of the training wrapper on random cached-latent-shaped inputs with
keyed noise (every policy parameter must receive a finite gradient, as DDP runs without unused-parameter
search); save_robot -> create_loopwam_robot(checkpoint_path) restores every tensor exactly and the payload
validates as the same variant.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from fastwam.datasets.chronoloop_robot import LATENT_SHAPE, OBS_TOKENS
from fastwam.models.wan22.loopwam_robot import (VARIANTS, RobotTrainForward, create_loopwam_robot, payload_variant,
                                                robot_parameter_count, validate_robot_payload)
from train_loopwam_robotwin10 import slot_noise, slot_seed


def fake_sample(batch, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    pad = torch.zeros(batch, 32, dtype=torch.bool); pad[0, 28:] = True
    mask = torch.zeros(batch, 128, dtype=torch.bool); mask[:, :20] = True
    s = dict(action=torch.randn(batch, 32, 14, generator=g), proprio=torch.randn(batch, 32, 14, generator=g),
             action_is_pad=pad, image_is_pad=torch.zeros(batch, 9, dtype=torch.bool),
             context=torch.randn(batch, 128, 4096, generator=g).to(torch.bfloat16) * mask[..., None],
             context_mask=mask, latents=torch.randn(batch, *LATENT_SHAPE, generator=g).to(torch.bfloat16).float())
    s = {k: v.to(device) for k, v in s.items()}
    s['video'] = torch.zeros((), device=device).expand(batch, 3, 9, 384, 320)
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--variants', default=','.join(VARIANTS))
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--init-artifact', default='checkpoints/LoopWAM/wan21_compact_donors.pt')
    p.add_argument('--vae-path', default='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda', torch.cuda.current_device())
    results, interfaces = {}, None
    for variant in a.variants.split(','):
        t0 = time.time(); torch.cuda.reset_peak_memory_stats()
        model, _ = create_loopwam_robot(a.vae_path, init_artifact=a.init_artifact, device=device, checkpoint_blocks=False,
                                        seed=42, variant=variant)
        model.train(); model.mot.structured_attention = True; model.mot.structured_attention_observation_tokens = OBS_TOKENS
        params = model.policy_parameters(); count = sum(x.numel() for x in params)
        assert count == robot_parameter_count(variant), (variant, count)
        io = {k: v.detach().cpu().clone() for layer, name in ((model.action_expert.action_encoder, 'enc'),
              (model.action_expert.head, 'head'), (model.proprio_encoder, 'prop')) for k, v in
              ((f'{name}.{n}', t) for n, t in layer.state_dict().items())}
        if interfaces is None:
            interfaces = io
        assert io.keys() == interfaces.keys() and all(torch.equal(io[k], interfaces[k]) for k in io), 'interfaces differ'
        sample = fake_sample(a.batch, device)
        noise = slot_noise(model, [slot_seed(42, 0, i, 0) for i in range(a.batch)], device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, logs, _ = RobotTrainForward(model)([dict(sample=sample, noise=noise)])
        loss.backward()
        missing = [n for n, q in model.named_parameters() if q.requires_grad and q.grad is None]
        nonfinite = [n for n, q in model.named_parameters() if q.grad is not None and not torch.isfinite(q.grad).all()]
        assert not missing and not nonfinite, (variant, missing[:3], nonfinite[:3])
        assert torch.isfinite(loss) and all(torch.isfinite(v) for v in logs.values())
        path = out / f'{variant}.pt'
        model.save_robot(path, step=0, training_state=dict(smoke=True))
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith('vae.')}
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        validate_robot_payload(payload, variant); assert payload_variant(payload) == variant
        header = {k: payload[k] for k in ('version', 'video_loops', 'action_loops', 'inference_loops', 'trained_max_loops',
                                          'loop_alignment', 'action_kv_mode')}
        peak = torch.cuda.max_memory_allocated() / 1e9
        del model, params, sample, noise, loss, logs, payload; torch.cuda.empty_cache()
        restored, _ = create_loopwam_robot(a.vae_path, checkpoint_path=path, device='cpu', checkpoint_blocks=False)
        rstate = {k: v for k, v in restored.state_dict().items() if not k.startswith('vae.')}
        assert rstate.keys() == state.keys() and all(torch.equal(rstate[k], state[k]) for k in state), 'round trip differs'
        assert (restored.version, restored.mot.loops, restored.mot.action_loops) == VARIANTS[variant]
        del restored, rstate, state; path.unlink()
        results[variant] = dict(parameters=count, header=header, peak_allocated_gb=round(peak, 2), seconds=round(time.time() - t0, 1),
                                round_trip_exact=True, all_gradients_finite=True)
        print(json.dumps({variant: results[variant]}), flush=True)
    (out / 'smoke_variants.json').write_text(json.dumps(dict(results=results, interfaces_identical=True), indent=2))


if __name__ == '__main__':
    main()
