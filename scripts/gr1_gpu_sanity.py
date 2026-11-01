#!/usr/bin/env python3
"""Single-GPU sanity check of the GR1 v0 model (no dataset needed).

Builds the model from the canonical Wan2.1 donors, checks the parameter count and the
structured F/U/A attention with 256 observation tokens, runs one training forward/backward per
microbatch size with random cached latents (peak memory), and one 10-step infer_action on a
256x256 image (32x29 finite output).
"""
import argparse
import json
import time

import torch

from fastwam.datasets.robocasa_gr1 import LATENT_SHAPE, OBS_TOKENS, ROBOT_DIM
from fastwam.models.wan22.loopwam_gr1 import PARAMETERS, RobotTrainForward, create_loopwam_gr1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init-artifact', default='checkpoints/LoopWAM/wan21_compact_donors.pt')
    p.add_argument('--vae-path', default='checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth')
    p.add_argument('--micro', default='8,16')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    dev = torch.device('cuda')
    model, _ = create_loopwam_gr1(a.vae_path, init_artifact=a.init_artifact, device=dev, checkpoint_blocks=False, seed=42)
    params = model.policy_parameters()
    n = sum(x.numel() for x in params)
    assert n == PARAMETERS, n
    model.train(); model.mot.structured_attention = True; model.mot.structured_attention_observation_tokens = OBS_TOKENS
    runner = RobotTrainForward(model)
    report = dict(parameters=n, micro={})
    for mb in [int(x) for x in a.micro.split(',')]:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        g = torch.Generator(device=dev).manual_seed(0)
        mask = torch.zeros(mb, 128, dtype=torch.bool, device=dev); mask[:, :20] = True
        sample = dict(latents=torch.randn(mb, *LATENT_SHAPE, generator=g, device=dev).bfloat16().float(),
                      video=torch.zeros((), device=dev).expand(mb, 3, 9, 256, 256),
                      action=torch.randn(mb, 32, ROBOT_DIM, generator=g, device=dev),
                      proprio=torch.randn(mb, 32, ROBOT_DIM, generator=g, device=dev),
                      action_is_pad=torch.zeros(mb, 32, dtype=torch.bool, device=dev),
                      image_is_pad=torch.zeros(mb, 9, dtype=torch.bool, device=dev),
                      context=torch.randn(mb, 128, 4096, generator=g, device=dev) * mask[..., None], context_mask=mask)
        noise = dict(video=torch.randn(mb, *LATENT_SHAPE, generator=g, device=dev),
                     action=torch.randn(mb, 32, ROBOT_DIM, generator=g, device=dev),
                     t_video=torch.rand(mb, generator=g, device=dev) * 1000, t_action=torch.rand(mb, generator=g, device=dev) * 1000)
        t = time.time()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, logs, _ = runner([dict(sample=sample, noise=noise)])
        loss.backward()
        torch.cuda.synchronize()
        missing = [i for i, x in enumerate(params) if x.grad is None or not torch.isfinite(x.grad).all()]
        assert not missing, f'{len(missing)} parameters without finite gradients'
        report['micro'][mb] = dict(loss_per_window=float(loss) / mb, seconds=round(time.time() - t, 3),
                                   peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        model.zero_grad(set_to_none=True)
        print(json.dumps({mb: report['micro'][mb]}), flush=True)
    model.eval()
    image = torch.rand(1, 3, 256, 256, device=dev) * 2 - 1
    ctx = torch.zeros(1, 128, 4096, device=dev); m = torch.zeros(1, 128, dtype=torch.bool, device=dev); m[:, :20] = True
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        out = model.infer_action(input_image=image, proprio=torch.zeros(1, ROBOT_DIM, device=dev), context=ctx, context_mask=m,
                                 action_horizon=32, num_inference_steps=10, seed=42, rand_device='cpu')
    act = out['action']
    assert act.shape == (32, ROBOT_DIM) and torch.isfinite(act).all()
    report['inference'] = dict(shape=list(act.shape), finite=True)
    print(json.dumps(report), flush=True)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(report, f, indent=2)


if __name__ == '__main__':
    main()
