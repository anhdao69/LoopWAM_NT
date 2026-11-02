#!/usr/bin/env python3
"""GPU check that the re-wrapped CL-0 teacher (loopwam-s-v1) is the same policy as CL-0 itself.

Run with PYTHONPATH pointing at a checkout of the `chrono` branch (it provides chronoloop.py and, like every
branch, create_loopwam). Loads CL-0 through create_chronoloop (its evaluation path, chrono_infer_action) and the
re-wrap through create_loopwam (LoopWAM.infer_action), feeds both the same seeded image, proprio and text context
with the same seeded noise, at 10 Euler steps under BF16 autocast as in evaluation, and reports bit equality and
the max abs difference of the 32x7 action chunks.
"""
import argparse, json
from dataclasses import asdict
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chronoloop", required=True); p.add_argument("--converted", required=True)
    p.add_argument("--vae-path", required=True); p.add_argument("--text-cache", help="optional .pt text embedding")
    p.add_argument("--queries", type=int, default=4); p.add_argument("--out", required=True)
    a = p.parse_args()
    from fastwam.models.wan22.chronoloop import ChronoConfig, create_chronoloop
    from fastwam.models.wan22.loopwam import create_loopwam
    dev = "cuda"
    payload = torch.load(a.chronoloop, map_location="cpu", weights_only=False, mmap=True)
    cfg = ChronoConfig(**payload["chrono"]); del payload
    chrono, _ = create_chronoloop(cfg, a.vae_path, checkpoint_path=a.chronoloop, device=dev)
    plain = create_loopwam(checkpoint_path=a.converted, vae_path=a.vae_path, device=dev)
    chrono.eval(); plain.eval()
    weights_equal = (chrono.mot.state_dict().keys() == plain.mot.state_dict().keys() and all(
        torch.equal(v, plain.mot.state_dict()[k]) for k, v in chrono.mot.state_dict().items()))
    rows = []
    for q in range(a.queries):
        g = torch.Generator().manual_seed(1000 + q)
        image = (torch.rand(3, 224, 448, generator=g) * 2 - 1)
        proprio = torch.randn(8, generator=g)
        context = torch.randn(128, 4096, generator=g) * 0.2
        mask = torch.zeros(128, dtype=torch.bool); mask[:12 + 3 * q] = True
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = chrono.chrono_infer_action(image, proprio, context, mask, num_inference_steps=10, seed=q)["action"]
            y = plain.infer_action(prompt=None, input_image=image, action_horizon=32, proprio=proprio, context=context,
                                   context_mask=mask, num_inference_steps=10, seed=q)["action"]
        x, y = x.float().cpu(), y.float().cpu()
        rows.append(dict(query=q, bit_equal=torch.equal(x, y), max_abs_diff=(x - y).abs().max().item(),
                         action_abs_mean=x.abs().mean().item()))
        print(json.dumps(rows[-1]), flush=True)
    result = dict(chrono=asdict(cfg), mot_weights_equal=weights_equal, rows=rows,
                  all_bit_equal=all(r["bit_equal"] for r in rows))
    open(a.out, "w").write(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
