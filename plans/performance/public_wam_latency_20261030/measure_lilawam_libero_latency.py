#!/usr/bin/env python
"""Batch-one latency of LiLa-WAM trained on OUR LIBERO data (paper recipe), on the same 500 raw LIBERO observations.

The model is built exactly as the public robotwin_infer.RobotWinInference builds it (config, frozen DINOv3 ViT-L/16,
action model, strict state-dict load, bf16). One query mirrors RobotWinInference._predict_chunk with LIBERO inputs
instead of RoboTwin ones:
  raw obs -> libero_obs_to_inputs (both views rotated 180 deg, 256x256 INTER_AREA, [agentview | wrist] 512x256;
  8-D state) -> ImageNet normalize -> GPU -> normalize state -> DINOv3 multi-layer features -> 10 Euler flow steps
  of the action model -> denormalize -> host -> B-spline smoothing (public config smooth_actions=true).
torch.cuda.synchronize() precedes each clock start; the clock stops when the smoothed host action array exists.
All four suite policies share one architecture, so one suite's policy is timed on all 500 observations.
Usage: measure_lilawam_libero_latency.py --config <stage2 yaml> --ckpt <.pt> --stats <json> --out <json>
"""
import argparse, hashlib, json, os, pickle, socket, subprocess, sys, time
from pathlib import Path

OBS = "/groups/yshang/an221229/checkpoints/LoopWAM_NT/latency_20261030/observations_500.pkl"
OBS_SHA = "5815c6fdc22d2dfe0c241f89c687a6f2e83804f9bd298332867258eb7160e0f0"
ADAPTER = "/groups/yshang/an221229/code/lilawam_libero"
TASKCOND = "/groups/yshang/an221229/data/LiLaWAM_libero_ours_taskcond"


def main():
    p = argparse.ArgumentParser()
    for k in ("--config", "--ckpt", "--stats", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--n", type=int, default=500); p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--suite", default="libero_10")
    a = p.parse_args()
    raw = Path(OBS).read_bytes()
    if hashlib.sha256(raw).hexdigest() != OBS_SHA:
        raise SystemExit("observations file hash mismatch")
    records = pickle.loads(raw)[: a.n]
    sys.path.insert(0, ADAPTER)
    import numpy as np
    import torch
    import transformers
    from omegaconf import OmegaConf
    import libero_data as LD  # puts the LiLa-WAM repo on sys.path
    import robotwin_infer as R
    from dataloader.dataset import _normalize_image

    cfg = OmegaConf.load(a.config)
    torch.manual_seed(42); np.random.seed(42)
    t0 = time.perf_counter()
    inf = R.RobotWinInference(config_path=a.config, checkpoint_path=a.ckpt, norm_stats_path=a.stats, device="cuda",
                              dtype=torch.bfloat16, task_name=None)
    load_s = time.perf_counter() - t0
    m, dev, dt_ = inf.model, inf.device, inf.dtype
    task_dir = sorted(os.listdir(f"{TASKCOND}/{a.suite}"))[0]
    inf.task_cond = torch.from_numpy(np.load(f"{TASKCOND}/{a.suite}/{task_dir}/task_cond.npy"))[None].to(dev, dt_)
    L, D, S = cfg.common.action_chunk_size, cfg.common.action_dim, cfg.common.num_inference_steps

    def query(obs):
        torch.cuda.synchronize()
        t = time.perf_counter()
        img, state = LD.libero_obs_to_inputs(obs)
        px = torch.from_numpy(_normalize_image(img))[None].to(dev, dt_)
        with torch.no_grad():
            qpos = m.normalize_state(torch.from_numpy(state)[None, None].to(dev, dt_))
            feats = m.get_vision_features(px)
            x = torch.randn((1, L, D), device=dev, dtype=dt_)
            ts = torch.linspace(0, 1, S + 1, device=dev, dtype=dt_)
            for i in range(S):
                out = m.action_model(t=ts[i].unsqueeze(0), noisy_actions=x, qpos_history=qpos,
                                     dino_features_list=feats, task_cond=inf.task_cond)
                x = x + out["final_pred"] * (ts[i + 1] - ts[i])
            act = m.denormalize_action(x)[0].float().cpu().numpy()
        if inf.smooth_actions:
            act = R.bspline_smooth(act, degree=3, num_ctrl_pts=8)
        return act, (time.perf_counter() - t) * 1e3

    for r in records[: a.warmup]:
        query(r["obs"])
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times, shape, acts = [], None, []
    for r in records:
        act, dt = query(r["obs"])
        times.append(dt); shape = list(act.shape); acts.append(act)
    t = np.asarray(times)
    A = np.stack(acts)
    res = dict(model=f"LiLa-WAM (ours, LIBERO paper recipe, {a.suite} policy)", repo=str(Path(R.__file__).parent),
               repo_commit=subprocess.run(["git", "-C", str(Path(R.__file__).parent), "rev-parse", "HEAD"],
                                          capture_output=True, text=True).stdout.strip(),
               checkpoint=a.ckpt, config=a.config, stats=a.stats, task_cond=f"{a.suite}/{task_dir}", dtype="bfloat16",
               num_inference_steps=int(S), action_chunk_shape=shape, image_size_wh=list(cfg.dataset.image_size),
               smooth_actions=bool(inf.smooth_actions),
               parameters=sum(q.numel() for q in m.parameters()),
               parameters_action_model=sum(q.numel() for q in m.action_model.parameters()),
               load_seconds=load_s, host=socket.gethostname(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
               transformers=transformers.__version__, observations_sha256=OBS_SHA, warmup=a.warmup,
               slurm_job=os.environ.get("SLURM_JOB_ID"),
               actions_finite=bool(np.isfinite(A).all()), actions_abs_max=float(np.abs(A).max()),
               modes={"native": dict(n=len(t), mean_ms=float(t.mean()), median_ms=float(np.median(t)),
                                     p5_ms=float(np.percentile(t, 5)), p95_ms=float(np.percentile(t, 95)),
                                     std_ms=float(t.std()), min_ms=float(t.min()), max_ms=float(t.max()),
                                     peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                                     per_query_ms=[round(x, 3) for x in times])})
    print(json.dumps({k: v for k, v in res["modes"]["native"].items() if k != "per_query_ms"}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
