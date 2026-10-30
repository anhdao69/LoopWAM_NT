"""Open-loop sanity check of a trained LiLa-WAM LIBERO checkpoint on demonstration frames (held-in training data).

Builds the model exactly as the public robotwin_infer.RobotWinInference does (config, frozen DINOv3, action model,
strict state-dict load, bf16) and samples action chunks with its own procedure (Gaussian x_0, 10 Euler steps of the
flow, denormalize, optional B-spline smoothing), batched. Inputs are the training dataset's own samples
(libero_data.LiberoTaskDataset: [agentview | wrist] composite, 8-D state, task-condition vector), so this checks that
the checkpoint + stats + task vectors + inference path reproduce the demonstrations. It is NOT a success-rate metric:
the frames were seen in training. Baseline row: always predicting the suite's mean action.
Usage: openloop_check.py --config <stage yaml> --ckpt <.pt> --stats <json> --out <json> [--n 512] [--seed 0]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import libero_data  # noqa: E402  (puts the LiLa-WAM repo on sys.path)
from omegaconf import OmegaConf  # noqa: E402
import robotwin_infer as R  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--ckpt", required=True)
    p.add_argument("--stats", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=512); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    cfg = OmegaConf.load(a.config)
    t0 = time.time()
    inf = R.RobotWinInference(config_path=a.config, checkpoint_path=a.ckpt, norm_stats_path=a.stats,
                              device="cuda", dtype=torch.bfloat16, task_name=None)
    m, dev, dt_ = inf.model, inf.device, inf.dtype
    ds = libero_data.create_dataset(cfg, val=True)
    idx = np.random.RandomState(a.seed).choice(len(ds), size=min(a.n, len(ds)), replace=False)
    L, D, steps_n = cfg.common.action_chunk_size, cfg.common.action_dim, cfg.common.num_inference_steps
    stats = json.load(open(a.stats))["robotwin2"]
    mean_action = np.asarray(stats["action"]["mean"], dtype=np.float32)
    preds, preds_smooth, gts, masks = [], [], [], []
    for b0 in range(0, len(idx), a.batch):
        items = [ds[int(i)] for i in idx[b0:b0 + a.batch]]
        B = len(items)
        px = torch.stack([it["pixel_values"] for it in items])
        state = torch.stack([it["state"] for it in items]).to(dev, dt_)
        tc = torch.stack([it["task_cond"] for it in items]).to(dev, dt_)
        with torch.no_grad():
            qpos = m.normalize_state(state if state.dim() == 3 else state.unsqueeze(1))
            feats = m.get_vision_features(px)
            x = torch.randn((B, L, D), device=dev, dtype=dt_)
            ts = torch.linspace(0, 1, steps_n + 1, device=dev, dtype=dt_)
            for i in range(steps_n):
                out = m.action_model(t=ts[i].repeat(B), noisy_actions=x, qpos_history=qpos,
                                     dino_features_list=feats, task_cond=tc)
                x = x + out["final_pred"] * (ts[i + 1] - ts[i])
            act = m.denormalize_action(x).float().cpu().numpy()
        preds.append(act)
        preds_smooth.append(np.stack([R.bspline_smooth(c, degree=3, num_ctrl_pts=8) for c in act]))
        gts.append(torch.stack([it["action_sequence"] for it in items]).numpy())
        masks.append(torch.stack([it["action_mask"] for it in items]).numpy())
    P, PS, G, M = map(np.concatenate, (preds, preds_smooth, gts, masks))
    w = M[..., None].astype(np.float32)

    def metrics(pred):
        err = np.abs(pred - G) * w
        mae6 = float(err[..., :6].sum() / (w.sum() * 6))
        grip_ok = (np.sign(pred[..., 6]) == np.sign(G[..., 6])) & M
        first = np.abs(pred[:, 0, :6] - G[:, 0, :6]).mean()
        return {"mae_xyzrot": mae6, "mae_xyzrot_first_action": float(first),
                "gripper_sign_agreement": float(grip_ok.sum() / M.sum()),
                "mae_per_dim": [float(err[..., d].sum() / w.sum()) for d in range(D)]}

    res = {"checkpoint": a.ckpt, "config": a.config, "stats": a.stats, "samples": int(len(idx)), "seed": a.seed,
           "valid_action_steps": int(M.sum()), "model_ms_per_sample": None, "seconds": round(time.time() - t0, 1),
           "raw": metrics(P), "bspline_smoothed": metrics(PS),
           "max_abs_raw_minus_smoothed": float(np.abs(P - PS).max()),
           "mean_abs_raw_minus_smoothed": float(np.abs(P - PS).mean()),
           "baseline_mean_action": metrics(np.broadcast_to(mean_action, G.shape).copy())}
    json.dump(res, open(a.out, "w"), indent=1)
    for k in ("raw", "bspline_smoothed", "baseline_mean_action"):
        r = res[k]
        print(f"{k:22s} MAE(6 deltas)={r['mae_xyzrot']:.4f}  first-step MAE={r['mae_xyzrot_first_action']:.4f}  "
              f"gripper sign agreement={r['gripper_sign_agreement']:.4f}")


if __name__ == "__main__":
    main()
