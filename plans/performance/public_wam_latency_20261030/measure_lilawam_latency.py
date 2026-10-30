#!/usr/bin/env python
"""Batch-one latency of the public LiLa-WAM RoboTwin 2.0 release, timed through its own robotwin_infer.py.

LiLa-WAM has no released LIBERO checkpoint, so this times the RoboTwin 2.0 (50-task) release with its own config:
one head camera resized by its processor to 320x240, a 16-D dual-arm state, a 32x14 action chunk, ten flow steps,
B-spline smoothing. A query is what RobotWinInference.step() does when it re-plans: update_state_buffer() plus
_predict_chunk() (preprocess -> DINOv3 ViT-L/16 features -> ten action-expert steps -> denormalize -> smoothing).

Inputs are built from the same 500 recorded LIBERO observations as the other models: the 256x256 agent-view image
is the head-camera frame, and the end-effector pose / gripper fill both arms' 16-D state. Values do not change
the computation; shapes follow the RoboTwin release. torch.cuda.synchronize() precedes each clock start; the clock
stops when the smoothed host action array exists.
Usage: measure_lilawam_latency.py --out <json> [--n 500] [--warmup 20] [--task adjust_bottle]
"""
import argparse, hashlib, json, os, pickle, socket, subprocess, sys, time
from pathlib import Path

REPO = Path("/groups/yshang/an221229/code/wam_baselines_20261030/LiLa-WAM")
CK = Path("/groups/yshang/an221229/checkpoints/wam_baselines/lilawam")
OBS = "/groups/yshang/an221229/checkpoints/LoopWAM_NT/latency_20261030/observations_500.pkl"
OBS_SHA = "5815c6fdc22d2dfe0c241f89c687a6f2e83804f9bd298332867258eb7160e0f0"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=500)
    p.add_argument("--warmup", type=int, default=20); p.add_argument("--task", default="adjust_bottle")
    p.add_argument("--device", default="cuda")  # cpu only for a smoke test
    a = p.parse_args()
    raw = Path(OBS).read_bytes()
    if hashlib.sha256(raw).hexdigest() != OBS_SHA:
        raise SystemExit("observations file hash mismatch")
    records = pickle.loads(raw)[: a.n]

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    import transformers
    from omegaconf import OmegaConf
    import robotwin_infer as R

    # The released config.yaml with only its two machine-local paths pointed at this cluster's copies.
    cfg = OmegaConf.load(CK / "robotwin" / "config.yaml")
    cfg.model.vision_encoder.checkpoint_path = str(CK / "dinov3_pretrain" / "dinov3-vitl16-pretrain-lvd1689m")
    cfg.dataset.task_cond_dir = str(REPO / "data-500-taskcond")
    cfg_path = Path(a.out).with_suffix(".config.yaml")
    OmegaConf.save(cfg, cfg_path)

    torch.manual_seed(42); np.random.seed(42)
    t0 = time.perf_counter()
    inf = R.RobotWinInference(config_path=str(cfg_path), checkpoint_path=str(CK / "robotwin" / "checkpoint.pt"),
                              norm_stats_path=str(REPO / "utils" / "stat-500-all.json"), device=a.device,
                              dtype=torch.bfloat16, task_name=a.task)
    load_s = time.perf_counter() - t0

    def robotwin_obs(o):
        pose = np.concatenate([o["robot0_eef_pos"], o["robot0_eef_quat"]]).astype(np.float32)
        grip = float(o["robot0_gripper_qpos"][0])
        return {"observation": {"head_camera": {"rgb": np.ascontiguousarray(o["agentview_image"][::-1, ::-1])}},
                "endpose": {"left_endpose": pose, "left_gripper": grip, "right_endpose": pose, "right_gripper": grip}}

    obs = [robotwin_obs(r["obs"]) for r in records]

    cuda = a.device.startswith("cuda")
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    def query(o):
        sync()
        t = time.perf_counter()
        inf.processor.update_state_buffer(o)
        act = inf._predict_chunk(o)
        return act, (time.perf_counter() - t) * 1e3

    inf.reset()
    for o in obs[: a.warmup]:
        query(o)
    inf.reset()
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    times, shape = [], None
    for o in obs:
        act, dt = query(o)
        times.append(dt); shape = list(act.shape)
    t = np.asarray(times)
    res = dict(model="LiLa-WAM (RoboTwin 2.0 release)", repo=str(REPO),
               repo_commit=subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                          text=True).stdout.strip(),
               checkpoint=str(CK / "robotwin" / "checkpoint.pt"), checkpoint_source="modelscope yangfan97/LiLa-WAM_RoboTwin2_0",
               vision_encoder="facebook/dinov3-vitl16-pretrain-lvd1689m", task_cond=a.task, dtype="bfloat16",
               num_inference_steps=int(cfg.common.num_inference_steps), action_chunk_shape=shape,
               image_size_wh=list(cfg.dataset.image_size), smooth_actions=bool(cfg.inference.smooth_actions),
               parameters=sum(p.numel() for p in inf.model.parameters()), load_seconds=load_s,
               host=socket.gethostname(), gpu=torch.cuda.get_device_name(0) if cuda else "cpu", torch=torch.__version__,
               transformers=transformers.__version__, observations_sha256=OBS_SHA, warmup=a.warmup,
               slurm_job=os.environ.get("SLURM_JOB_ID"),
               modes={"native": dict(n=len(t), mean_ms=float(t.mean()), median_ms=float(np.median(t)),
                                     p5_ms=float(np.percentile(t, 5)), p95_ms=float(np.percentile(t, 95)),
                                     std_ms=float(t.std()), min_ms=float(t.min()), max_ms=float(t.max()),
                                     peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9 if cuda else None,
                                     per_query_ms=[round(x, 3) for x in times])})
    print(json.dumps({k: v for k, v in res["modes"]["native"].items() if k != "per_query_ms"}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
