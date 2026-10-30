#!/usr/bin/env python
"""Convert OUR LIBERO training data (FastWAM/LoopWAM LeRobot v2.1, MuJoCo 3.3.2 no-noops regeneration, 4 suites)
into the per-episode HDF5 layout of the LiLa-WAM authors' LIBERO release (modelscope yangfan97/LiLa-WAM_Libero,
inspected 2026-10-30: <suite>/<task_name>/demo_clean/data/episode_XXXXX.hdf5 with
  action (T,7) float64 | proprio (T,8) float64 | observation/agentview/rgb (T,) PNG bytes 256x256 |
  observation/robot0_eye_in_hand/rgb (T,) PNG bytes 256x256).

Choices (each recorded in the file attrs):
- Images: our 512x512 AV1 frames (already upright, same orientation as the authors' frames) resized to 256x256
  with cv2.INTER_AREA and stored as lossless PNG, as in the authors' release.
- action: our 6 EEF deltas unchanged; gripper converted from our 0=close/1=open to LIBERO's raw -1=open/+1=close
  (raw = 1 - 2g), the same flip FastWAM's eval applies, so a trained policy emits raw environment actions.
- proprio: OUR 8-D observation.state (EEF pos 3, axis-angle 3, gripper joints 2) - what our matched evaluation
  feeds every model. (The authors' proprio is pos 3 + quaternion 4 + 1.)
- All episodes are kept (no stationary-segment cleaning).
Usage: convert_ours_to_lilawam_hdf5.py <out root> [--workers 8] [--limit N]
"""
import argparse, glob, json, os, re, sys
from multiprocessing import Pool
from pathlib import Path

SRC = "/groups/yshang/an221229/data/FastWAM/libero_mujoco3.3.2"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def task_dir_name(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def frames(path):
    import av, cv2
    out = []
    with av.open(path) as c:
        for fr in c.decode(video=0):
            img = cv2.resize(fr.to_ndarray(format="rgb24"), (256, 256), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            assert ok
            out.append(buf)
    return out


def convert(job):
    import h5py, numpy as np, pyarrow.parquet as pq
    suite, ep, task, out_root = job
    d = f"{SRC}/{suite}_no_noops_lerobot"
    out = Path(out_root) / suite / task_dir_name(task) / "demo_clean" / "data" / f"episode_{ep:05d}.hdf5"
    if out.exists():
        return str(out), "exists"
    t = pq.read_table(f"{d}/data/chunk-000/episode_{ep:06d}.parquet", columns=["action", "observation.state", "frame_index"]).to_pydict()
    action = np.asarray(t["action"], dtype=np.float64)
    state = np.asarray(t["observation.state"], dtype=np.float64)
    g = action[:, -1]
    if not np.isin(g, [0.0, 1.0]).all():
        raise ValueError(f"{suite} ep {ep}: gripper action not in {{0,1}}")
    action[:, -1] = 1.0 - 2.0 * g
    agent = frames(f"{d}/videos/chunk-000/observation.images.image/episode_{ep:06d}.mp4")
    wrist = frames(f"{d}/videos/chunk-000/observation.images.wrist_image/episode_{ep:06d}.mp4")
    n = len(action)
    if len(agent) != n or len(wrist) != n or list(t["frame_index"]) != list(range(n)):
        raise ValueError(f"{suite} ep {ep}: {n} rows, {len(agent)}/{len(wrist)} frames")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    vl = h5py.vlen_dtype(np.uint8)
    with h5py.File(tmp, "w") as h:
        h.create_dataset("action", data=action)
        h.create_dataset("proprio", data=state)
        for key, imgs in (("observation/agentview/rgb", agent), ("observation/robot0_eye_in_hand/rgb", wrist)):
            ds = h.create_dataset(key, (n,), dtype=vl)
            for i, b in enumerate(imgs):
                ds[i] = b.reshape(-1)
        h.attrs.update(source=f"{d} episode {ep}", suite=suite, task=task, image="256x256 PNG, INTER_AREA from 512x512",
                       gripper="raw LIBERO: -1 open, +1 close (= 1 - 2*ours)",
                       proprio="ours: eef pos(3), axis-angle(3), gripper qpos(2)")
    os.replace(tmp, out)
    return str(out), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_root"); p.add_argument("--workers", type=int, default=8); p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    jobs = []
    for suite in SUITES:
        d = f"{SRC}/{suite}_no_noops_lerobot"
        tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(f"{d}/meta/tasks.jsonl")}
        for line in open(f"{d}/meta/episodes.jsonl"):
            e = json.loads(line)
            assert len(e["tasks"]) == 1
            jobs.append((suite, e["episode_index"], e["tasks"][0], a.out_root))
    if a.limit:
        jobs = jobs[: a.limit]
    print(len(jobs), "episodes", flush=True)
    done = 0
    with Pool(a.workers) as pool:
        for path, n in pool.imap_unordered(convert, jobs, chunksize=2):
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(done, path, n, flush=True)
    print("DONE", done, flush=True)


if __name__ == "__main__":
    main()
