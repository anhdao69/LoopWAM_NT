"""Spot-check the LIBERO -> LiLa-WAM HDF5 conversion against the source LeRobot data on randomly chosen episodes:
actions (gripper mapped 1-2g), states, frame counts, and every image of both cameras (re-decoded source frame,
INTER_AREA 256x256, must equal the stored PNG exactly). Usage: verify_conversion.py <out.json> [n=24]"""
import json, random, sys
import av, cv2, h5py, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "/groups/yshang/an221229/checkpoints/wam_baselines/lilawam")
from convert_ours_to_lilawam_hdf5 import SRC, SUITES, task_dir_name
OUT = "/groups/yshang/an221229/data/LiLaWAM_libero_ours_hdf5"
jobs = []
for s in SUITES:
    for line in open(f"{SRC}/{s}_no_noops_lerobot/meta/episodes.jsonl"):
        e = json.loads(line); jobs.append((s, e["episode_index"], e["tasks"][0], e["length"]))
random.seed(0); pick = random.sample(jobs, int(sys.argv[2]) if len(sys.argv) > 2 else 24)
res = []
for s, ep, task, length in pick:
    d = f"{SRC}/{s}_no_noops_lerobot"
    t = pq.read_table(f"{d}/data/chunk-000/episode_{ep:06d}.parquet", columns=["action", "observation.state"]).to_pydict()
    a, st = np.asarray(t["action"]), np.asarray(t["observation.state"])
    with h5py.File(f"{OUT}/{s}/{task_dir_name(task)}/demo_clean/data/episode_{ep:05d}.hdf5") as h:
        ha, hs = h["action"][:], h["proprio"][:]
        ok = dict(len=len(ha) == len(a) == length, action6=bool(np.array_equal(ha[:, :6], a[:, :6])),
                  gripper=bool(np.array_equal(ha[:, 6], 1 - 2 * a[:, 6])), state=bool(np.array_equal(hs, st)))
        for cam, key in (("image", "observation/agentview/rgb"), ("wrist_image", "observation/robot0_eye_in_hand/rgb")):
            with av.open(f"{d}/videos/chunk-000/observation.images.{cam}/episode_{ep:06d}.mp4") as c:
                src = [cv2.resize(f.to_ndarray(format="rgb24"), (256, 256), interpolation=cv2.INTER_AREA) for f in c.decode(video=0)]
            stored = [cv2.cvtColor(cv2.imdecode(np.frombuffer(bytes(b), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for b in h[key][:]]
            ok[f"{cam}_frames"] = len(src) == len(stored) == length
            ok[f"{cam}_pixels_equal"] = all(np.array_equal(x, y) for x, y in zip(src, stored))
    res.append({"suite": s, "episode": ep, "length": length, **ok, "all_ok": all(ok.values())})
summary = {"episodes_checked": len(res), "all_ok": all(r["all_ok"] for r in res), "frames_checked": sum(r["length"] for r in res), "episodes": res}
json.dump(summary, open(sys.argv[1], "w"), indent=1)
print(summary["episodes_checked"], "episodes", summary["frames_checked"], "frames, all_ok =", summary["all_ok"])
