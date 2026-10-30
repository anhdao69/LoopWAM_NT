"""Action/state min-max over every frame of the converted LIBERO data, in the JSON layout LiLa-WAM's
load_norm_stats reads (top key 'robotwin2' is hard-coded there). Usage: stats_libero.py <data root> <out.json> [suite]"""
import glob, json, sys, time
import h5py, numpy as np
SUITE = sys.argv[3] if len(sys.argv) > 3 else "*"  # optional: one suite
files = sorted(glob.glob(f"{sys.argv[1]}/{SUITE}/*/demo_clean/data/*.hdf5"))
A, S = [], []
for p in files:
    with h5py.File(p, "r") as f:
        A.append(f["action"][:]); S.append(f["proprio"][:])
A, S = np.concatenate(A), np.concatenate(S)
st = lambda x: {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(), "std": x.std(0).tolist()}
json.dump({"robotwin2": {"meta": {"dataset": sys.argv[1], "files": len(files), "frames": len(A),
                                   "note": "OUR LIBERO (4 suites); key name required by LiLa-WAM load_norm_stats",
                                   "timestamp": time.strftime("%F %T")},
                          "action": st(A), "state": st(S)}}, open(sys.argv[2], "w"), indent=1)
print(len(files), "files", len(A), "frames"); print("action min", np.round(A.min(0), 3), "max", np.round(A.max(0), 3))
