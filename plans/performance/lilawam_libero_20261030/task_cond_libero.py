"""Task-condition vectors as in LiLa-WAM utils/precompute_task_cond.py (frozen DINOv3, last-layer CLS of the last
frame minus the first frame, averaged over a task's episodes), on the same [agentview | wrist] composite the policy
sees. Writes <out>/<suite>/<task>/task_cond.npy. Usage: task_cond_libero.py <data root> <out dir> <dinov3 dir>"""
import glob, json, os, sys
from collections import defaultdict
import h5py, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero_data import composite, decode_rgb
from transformers import AutoModel
root, out, ck = sys.argv[1:4]
sys.path.insert(0, "/groups/yshang/an221229/code/wam_baselines_20261030/LiLa-WAM")
from utils.precompute_task_cond import preprocess_image
m = AutoModel.from_pretrained(ck, torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
diffs = defaultdict(list)
for p in sorted(glob.glob(f"{root}/*/*/demo_clean/data/*.hdf5")):
    task = "/".join(p.split("/")[-5:-3])
    with h5py.File(p, "r") as f:
        a, w = f["observation/agentview/rgb"], f["observation/robot0_eye_in_hand/rgb"]
        ims = [composite(decode_rgb(a[i]), decode_rgb(w[i])) for i in (0, len(a) - 1)]
    px = torch.from_numpy(np.concatenate([preprocess_image(x, (512, 256)) for x in ims])).cuda().to(torch.bfloat16)
    with torch.no_grad():
        cls = m(pixel_values=px, return_dict=True).last_hidden_state[:, 0].float().cpu().numpy()
    diffs[task].append(cls[1] - cls[0])
for task, d in diffs.items():
    os.makedirs(f"{out}/{task}", exist_ok=True)
    np.save(f"{out}/{task}/task_cond.npy", np.mean(d, 0).astype(np.float32))
json.dump({"tasks": {t: len(d) for t, d in diffs.items()}, "dim": int(len(next(iter(diffs.values()))[0])),
           "image": "[agentview | wrist] 512x256"}, open(f"{out}/meta.json", "w"), indent=1)
print(len(diffs), "tasks", sum(len(d) for d in diffs.values()), "episodes")
