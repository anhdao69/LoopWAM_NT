import time, json, numpy as np, torch, sys
sys.path.insert(0, "scripts")
from gr1_precompute_latents import decode_rgb
from fastwam.datasets.robocasa_gr1 import preprocess_frames_gr1
from fastwam.datasets.chronoloop_robot import VIDEO_OFFSETS
from fastwam.models.wan22.loopwam_init import load_wan21_vae
D="/mnt/data/vmo-ai-task/anhdh35/datasets/robocasa_gr1/gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000/videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
t=time.time(); rgb=decode_rgb(D, 10**9 - 1) if False else None
import av
t=time.time(); fr=[]
with av.open(D) as c:
    for f in c.decode(c.streams.video[0]): fr.append(f.to_ndarray(format="rgb24"))
dec=time.time()-t; T=len(fr)
frames=preprocess_frames_gr1(torch.from_numpy(np.stack(fr)).permute(0,3,1,2)).cuda()
vae=load_wan21_vae("checkpoints/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", device="cuda", dtype=torch.bfloat16)
off=torch.tensor(VIDEO_OFFSETS)
res={}
for B in (32,64,128):
    n=0; torch.cuda.synchronize(); t=time.time()
    for lo in range(0, min(T, 4*B), B):
        a=torch.arange(lo, min(lo+B,T)); idx=torch.clamp(a[:,None]+off, max=T-1)
        v=frames[idx.cuda()].permute(0,2,1,3,4).contiguous()
        with torch.no_grad(): lat=vae.model.encode(v, vae.scale); vae.model.clear_cache()
        n+=len(a)
    torch.cuda.synchronize(); el=time.time()-t
    res[B]=dict(windows_per_s=round(n/el,1), peak_gb=round(torch.cuda.max_memory_allocated()/1e9,1))
print(json.dumps(dict(frames=T, decode_s=round(dec,2), decode_fps=round(T/dec,1), latent=list(lat.shape), enc=res)))
