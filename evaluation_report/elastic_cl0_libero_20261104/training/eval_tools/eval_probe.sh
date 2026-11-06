#!/usr/bin/env bash
# Environment probe for the CL-0 matched LIBERO evaluation (run inside job 5623 on worker-2).
set -u
source /mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_cl0/plans/elastic_cl0/env.sh
echo "host $(hostname) job ${SLURM_JOB_ID:-?} cpus $(nproc) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv,noheader
which nvidia-cuda-mps-control || echo "no MPS binary"
free -g | head -2; df -h /tmp /dev/shm | tail -2
cat /sys/fs/cgroup/memory.max 2>/dev/null; for f in /sys/fs/cgroup/*/job_${SLURM_JOB_ID}/memory.current /sys/fs/cgroup/system.slice/*/job_${SLURM_JOB_ID}/memory.current; do [ -f "$f" ] && echo "$f $(cat $f)"; done 2>/dev/null
python - <<'EOF'
import importlib, importlib.metadata as md, sys
print("python", sys.version.split()[0], sys.executable)
for p in ("torch","torchvision","mujoco","robosuite","bddl","libero","numpy","imageio","easydict","gym","robomimic","hydra-core","transformers","egl_probe","PyOpenGL"):
    try: print(p, md.version(p))
    except Exception as e: print(p, "-", type(e).__name__)
import torch; print("cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
try:
    import libero, libero.libero as L; print("libero at", libero.__file__)
    from libero.libero import get_libero_path
    for k in ("bddl_files","init_states","assets","benchmark_root"): print(k, get_libero_path(k))
except Exception as e: print("libero import failed:", repr(e))
EOF
echo "LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-unset}"; ls ~/.libero 2>/dev/null && cat ~/.libero/config.yaml 2>/dev/null
ls /mnt/data/vmo-ai-task/anhdh35/FastWAM/third_party/LIBERO | head
