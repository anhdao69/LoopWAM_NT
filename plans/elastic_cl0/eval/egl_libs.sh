#!/usr/bin/env bash
# GPU EGL for MuJoCo on worker-2. The container carries only the compute part of NVIDIA driver 570.158.01 (no
# libEGL_nvidia, no glvnd vendor files), which is why this cluster's earlier LIBERO runs used OSMesa. The reference
# evaluations rendered with MUJOCO_GL=egl, so this copies the EGL/GL user-space subset of the same driver release,
# as already extracted (archive integrity verified) by the RoboTwin eval runtime, into $EVAL_OUT/egl_libs, read-only
# from the source; nothing in the RoboTwin runtime is modified. Versions must equal the kernel module (570.158.01).
set -euo pipefail
SRC=/mnt/data/vmo-ai-task/anhdh35/robotwin_eval_runtime_20261030/nvidia570
DST=${EVAL_OUT:?}/egl_libs
grep -q "570.158.01" /proc/driver/nvidia/version || { echo "kernel module is not 570.158.01"; exit 1; }
mkdir -p "$DST"
V=570.158.01
for f in libEGL_nvidia.so.$V libnvidia-eglcore.so.$V libnvidia-glsi.so.$V libnvidia-gpucomp.so.$V libnvidia-glcore.so.$V \
         libnvidia-tls.so.$V libEGL.so.1.1.0 libGLdispatch.so.0 libOpenGL.so.0 libGL.so.1.7.0 libGLX.so.0; do
  cp -L "$SRC/$f" "$DST/$f"
done
ln -sfn libEGL_nvidia.so.$V "$DST/libEGL_nvidia.so.0"
ln -sfn libEGL.so.1.1.0 "$DST/libEGL.so.1"; ln -sfn libEGL.so.1 "$DST/libEGL.so"
ln -sfn libOpenGL.so.0 "$DST/libOpenGL.so"; ln -sfn libGL.so.1.7.0 "$DST/libGL.so.1"; ln -sfn libGL.so.1 "$DST/libGL.so"
cat > "$DST/10_nvidia.json" <<EOF
{"file_format_version": "1.0.0", "ICD": {"library_path": "$DST/libEGL_nvidia.so.0"}}
EOF
(cd "$DST" && sha256sum *.so* > SHA256SUMS)
echo "EGL libs in $DST:"; cat "$DST/SHA256SUMS"
