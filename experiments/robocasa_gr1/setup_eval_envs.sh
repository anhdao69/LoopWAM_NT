#!/usr/bin/env bash
# One-time setup of the RoboCasa-GR1 tabletop evaluation environments (login node; network only).
#   robocasa venv : sim client (robosuite v1.5.1 source + mink, robocasa-gr1-tabletop-tasks, starVLA eval
#                   client deps, CPU torch for tianshou). Renders with osmesa on CPU.
#   starvla venv  : reference policy server for StarVLA/Qwen3-VL-OFT-Robocasa (check (a)).
# Pinned sources live in $TP; nothing is installed into the shared FastWAM venv.
set -euo pipefail
ROOT=${GR1_EVAL_ROOT:-/mnt/data/vmo-ai-task/anhdh35/gr1_eval}
TP=$ROOT/third_party
STARVLA_REV=13d8ad43eb4128554c57227507055ecf9c3a62a2
ROBOCASA_REV=4840e671596f93ca03651524b9f72ffb1aadfeff
ROBOSUITE_TAG=v1.5.1
PY=${PY:-/usr/bin/python3.10}
mkdir -p "$TP" "$ROOT/envs"
export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=8 PIP_DISABLE_PIP_VERSION_CHECK=1

clone() {  # url dir rev
  if [ ! -d "$TP/$2/.git" ]; then git clone -q "$1" "$TP/$2"; fi
  git -C "$TP/$2" fetch -q --tags origin
  git -C "$TP/$2" checkout -q "$3"
  echo "$2 $(git -C "$TP/$2" rev-parse HEAD)"
}
clone https://github.com/ARISE-Initiative/robosuite.git robosuite "$ROBOSUITE_TAG"
clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git robocasa-gr1-tabletop-tasks "$ROBOCASA_REV"
clone https://github.com/starVLA/starVLA.git starVLA "$STARVLA_REV"

if [ "${SKIP_ROBOCASA:-0}" != 1 ]; then
  E=$ROOT/envs/robocasa
  [ -x "$E/bin/python" ] || "$PY" -m venv "$E"
  "$E/bin/pip" install -q -U pip "setuptools<81" wheel
  "$E/bin/pip" install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
  "$E/bin/pip" install -q -e "$TP/robosuite"
  "$E/bin/pip" install -q mink==0.0.5 robosuite_models
  "$E/bin/pip" install -q -e "$TP/robocasa-gr1-tabletop-tasks"
  # opencv 5.x requires numpy>=2; robocasa pins numpy 1.26.4, so use the last numpy-1 compatible headless wheel.
  "$E/bin/pip" uninstall -q -y opencv-python opencv-python-headless 2>/dev/null || true
  "$E/bin/pip" install -q numpy==1.26.4 mujoco==3.2.6 gymnasium==1.2.2 websockets==16.0 msgpack av matplotlib mediapy \
      opencv-python-headless==4.10.0.84 pydantic tyro typing_extensions websocket-client
  (cd "$TP/robocasa-gr1-tabletop-tasks" && "$E/bin/python" robocasa/scripts/download_tabletop_assets.py -y)
  "$E/bin/pip" freeze > "$ROOT/envs/robocasa.freeze.txt"
fi

if [ "${SKIP_STARVLA:-0}" != 1 ]; then
  E=$ROOT/envs/starvla
  [ -x "$E/bin/python" ] || "$PY" -m venv "$E"
  "$E/bin/pip" install -q -U pip "setuptools<81" wheel
  "$E/bin/pip" install -q torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
  "$E/bin/pip" install -q transformers==4.57.0 accelerate==1.5.2 qwen-vl-utils omegaconf einops tiktoken timm diffusers \
      numpy==1.26.4 pillow scipy pydantic==2.10.6 websockets msgpack tyro rich opencv-python-headless==4.10.0.84 matplotlib \
      huggingface_hub safetensors
  "$E/bin/pip" freeze > "$ROOT/envs/starvla.freeze.txt"
fi
echo SETUP_DONE
