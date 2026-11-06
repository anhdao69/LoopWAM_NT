#!/usr/bin/env bash
# RoboCasa-GR1 evaluation on ws_anhdh35, phase 1 (network + CPU only; no GPU use):
#   code/         LoopWAM_NT branch robocasa_gr1_v0 (git bundle from the Mac), must be 45ee0c0 (= H100 server checkout)
#   third_party/  robosuite v1.5.1 (51cc0178), robocasa-gr1-tabletop-tasks 4840e671, starVLA 13d8ad43 (= H100 pins)
#   envs/robocasa python3.10 venv with the exact H100 package set (robocasa.freeze.txt), CPU torch 2.7.1
#   assets        robocasa download_tabletop_assets.py -y (box.com kitchen assets + GR00T HF registries)
# Everything lives under a new directory; nothing outside it is modified.
set -euo pipefail
ROOT=/storage/anhdh35/robocasa_gr1_eval
TP=$ROOT/third_party
mkdir -p "$TP" "$ROOT/envs" "$ROOT/logs"
export PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=8 PIP_DISABLE_PIP_VERSION_CHECK=1
stage() { echo "=== $(date -u +%FT%TZ) $*"; }

stage code
if [ ! -d "$ROOT/code/.git" ]; then git clone -q -b robocasa_gr1_v0 "$ROOT/gr1_code.bundle" "$ROOT/code"; fi
[ "$(git -C "$ROOT/code" rev-parse HEAD)" = 45ee0c07791f59b1cabcdab125781ca17bed4f07 ] || { echo "code revision mismatch"; exit 1; }

clone() {  # url dir rev
  if [ ! -d "$TP/$2/.git" ]; then git clone -q "$1" "$TP/$2"; fi
  git -C "$TP/$2" fetch -q --tags origin
  git -C "$TP/$2" checkout -q "$3"
  [ "$(git -C "$TP/$2" rev-parse HEAD)" = "$4" ] || { echo "$2 revision mismatch"; exit 1; }
  echo "$2 $(git -C "$TP/$2" rev-parse HEAD)"
}
stage third_party
clone https://github.com/ARISE-Initiative/robosuite.git robosuite v1.5.1 51cc01785bab80ffeed20da15e67d7dd4140e76a
clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git robocasa-gr1-tabletop-tasks 4840e671596f93ca03651524b9f72ffb1aadfeff 4840e671596f93ca03651524b9f72ffb1aadfeff
clone https://github.com/starVLA/starVLA.git starVLA 13d8ad43eb4128554c57227507055ecf9c3a62a2 13d8ad43eb4128554c57227507055ecf9c3a62a2

stage robocasa_venv
E=$ROOT/envs/robocasa
[ -x "$E/bin/python" ] || /usr/bin/python3.10 -m venv "$E"
"$E/bin/pip" install -q -U pip "setuptools<81" wheel
"$E/bin/pip" install -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
"$E/bin/pip" install -q -e "$TP/robosuite"
"$E/bin/pip" install -q -e "$TP/robocasa-gr1-tabletop-tasks"
"$E/bin/pip" uninstall -q -y opencv-python opencv-python-headless 2>/dev/null || true
# exact H100 package set (editables and torch handled above)
grep -vE '^(-e |torch==)' "$ROOT/robocasa.freeze.txt" > "$ROOT/logs/robocasa.pins.txt"
"$E/bin/pip" install -q -r "$ROOT/logs/robocasa.pins.txt"
"$E/bin/pip" freeze > "$ROOT/envs/robocasa.freeze.txt"
"$E/bin/pip" check || true
python3 - "$ROOT/robocasa.freeze.txt" "$ROOT/envs/robocasa.freeze.txt" <<'EOF'
import sys
norm = lambda p: {l.split('#egg=')[-1] if l.startswith('-e ') else l.strip() for l in open(p) if l.strip()}
h, w = norm(sys.argv[1]), norm(sys.argv[2])
print("freeze identical to H100" if h == w else f"FREEZE DIFF only_h100={sorted(h - w)} only_ws={sorted(w - h)}")
EOF

stage assets
(cd "$TP/robocasa-gr1-tabletop-tasks" && "$E/bin/python" robocasa/scripts/download_tabletop_assets.py -y)
echo PHASE1_DONE
