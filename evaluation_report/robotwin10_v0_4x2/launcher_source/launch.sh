#!/usr/bin/env bash
set -euo pipefail
source /data14t/anhdh35/loopwam_robotwin10_v0_4x2_20261104/environment.sh
cd /storage/anhdh35/LoopWAM-robotwin10-dense-eval
exec python /data14t/anhdh35/loopwam_robotwin10_v0_4x2_20261104/scripts/robotwin_local_queue.py
