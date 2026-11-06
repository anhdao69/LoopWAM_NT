#!/usr/bin/env bash
set -euo pipefail
source /data14t/anhdh35/loopwam_robotwin10_v0_4x4_20261103/environment.sh
cd /storage/anhdh35/LoopWAM-robotwin10-dense-eval
exec python /data14t/anhdh35/loopwam_robotwin10_v0_4x4_20261103/scripts/robotwin_local_queue.py
