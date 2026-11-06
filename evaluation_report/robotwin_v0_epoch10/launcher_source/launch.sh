#!/usr/bin/env bash
set -euo pipefail
source /data14t/anhdh35/loopwam_robotwin_v0_epoch10_20261106/environment.sh
cd /storage/anhdh35/LoopWAM-robotwin-eval
exec python /data14t/anhdh35/loopwam_robotwin_v0_epoch10_20261106/scripts/robotwin_local_queue.py
