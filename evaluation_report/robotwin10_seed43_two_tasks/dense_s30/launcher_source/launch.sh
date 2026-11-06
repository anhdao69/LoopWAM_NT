#!/usr/bin/env bash
set -euo pipefail
source /data14t/anhdh35/loopwam_robotwin10_seed43_two_tasks_20261103/dense_s30/environment.sh
cd /storage/anhdh35/LoopWAM-robotwin10-dense-eval
exec python /data14t/anhdh35/loopwam_robotwin10_seed43_two_tasks_20261103/dense_s30/scripts/robotwin_local_queue.py
