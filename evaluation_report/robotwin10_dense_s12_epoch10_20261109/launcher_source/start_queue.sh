#!/bin/bash
set -euo pipefail
source /data14t/anhdh35/loopwam_robotwin10_dense_s12_epoch10_20261109/environment.sh
cd /storage/anhdh35/LoopWAM-robotwin10-dense-eval
exec /data14t/anhdh35/loopwam_robotwin10_dense_s12_epoch10_20261109/env/bin/python /data14t/anhdh35/loopwam_robotwin10_dense_s12_epoch10_20261109/scripts/queue_pipeline.py
