#!/usr/bin/env bash
set -euo pipefail
source /data14t/anhdh35/robotwin10_h100_20261106/v0-video4-action4/environment.sh
cd /storage/anhdh35/LoopWAM-robotwin10-dense-eval
exec python /data14t/anhdh35/robotwin10_h100_20261106/v0-video4-action4/scripts/queue_pipeline.py
