#!/usr/bin/env bash
set -euo pipefail
source /media/newdisk/loopwam_robotwin10_v0_4x4_20261104/environment.sh
cd /media/newdisk/loopwam_robotwin10_v0_4x4_20261104/repo
exec python /media/newdisk/loopwam_robotwin10_v0_4x4_20261104/scripts/queue_pipeline.py
