#!/usr/bin/env bash
set -euo pipefail
source /media/vmo-perception/disk_2/anhdh35/loopwam_robotwin10_v0_4x4_20261105/environment.sh
cd /media/vmo-perception/disk_2/anhdh35/loopwam_robotwin10_v0_4x4_20261104/repo
exec python /media/vmo-perception/disk_2/anhdh35/loopwam_robotwin10_v0_4x4_20261105/scripts/queue_pipeline.py
