#!/usr/bin/env bash
set -euo pipefail
cd '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/source_kv_3a89bfa'
source scripts/h100_env.sh
export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python -u '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/training_only_handoff_20261009/scripts/run_training_only_continuation.py' --phase guard --queue mix --predecessor-job 4771 --predecessor-root '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4771/campaign_3a89bfa' --handoff-dir '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/training_only_handoff_20261009/handoffs' --runner-sha256 a4316882fa7675706b4f5afe6b9bc80e2bc3a72ea89ee35d7183c4d066f3b339
