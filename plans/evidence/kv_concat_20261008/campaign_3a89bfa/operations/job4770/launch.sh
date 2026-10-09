#!/usr/bin/env bash
set -euo pipefail
cd '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/source_kv_3a89bfa'
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LP_NUM_THREADS=3
exec python -u scripts/run_kv_campaign.py --queue concat --output-root '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/campaign_3a89bfa' --peer-root '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4771/campaign_3a89bfa' --phase run > '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/production_3a89bfa.log' 2>&1
