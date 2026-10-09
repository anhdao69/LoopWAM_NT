#!/bin/bash
set -euo pipefail
cd '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/source_kv_3a89bfa'
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LP_NUM_THREADS=3
[[ -f '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4771/campaign_3a89bfa/prepared.json' ]]
python /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/run_native_compile_with_provenance.py '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4771/campaign_3a89bfa/mix_long_smoke_warm/latest.pt' '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4771/campaign_3a89bfa/native_compiled.json'
touch '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4771/campaign_3a89bfa/native_compile_passed'
