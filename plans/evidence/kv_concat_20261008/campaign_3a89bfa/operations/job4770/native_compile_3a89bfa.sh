#!/bin/bash
set -euo pipefail
cd '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/source_kv_3a89bfa'
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LP_NUM_THREADS=3
[[ -f '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/campaign_3a89bfa/prepared.json' ]]
python /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/run_native_compile_with_provenance.py '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/campaign_3a89bfa/concat_long_smoke_warm/latest.pt' '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/campaign_3a89bfa/native_compiled.json'
python /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/run_native_compile_with_provenance.py '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_v4a1_job4728_20261007/train/latest.pt' '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/campaign_3a89bfa/native_compiled_aligned41.json'
python scripts/measure_policy_latency.py --loop-checkpoint '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_v4a1_job4728_20261007/train/latest.pt' --output '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/latency_aligned41_compiled.json' --trials 2 --samples 50 --warmup 5 --compiled
python /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/run_native_compile_with_provenance.py '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_scratch_fast_bs128_20261005/latest.pt' '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/campaign_3a89bfa/native_compiled_original44.json'
python scripts/measure_policy_latency.py --loop-checkpoint '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_scratch_fast_bs128_20261005/latest.pt' --output '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/latency_original44_compiled.json' --trials 2 --samples 50 --warmup 5 --compiled
touch '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/campaign_3a89bfa/native_compile_passed'
