#!/bin/bash
set -euo pipefail
cd '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/source_kv_3a89bfa'
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LP_NUM_THREADS=3
[[ -f '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_evaluator_diagnosis_20261008/reproduction_passed.json' ]]
[[ -f '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4771/control_aligned_v4a1_accepted/summary.json' ]]
[[ -f '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/control_repeat_v4a4_grouped/summary.json' ]]
python -m pytest tests -q > plans/evidence/kv_concat_20261008/gpu_3a89bfa_4771.log 2>&1
touch plans/evidence/kv_concat_20261008/gpu_3a89bfa_4771_passed
for ((i=0;i<60;i++)); do
 if [[ -f plans/evidence/kv_concat_20261008/cpu_3a89bfa_passed ]]; then break; fi
 sleep 5
done
[[ -f plans/evidence/kv_concat_20261008/cpu_3a89bfa_passed ]]
python scripts/run_kv_campaign.py --queue 'mix' --output-root '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4771/campaign_3a89bfa' --peer-root '/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/campaign_3a89bfa' --phase prepare --eval-workers 5 --eval-render-threads 3
