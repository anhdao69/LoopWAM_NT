set -euo pipefail
cd /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/source_kv_3a89bfa
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
for backend in inductor eager; do
python /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/diagnose_kv_compile.py --checkpoint /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/campaign_3a89bfa/concat_long_smoke_warm/latest.pt --output /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4770/diagnostic_concat_${backend}.json --backend "$backend"
done
