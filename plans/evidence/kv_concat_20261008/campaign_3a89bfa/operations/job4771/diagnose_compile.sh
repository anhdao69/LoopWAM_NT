set -euo pipefail
cd /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/source_kv_3a89bfa
source scripts/h100_env.sh
export PYTHONPATH="$PWD/src:$PWD:$PWD/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
for backend in inductor eager; do
python /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/diagnose_kv_compile.py --checkpoint /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4771/campaign_3a89bfa/mix_long_smoke_warm/latest.pt --output /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4771/diagnostic_mix_${backend}.json --backend "$backend"
done
python /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/diagnose_kv_compile.py --checkpoint /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/v0_v4a1_job4728_20261007/train/latest.pt --output /mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/loopwam_nt/kv_campaign_job4771/diagnostic_aligned_inductor.json
