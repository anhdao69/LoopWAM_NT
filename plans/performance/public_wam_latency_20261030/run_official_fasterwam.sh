#!/usr/bin/env bash
# FasterWAM's own architecture-only latency script (random init, synthetic inputs), unmodified, repo defaults.
source /groups/yshang/an221229/checkpoints/wam_baselines/latency/env.sh
cd $CODE/FasterWAM
echo "[$(date -Is)] official start on $(hostname) job ${SLURM_JOB_ID}"
PYTHONPATH=$CODE/FasterWAM/src:$CODE/FasterWAM:$PYTHONPATH $PY experiments/latency/measure_latency.py --output-dir $LAT/official_fasterwam_script > $LAT/logs/official_fasterwam_script.log 2>&1
echo "[$(date -Is)] official rc=$?"
