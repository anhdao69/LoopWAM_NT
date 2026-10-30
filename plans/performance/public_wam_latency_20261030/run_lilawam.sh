#!/usr/bin/env bash
# Usage: run_lilawam.sh <node tag>  -- LiLa-WAM RoboTwin-release latency (FastWAM env + transformers 5.0.0rc0 overlay)
source /groups/yshang/an221229/checkpoints/wam_baselines/latency/env.sh
export PYTHONPATH=/groups/yshang/an221229/envs/lilawam_venv/lib/python3.10/site-packages:$PYTHONPATH
echo "[$(date -Is)] lilawam start on $(hostname) job ${SLURM_JOB_ID}"
$PY $LAT/measure_lilawam_latency.py --out $LAT/results/lilawam_$1.json > $LAT/logs/lilawam_$1.log 2>&1
echo "[$(date -Is)] lilawam rc=$?"
