#!/usr/bin/env bash
# Usage: run_models.sh <node tag> <model>...   models: fasterwam fastwam fastwam_nocompile lightwam
source /groups/yshang/an221229/checkpoints/wam_baselines/latency/env.sh
tag=$1; shift
F=/groups/yshang/an221229/checkpoints/FastWAM/fastwam_release
for m in "$@"; do
  case $m in
    fastwam) args=(--repo $CODE/FastWAM --name FastWAM --task libero_uncond_2cam224_1e-4 --override model.redirect_common_files=false
                   --ckpt $F/libero_uncond_2cam224.pt --stats $F/libero_uncond_2cam224_dataset_stats.json) ;;
    fastwam_nocompile) args=(--repo $CODE/FastWAM --name "FastWAM (compile_action_infer=false)" --task libero_uncond_2cam224_1e-4
                   --override model.redirect_common_files=false --override EVALUATION.compile_action_infer=false
                   --ckpt $F/libero_uncond_2cam224.pt --stats $F/libero_uncond_2cam224_dataset_stats.json) ;;
    fasterwam) args=(--repo $CODE/FasterWAM --name FasterWAM --task libero_fasterwam_2cam224_1e-4 --override model.redirect_common_files=false
                   --ckpt $CK/FasterWAM/libero/step_021700.pt --stats $CK/FasterWAM/libero/dataset_stats.json) ;;
    lightwam) args=(--repo $CODE/Light-WAM --name "Light-WAM (libero_10 release)"
                   --ckpt $CK/lightwam-checkpoints/lightwam_libero_10/checkpoints/weights/step_080000.pt
                   --stats $CK/lightwam-checkpoints/lightwam_libero_10/dataset_stats.json) ;;
  esac
  echo "[$(date -Is)] $m start on $(hostname) job ${SLURM_JOB_ID}"
  $PY $LAT/measure_public_latency.py "${args[@]}" --out $LAT/results/${m}_${tag}.json > $LAT/logs/${m}_${tag}.log 2>&1
  echo "[$(date -Is)] $m rc=$?"
done
