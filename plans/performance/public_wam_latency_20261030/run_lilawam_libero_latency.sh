#!/usr/bin/env bash
# After training ("all done") and the open-loop checks ("openloop all done") finish, time the trained libero_10
# policy on the 500 observations with the GPU otherwise idle. Runs on evc103 inside job 1094056.
RUN=/groups/yshang/an221229/checkpoints/wam_baselines/lilawam/libero_ours_paper_recipe
LAT=/groups/yshang/an221229/checkpoints/wam_baselines/latency
until grep -q "all done" $RUN/train_job.log && grep -q "openloop all done" $RUN/openloop/watcher.log; do sleep 15; done
D=$(ls -d $RUN/libero_10/stage2/sft_* | tail -n 1)
echo "[$(date -Is)] start; GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
cd $LAT && PYTHONPATH=/groups/yshang/an221229/envs/lilawam_venv/lib/python3.10/site-packages \
  /groups/yshang/an221229/envs/FastWAM/bin/python measure_lilawam_libero_latency.py --config $D/config.yaml \
  --ckpt $D/checkpoint_epoch_2.pt --stats /groups/yshang/an221229/data/LiLaWAM_libero_ours_stats_libero_10.json \
  --out $LAT/results/lilawam_libero_evc103.json --suite libero_10
echo "[$(date -Is)] rc=$?"
