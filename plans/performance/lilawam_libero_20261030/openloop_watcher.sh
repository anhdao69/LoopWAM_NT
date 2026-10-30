#!/usr/bin/env bash
# For each suite, once train_job.sh reports its stage 2, run openloop_check.py on the stage-1 final and stage-2
# final checkpoints (512 held-in samples, seed 0). Runs on evc103 inside job 1094056, shares the GPU with training.
export PYTHONPATH=/groups/yshang/an221229/envs/lilawam_venv/lib/python3.10/site-packages
RUN=/groups/yshang/an221229/checkpoints/wam_baselines/lilawam/libero_ours_paper_recipe
P=/groups/yshang/an221229/envs/FastWAM/bin/python; C=/groups/yshang/an221229/code/lilawam_libero/openloop_check.py
cd $RUN/openloop
for s in libero_10 libero_goal libero_object libero_spatial; do
  until grep -q "\] $s stage2 rc=" $RUN/train_job.log; do sleep 30; done
  ST=/groups/yshang/an221229/data/LiLaWAM_libero_ours_stats_$s.json
  for stage in stage1 stage2; do
    D=$(ls -d $RUN/$s/$stage/sft_* 2>/dev/null | tail -n 1); CK=$(ls -t $D/checkpoint_epoch_*.pt 2>/dev/null | head -n 1)
    [ -f "$CK" ] || { echo "[$(date -Is)] $s $stage no checkpoint"; continue; }
    echo "[$(date -Is)] $s $stage $CK"
    $P $C --config $D/config.yaml --ckpt $CK --stats $ST --out $RUN/openloop/${s}_${stage}.json --n 512 > $RUN/openloop/${s}_${stage}.log 2>&1
    echo "[$(date -Is)] $s $stage rc=$?"; grep -E "^(raw|bspline|baseline)" $RUN/openloop/${s}_${stage}.log
  done
done
echo "[$(date -Is)] openloop all done"
