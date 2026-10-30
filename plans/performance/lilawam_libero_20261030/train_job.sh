#!/usr/bin/env bash
# LiLa-WAM on OUR LIBERO data, following the paper's LIBERO recipe (arXiv 2608.03701 Sec. 4.1-4.2):
# one policy per suite; third-person + wrist views at 256x256; chunk 12; AdamW(0.9,0.99, wd 0.01), batch 128,
# clip 1.0, lambda_ff 0.5, logit-normal t; stage-wise cosine LR: stage 1 peak 2e-4, stage 2 peak 5e-5 (public
# train.py --init_from = new stage, fresh schedule). Not stated in the paper, chosen here: epochs per stage 5 + 2,
# stage-1 floor 5e-5 (public config), stage-2 floor 5e-6, side-by-side view fusion.
# Runs inside the user's interactive job 1094056; the srun client lives on evuser2.
export PYTHONPATH=/groups/yshang/an221229/envs/lilawam_venv/lib/python3.10/site-packages
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
P=/groups/yshang/an221229/envs/FastWAM/bin/python
T=/groups/yshang/an221229/code/lilawam_libero/train_libero.py
RUN=/groups/yshang/an221229/checkpoints/wam_baselines/lilawam/libero_ours_paper_recipe
echo "[$(date -Is)] start on $(hostname) job ${SLURM_JOB_ID} step ${SLURM_STEP_ID}"
for s in libero_10 libero_goal libero_object libero_spatial; do
  ST=/groups/yshang/an221229/data/LiLaWAM_libero_ours_stats_$s.json
  mkdir -p $RUN/$s && cd $RUN/$s
  echo "[$(date -Is)] $s stage1 start"
  $P $T --config $RUN/configs/${s}_stage1.yaml --norm_stats_path $ST --save_dir $RUN/$s/stage1 > $RUN/logs/${s}_stage1.log 2>&1
  rc=$?; C1=$(ls -t $RUN/$s/stage1/sft_*/checkpoint_epoch_5.pt 2>/dev/null | head -n 1)
  echo "[$(date -Is)] $s stage1 rc=$rc ckpt=$C1"
  [ -f "$C1" ] || continue
  $P $T --config $RUN/configs/${s}_stage2.yaml --norm_stats_path $ST --save_dir $RUN/$s/stage2 --init_from $C1 > $RUN/logs/${s}_stage2.log 2>&1
  rc=$?; C2=$(ls -t $RUN/$s/stage2/sft_*/checkpoint_epoch_2.pt 2>/dev/null | head -n 1)
  echo "[$(date -Is)] $s stage2 rc=$rc ckpt=$C2"
done
echo "[$(date -Is)] all done"
