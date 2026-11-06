#!/usr/bin/env bash
# CL-0 campaign GPU validation, one process per GPU of job 5623.
source /mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_cl0/plans/elastic_cl0/env.sh
V=$CL0_WORK/validation; cd $ELASTIC_ROOT
nvidia-smi -L > $V/gpus.txt
CUDA_VISIBLE_DEVICES=0 python scripts/preflight_cl0.py --common-init $ELASTIC_INIT --runs CL0-E0,CL0-E2 --out $V/preflight_cl0.json > $V/preflight.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_chrono_ro/src python $ELASTIC_ROOT/scripts/check_cl0_equivalence.py \
  --chronoloop $CL0_WORK/assets/CL-0_epoch_10/policy.pt --converted $ELASTIC_TEACHER --vae-path $ELASTIC_VAE --out $V/equivalence.json > $V/equivalence.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/bench_cl0.py --common-init $ELASTIC_INIT --microbatch 32 --options eager,bf16,bf16+graph --profile flat_k1_n5 --out $V/bench_mb32.json > $V/bench_mb32.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/bench_cl0.py --common-init $ELASTIC_INIT --microbatch 64 --ckpt-min-blocks 180 --options eager,bf16+graph --out $V/bench_mb64.json > $V/bench_mb64.log 2>&1 &
wait
echo GPU_CHECKS_DONE
