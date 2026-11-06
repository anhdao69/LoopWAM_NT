#!/usr/bin/env bash
source /mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_cl0/plans/elastic_cl0/env.sh
V=$CL0_WORK/validation; cd $ELASTIC_ROOT
CUDA_VISIBLE_DEVICES=0 python scripts/preflight_cl0.py --common-init $ELASTIC_INIT --runs CL0-E0,CL0-E1,CL0-E2 --options no_fill,all --out $V/preflight_cl0_nofill.json > $V/preflight2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/bench_cl0.py --common-init $ELASTIC_INIT --microbatch 32 --options bf16+graph+nofill --profile flat_k1_n5 --out $V/bench2_mb32.json > $V/bench2_mb32.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/bench_cl0.py --common-init $ELASTIC_INIT --microbatch 64 --ckpt-min-blocks 180 --options bf16+graph+nofill --out $V/bench2_mb64.json > $V/bench2_mb64.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python scripts/bench_cl0.py --run CL0-E2 --common-init $ELASTIC_INIT --microbatch 32 --options bf16+graph+nofill --out $V/bench2_mb32_E2.json > $V/bench2_mb32_E2.log 2>&1 &
wait
echo GPU_CHECKS2_DONE
