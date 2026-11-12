#!/usr/bin/env bash
# GR1-E0 x5 campaign (record dir $EGR_WORK) on GPUs 4-7 of job 5758: code = the frozen clone at $ELASTIC_ROOT.
export ELASTIC_ROOT=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_gr1
export EGR_WORK=/mnt/data/vmo-ai-task/anhdh35/elastic_gr1
source "$ELASTIC_ROOT/plans/elastic_gr1/env.sh"
export GR1E_RUN=GR1-E0 GR1E_EPOCHS=5 GR1E_PER_EPOCH=47032 GR1E_GPUS=4,5,6,7
export GR1_CONT=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101/continuation_20261106
export GR1_EVAL=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robocasa_gr1_v0_20261101/eval
export GR1E_EGL=/mnt/data/vmo-ai-task/anhdh35/elastic_cl0/eval/egl_libs
# eval schedules per epoch (user, 2026-11-08): epochs 1-4 [1,1]; epoch 5 [1,1], [4,4], [4]x5
gr1e_schedules() { if [ "$1" = 5 ]; then echo flat_k1_n2 flat_k4_n2 flat_k4_n5; else echo flat_k1_n2; fi; }
