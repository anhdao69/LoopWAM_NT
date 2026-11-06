#!/usr/bin/env bash
# RT50-ep5 (teacher = student init = RoboTwin-50 v0 4/4 epoch 5, = HF robotwin/v0-video4-action4/policy.pt, sha 38975220).
# ELASTIC_ROOT is the checkout this file lives in (the frozen production clone when run from there).
export ELASTIC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export ELASTIC_TEACHER=/mnt/data/vmo-ai-task/anhdh35/FastWAM/runs/robotwin_full_v0_20261028/release_v2/production/checkpoints/epoch_05.pt
export ELASTIC_INIT=$ERT_WORK/init/rt50_ep5_common_init_v1.pt
export ERT_FAMILY=RT50-ep5 ERT_EP5=$ERT_WORK/ep5
