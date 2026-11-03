#!/usr/bin/env bash
# Reproduction gate (inside an overlapping srun step of job 5623): the published BASE10 E0 epoch_10.pt (sha256
# 5569c47f..., HF libero-all-suites/Elastic/E0) through this cluster's stack (same runner, same flags, private MPS
# daemon as in the production lanes) on three complete task/seed groups, compared episode by episode with the
# reference H100 records (HF libero-all-suites/Elastic/eval/conditions/E0__<schedule>/episodes.csv):
#   E0 [1,1]   libero_10 seed 42 task 0, libero_spatial seed 42 task 0   (GPUs 0, 1)
#   E0 [4]x10  libero_goal seed 44 task 5                                (GPU 2)
# Scored output goes to $EVAL_OUT/parity (tag parity_repro), never into final_rollout.
# Prompt cache: E0 was trained and scored on the other cluster with a regenerated cache (same files and valid tokens,
# different bytes; setup_eval.json), which is not available here. So --verify-text-hash is off for this run only and
# E0 reads this cluster's training cache (the BASE10 parent's bytes). An exact result therefore establishes the whole
# stack; an inexact one measures this cluster + prompt bytes together, as the RTX PRO 6000 batch's gate G4 did.
set -uo pipefail
TOOLS=/mnt/data/vmo-ai-task/anhdh35/LoopWAM_elastic_cl0/plans/elastic_cl0/eval
source $TOOLS/eval_env.sh
cd "$ELASTIC_ROOT"
HF=$EVAL_OUT/reference_hf/libero-all-suites/Elastic
REF=$EVAL_OUT/reference_runs/E0
mkdir -p "$REF/data" "$EVAL_OUT/parity" "$EVAL_OUT/logs"
for f in epoch_10.pt config.json dataset_stats.json data/data_manifest.json data/dataset_stats.json; do ln -sfn "$HF/E0/$f" "$REF/$f"; done
E0SHA=5569c47fa6c91e136a611d2b819a053c94f6b2956a7f602bbed4db23c69f60e8
M=/dev/shm/elastic_cl0_mps_${SLURM_JOB_ID:-x}_parity
rm -rf "$M"; mkdir -p "$M/pipe" "$M/log"
export CUDA_MPS_PIPE_DIRECTORY=$M/pipe CUDA_MPS_LOG_DIRECTORY=$M/log
( unset CUDA_VISIBLE_DEVICES; nvidia-cuda-mps-control -d ) && echo "[$(date -Is)] parity: MPS daemon started"
common=(--tag parity_repro --key-prefix elastic --expected-sha "E0=$E0SHA" --output-root "$EVAL_OUT/parity"
        --weights-root "$EVAL_OUT/reference_runs" --vae-path "$ELASTIC_VAE" --text-cache-dir "$EVAL_TEXT"
        --workers-per-gpu 1 --no-videos)
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_rt_matched.py "${common[@]}" --conditions E0:flat_k1_n2:cuda-graph \
  --suites libero_10,libero_spatial --tasks 0 --seeds 42 --gpus 0,1 > "$EVAL_OUT/logs/parity_k1n2.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate_rt_matched.py "${common[@]}" --conditions E0:flat_k4_n10:cuda-graph \
  --suites libero_goal --tasks 5 --seeds 44 --gpus 2 > "$EVAL_OUT/logs/parity_k4n10.log" 2>&1 &
p2=$!
wait $p1; r1=$?; wait $p2; r2=$?
echo quit | nvidia-cuda-mps-control >/dev/null 2>&1; sleep 3; rm -rf "$M"
echo "[$(date -Is)] parity runners rc=$r1,$r2"
python $TOOLS/compare_parity.py --parity-root "$EVAL_OUT/parity/parity_repro" --reference "$HF/eval/conditions" \
  --out "$EVAL_OUT/validation/parity_e0.json"
