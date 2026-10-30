#!/usr/bin/env bash
# Matched LIBERO eval of the four LiLa-WAM suite policies inside the user's interactive job 1102933 (2 H100, 64 CPUs).
# Launched by eval_login_watcher.sh from evuser2 with srun --overlap. Resumable: finished groups are skipped.
RUN=/groups/yshang/an221229/checkpoints/wam_baselines/lilawam/libero_ours_paper_recipe
OUT=$RUN/eval_matched
# the PAT (if the watcher was given one) is kept only for the final push; nothing else sees it
PUSH_TOKEN="${GH_TOKEN:-}"; unset GH_TOKEN
source /groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work/eval/eval_env.sh
export PYTHONPATH="/groups/yshang/an221229/envs/lilawam_venv/lib/python3.10/site-packages:$PYTHONPATH"
echo "[$(date -Is)] step start on $(hostname) job ${SLURM_JOB_ID} step ${SLURM_STEP_ID}; nproc $(nproc); CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
# wait for training to finish (it runs in job 1094056, which ends 12:51:40)
while ! grep -q "all done" $RUN/train_job.log && squeue -h -j 1094056 -o %T 2>/dev/null | grep -q RUNNING; do sleep 30; done
SUITES=""
for s in libero_10 libero_goal libero_object libero_spatial; do
  if ls $RUN/$s/stage2/sft_*/checkpoint_epoch_2.pt >/dev/null 2>&1; then SUITES="$SUITES,$s"; else echo "WARNING: $s has no final checkpoint; not evaluated"; fi
done
SUITES=${SUITES#,}
echo "[$(date -Is)] suites: $SUITES"
NG=$(nvidia-smi -L | wc -l); GPUS=$(seq -s, 0 $((NG - 1)))
NP=$(nproc); W=$(( NP > 12 ? NP - 8 : NP - 2 ))
for attempt in 1 2 3; do
  echo "[$(date -Is)] attempt $attempt: gpus $GPUS env workers $W"
  python /groups/yshang/an221229/code/lilawam_libero/evaluate_lilawam_matched.py --output-dir $OUT --gpus $GPUS \
    --env-workers $W --suites $SUITES && break
  echo "[$(date -Is)] attempt $attempt failed rc=$?"; sleep 30
done
if [ -f $OUT/summary.json ]; then
  echo "[$(date -Is)] EVAL COMPLETE"; python -c "import json; s=json.load(open('$OUT/summary.json')); print(json.dumps({k: s[k] for k in ('success_rate','successes','episodes')}), {k: round(100*v['success_rate'],2) for k,v in s['per_suite'].items()})"
  python $RUN/update_hf_readme.py || echo "HF model card update failed"
  bash /groups/yshang/an221229/checkpoints/wam_baselines/latency/finalize_to_repo.sh "LiLa-WAM (ours, paper recipe) matched LIBERO evaluation results" || echo "repo finalize failed"
  GH_TOKEN="$PUSH_TOKEN" bash /groups/yshang/an221229/checkpoints/wam_baselines/latency/push_repo.sh || echo "push failed"
fi
echo "[$(date -Is)] step end"
