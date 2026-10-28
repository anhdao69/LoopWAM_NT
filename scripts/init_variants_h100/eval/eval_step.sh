#!/usr/bin/env bash
# Inside an srun step: matched 6,000-episode evaluation of each run in RUNS (default core_random then ffn_magnitude)
# on GPUS with WPG workers per GPU. Several steps/jobs may run this at once: task/seed groups are claimed atomically.
# A run is evaluated only after its training is complete and fairness_h100 passed, and after its CUDA-graph gate.
set -uo pipefail
E=/groups/yshang/an221229/checkpoints/LoopWAM_NT/v0_init_variants_work/eval
source $E/eval_env.sh || exit 1
cd $E
GPUS=${GPUS:?}; WPG=${WPG:-4}; RUNS=${RUNS:-"core_random ffn_magnitude"}
echo "$(date) eval step job=$SLURM_JOB_ID step=$SLURM_STEP_ID host=$(hostname) GPUS=$GPUS WPG=$WPG RUNS=$RUNS"
gpus_free() {  # every requested GPU below 2 GB used
    local g; for g in ${GPUS//,/ }; do
        [[ $(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits) -lt 2000 ]] || return 1
    done
}
ready() {
    python - "$EVAL_RUNS/$1" <<'PY'
import json, sys, pathlib
run = pathlib.Path(sys.argv[1])
try:
    ok = (json.loads((run / 'train' / 'timing.json').read_text())['status'] == 'complete'
          and json.loads((run / 'fairness_h100.json').read_text())['h100']['h100_passed'] is True)
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
}
gate_passed() {
    python -c "import json,sys; sys.exit(0 if json.load(open('$E/logs/gate_$1.json'))['all_bit_exact'] is True else 1)" 2>/dev/null
}
for run in $RUNS; do
    until ready $run; do echo "$(date) waiting for $run training to complete and pass fairness"; sleep 120; done
    until gpus_free; do echo "$(date) waiting for GPUs $GPUS to be free"; sleep 60; done
    common=(--run $run --checkpoint $EVAL_RUNS/$run/train/latest.pt --vae-path $EVAL_VAE --text-cache-dir $EVAL_TEXT
            --output-dir $EVAL_RUNS/$run/eval_matched)
    if ! gate_passed $run; then
        first=${GPUS%%,*}
        CUDA_VISIBLE_DEVICES=$first MUJOCO_EGL_DEVICE_ID=$first python evaluate_v0_matched.py "${common[@]}" \
            --gate --gate-queries 8 --gate-output $E/logs/gate_$run.json > $E/logs/gate_${run}_${SLURM_JOB_ID}_$first.log 2>&1
        gate_passed $run || { echo "$(date) CUDA graph gate FAILED for $run; not evaluating it"; continue; }
    fi
    for attempt in 1 2 3; do
        echo "$(date) $run runner attempt $attempt"
        python evaluate_v0_matched.py "${common[@]}" --gpus $GPUS --workers-per-gpu $WPG --no-videos && break
        echo "$(date) $run runner attempt $attempt failed"; sleep 60
    done
done
echo "$(date) eval step finished"
